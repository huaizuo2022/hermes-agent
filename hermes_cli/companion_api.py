import os
import json
import shutil
import queue
import threading
from typing import Dict, Any, Generator, Optional
from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

router = APIRouter(prefix="/companion/v1")

class ChatRequest(BaseModel):
    user_id: str
    character_id: str
    message_id: str
    user_message: str
    character_profile: Dict[str, Any]
    stream: bool = True
    provider: Optional[str] = None
    model: Optional[str] = None
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    directives: Optional[str] = None
    request_overrides: Optional[Dict[str, Any]] = None

class MemorySyncRequest(BaseModel):
    target: str
    action: str
    content: Optional[str] = None
    old_text: Optional[str] = None

def get_profile_path(session_id: str) -> str:
    from hermes_constants import get_default_hermes_root
    return str(get_default_hermes_root() / "profiles" / session_id)

def sync_soul_file(profile_dir: str, profile_data: Dict[str, Any]) -> None:
    soul_path = os.path.join(profile_dir, "SOUL.md")
    os.makedirs(os.path.dirname(soul_path), exist_ok=True)
    
    name = str(profile_data.get("name", "Companion"))
    
    # 兼容处理 speaking_style 可能是 List[str] 或 Dict[str, str] 的情况
    speaking_style = profile_data.get("speaking_style", "")
    if isinstance(speaking_style, list):
        speaking_style = ", ".join(speaking_style)
    elif isinstance(speaking_style, dict):
        speaking_style = ", ".join("{}: {}".format(k, v) for k, v in speaking_style.items())
        
    personality = str(profile_data.get("personality", "")).strip()
    background = str(profile_data.get("background", "")).strip()
    
    # 新增扩展字段处理
    behavior_tags = profile_data.get("behavior_tags") or ""
    if isinstance(behavior_tags, list):
        behavior_tags = ", ".join(behavior_tags)
    behavior_tags = str(behavior_tags).strip()
    
    appearance_details = str(profile_data.get("appearance_details") or "").strip()
    initial_scenario = str(profile_data.get("initial_scenario") or "").strip()
    
    sample_dialogues = profile_data.get("sample_dialogues") or []
    dialogue_text = ""
    if isinstance(sample_dialogues, list):
        dialogue_lines = []
        for item in sample_dialogues:
            if isinstance(item, dict):
                user_val = item.get("user") or item.get("user_input")
                char_val = item.get("character") or item.get("character_response")
                if user_val and char_val:
                    dialogue_lines.append("User: {}".format(user_val))
                    dialogue_lines.append("{}: {}".format(name, char_val))
                    dialogue_lines.append("")
        dialogue_text = "\n".join(dialogue_lines).strip()

    # Parse relationship metrics
    rel_data = profile_data.get("relationship") or {}
    relationship_section = ""
    if rel_data:
        stage = rel_data.get("relationship_stage")
        intimacy = rel_data.get("intimacy_score")
        trust = rel_data.get("trust_score")
        nickname = rel_data.get("preferred_nickname")
        profile = rel_data.get("persona_profile")
        constraints = rel_data.get("persona_prompt_constraints")
        
        stage_names = {
            "stranger": "stranger (初次相遇，礼貌疏离)",
            "acquaintance": "acquaintance (逐渐熟悉，互动自然)",
            "ambiguous": "ambiguous (暧昧期，关系推拉有张力)",
            "early_relationship": "early_relationship (热恋陪伴，甜蜜依恋)",
            "soulmate": "soulmate (灵魂伴侣，深度信任与脆弱感)",
            "deep_relationship": "deep_relationship (长线羁绊，生命中不可或缺)"
        }
        stage_desc = stage_names.get(stage, stage)
        
        parts = ["## Relationship with User"]
        if stage_desc:
            parts.append("- Current Stage: {}".format(stage_desc))
        if intimacy is not None:
            parts.append("- Intimacy level: {}/10".format(intimacy))
        if trust is not None:
            parts.append("- Trust level: {}/10".format(trust))
        if nickname:
            parts.append("- Preferred Nickname: {} (When referring to the user, you MUST call them by this nickname)".format(nickname))
        if profile:
            parts.append("- User Profile: {} (Your perception of the user's role/background in your life)".format(profile))
        if constraints:
            parts.append("- Prompt Constraints: {}".format(constraints))
            
        relationship_section = "\n".join(parts)

    content_parts = [
        "# {}".format(name),
        "## Personality\n{}".format(personality) if personality else "",
        "## Behavior Tags\n{}".format(behavior_tags) if behavior_tags else "",
        "## Speaking Style\n{}".format(speaking_style) if speaking_style else "",
        "## Appearance\n{}".format(appearance_details) if appearance_details else "",
        "## Scenario\n{}".format(initial_scenario) if initial_scenario else "",
        "## Background\n{}".format(background) if background else "",
        "## Sample Dialogues\n{}".format(dialogue_text) if dialogue_text else "",
        relationship_section if relationship_section else "",
    ]
    
    # 过滤掉空部分，并以两个换行符连接
    content = "\n\n".join([p for p in content_parts if p]) + "\n"
    
    # 仅在内容有差异时覆写，优化缓存
    if not os.path.exists(soul_path) or open(soul_path, "r", encoding="utf-8").read() != content:
        with open(soul_path, "w", encoding="utf-8") as f:
            f.write(content)

@router.post("/chat")
async def chat_endpoint(req: ChatRequest):
    session_id = "savana_{}_{}".format(req.user_id.lower(), req.character_id.lower())
    profile_dir = get_profile_path(session_id)
    
    # 1. 动态物理隔离 Profile 目录 (使用线程安全的 ContextVar 覆盖)
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    token = set_hermes_home_override(profile_dir)
    
    try:
        sync_soul_file(profile_dir, req.character_profile)

        # 2. 导入与运行 AIAgent
        from hermes_state import SessionDB
        from run_agent import AIAgent

        # 初始化本地 state.db
        db_path = Path(profile_dir) / "state.db"
        session_db = SessionDB(db_path=db_path)
        
        # 确保 session 存在
        try:
            session_db.create_session(session_id, "savana")
        except Exception:
            pass

        # 检查消息 ID 是否存在，若不存在则双写落库
        msg_exists = False
        try:
            conn = session_db._conn
            cursor = conn.cursor()
            cursor.execute(
                "SELECT 1 FROM messages WHERE session_id = ? AND platform_message_id = ?",
                (session_id, req.message_id)
            )
            if cursor.fetchone():
                msg_exists = True
        except Exception:
            pass

        if not msg_exists:
            try:
                session_db.append_message(
                    session_id=session_id,
                    role="user",
                    content=req.user_message,
                    platform_message_id=req.message_id
                )
            except Exception:
                pass

        # 动态解析模型及 API 配置
        api_key = req.api_key or os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")
        base_url = req.api_base or os.environ.get("DEEPSEEK_API_BASE") or "https://api.deepseek.com"
        provider = req.provider or "deepseek"
        model = req.model or "deepseek-v4-flash"

        # 3. 实例化 AI 代理 (硬编码工具限制为 memory，完全封死危险操作)
        agent = AIAgent(
            provider=provider,
            model=model,
            api_key=api_key,
            base_url=base_url,
            enabled_toolsets=["memory"],
            quiet_mode=True,
            platform="savana",
            session_db=session_db,
            load_soul_identity=True,
            skip_context_files=True,
            ephemeral_system_prompt=req.directives,
            request_overrides=req.request_overrides,
        )
        agent.suppress_status_output = True

        if req.stream:
            q = queue.Queue()

            def run_chat_thread():
                token_thread = set_hermes_home_override(profile_dir)
                try:
                    collected_chunks = []
                    
                    def stream_callback(delta: str) -> None:
                        q.put(delta)
                        collected_chunks.append(delta)

                    # 触发对话生成
                    final_reply = agent.chat(req.user_message, stream_callback=stream_callback)
                    
                    # 最终记录 AI 的回复落库
                    try:
                        session_db.append_message(
                            session_id=session_id,
                            role="assistant",
                            content=final_reply
                        )
                    except Exception:
                        pass
                except Exception as e:
                    q.put(e)
                finally:
                    q.put(None)  # 哨兵标记，代表生成结束
                    reset_hermes_home_override(token_thread)

            threading.Thread(target=run_chat_thread).start()

            def sse_generator() -> Generator[str, None, None]:
                while True:
                    item = q.get()
                    if item is None:
                        break
                    if isinstance(item, Exception):
                        # 如果出现异常，返回 error 事件，并中断
                        yield "event: error\ndata: {}\n\n".format(json.dumps({"detail": str(item)}))
                        break
                    yield_data = {"delta": item}
                    yield "event: token\ndata: {}\n\n".format(json.dumps(yield_data))
                
                modifications = []
                if getattr(agent, "_memory_store", None) and hasattr(agent._memory_store, "modifications"):
                    modifications = agent._memory_store.modifications
                metadata = {
                    "session_id": session_id,
                    "status": "completed",
                    "memory_modifications": modifications
                }
                yield "event: metadata\ndata: {}\n\n".format(json.dumps(metadata))

            return StreamingResponse(sse_generator(), media_type="text/event-stream")
        else:
            reply = agent.chat(req.user_message)
            try:
                session_db.append_message(
                    session_id=session_id,
                    role="assistant",
                    content=reply
                )
            except Exception:
                pass
            modifications = []
            if getattr(agent, "_memory_store", None) and hasattr(agent._memory_store, "modifications"):
                modifications = agent._memory_store.modifications
            return JSONResponse({
                "reply": reply,
                "session_id": session_id,
                "memory_modifications": modifications
            })
            
    finally:
        reset_hermes_home_override(token)

@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    profile_dir = get_profile_path(session_id)
    if os.path.exists(profile_dir):
        # 使用 shutil.rmtree 删除物理 Profile 目录
        shutil.rmtree(profile_dir)
        return JSONResponse({"success": True, "message": "Session directory removed"})
    raise HTTPException(status_code=404, detail="Session profile not found")

@router.post("/sessions/{session_id}/memories")
async def sync_memory(session_id: str, req: MemorySyncRequest):
    profile_dir = get_profile_path(session_id)
    os.makedirs(profile_dir, exist_ok=True)
    
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    token = set_hermes_home_override(profile_dir)
    try:
        from tools.memory_tool import MemoryStore, memory_tool
        store = MemoryStore()
        store.load_from_disk()
        
        result_str = memory_tool(
            action=req.action,
            target=req.target,
            content=req.content,
            old_text=req.old_text,
            store=store
        )
        result = json.loads(result_str)
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error"))
        return result
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        reset_hermes_home_override(token)

@router.delete("/sessions/{session_id}/messages/{message_id}")
async def delete_message(session_id: str, message_id: str):
    profile_dir = get_profile_path(session_id)
    
    # 动态覆盖 HOME 路径
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    token = set_hermes_home_override(profile_dir)
    
    try:
        from hermes_state import SessionDB
        db_path = Path(profile_dir) / "state.db"
        session_db = SessionDB(db_path=db_path)
        
        conn = session_db._conn
        cursor = conn.cursor()
        
        # 1. 查找对应 platform_message_id 消息的自增 id
        cursor.execute(
            "SELECT id FROM messages WHERE session_id = ? AND platform_message_id = ?",
            (session_id, message_id)
        )
        row = cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Message not found in local db")
            
        msg_internal_id = row[0]
        
        # 2. 删除该消息及之后的所有消息 (撤回此消息及其产生的回复)
        cursor.execute(
            "DELETE FROM messages WHERE session_id = ? AND id >= ?",
            (session_id, msg_internal_id)
        )
        conn.commit()
        return JSONResponse({"success": True, "message": "Message and subsequent replies deleted from local db"})
        
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        reset_hermes_home_override(token)
