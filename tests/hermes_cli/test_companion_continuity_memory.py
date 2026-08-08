from pathlib import Path

import pytest
from starlette.testclient import TestClient

from hermes_cli.companion_turn_guard import review_turn as real_review_turn
from hermes_cli.web_server import app


CONTINUITY_FACT = "角色已答应下周陪用户去看展"
CONTINUITY_QUOTE = "下周陪你去看展"
ASSISTANT_REPLY = "我答应你，下周陪你去看展。"
MEMORY_TOOL_DESCRIPTION = "将用户的个人信息永久写入记忆"


def _payload(message_id, user_message, stream=False):
    return {
        "user_id": "UserA",
        "character_id": "CharA",
        "message_id": message_id,
        "user_message": user_message,
        "stream": stream,
        "api_key": "test-key",
        "api_base": "https://example.invalid/v1",
        "provider": "test-provider",
        "model": "test-model",
        "character_profile": {
            "name": "测试角色",
            "personality": "温柔",
            "speaking_style": "自然",
        },
    }


def _review_result(turn_id, assistant_text):
    from hermes_cli.companion_turn_guard import RESULT_END, RESULT_START, assistant_sha256

    payload = {
        "turn_id": turn_id,
        "assistant_sha256": assistant_sha256(assistant_text),
        "style_decision": "clean",
        "style_reason": "角色与场景一致。",
        "continuity_summary": "角色答应下周陪用户去看展。",
        "memory_operations": [],
        "continuity_operations": [
            {
                "target": "continuity",
                "action": "add",
                "content": CONTINUITY_FACT,
                "evidence_quote": CONTINUITY_QUOTE,
            }
        ],
        "self_review": {
            "fits_character_and_scene": "pass",
            "no_technical_false_positive": "pass",
            "summary_preserves_facts": "pass",
            "summary_adds_no_new_facts": "pass",
        },
        "verdict": "pass",
    }
    import json

    return RESULT_START + json.dumps(payload, ensure_ascii=False) + RESULT_END


def _build_fake_agent(captures, reply=None, tool_description=None):
    class FakeAgent:
        def __init__(self, **kwargs):
            self.session_db = kwargs["session_db"]
            self.session_id = kwargs["session_id"]
            self._memory_store = None
            self.tools = []
            if tool_description is not None:
                self.tools = [
                    {
                        "type": "function",
                        "function": {
                            "name": "memory",
                            "description": tool_description,
                        },
                    }
                ]
            self.suppress_status_output = False
            self.model = kwargs["model"]
            self.provider = kwargs["provider"]
            self.base_url = kwargs["base_url"]
            self.ephemeral_system_prompt = kwargs.get("ephemeral_system_prompt")
            self._system_prompt_override = None

        def run_conversation(
            self,
            user_message,
            system_message=None,
            conversation_history=None,
            task_id=None,
            stream_callback=None,
            persist_user_message=None,
            platform_message_id=None,
        ):
            captures["histories"].append([dict(item) for item in (conversation_history or [])])
            captures["overrides"].append(self._system_prompt_override)
            captures["tool_descriptions"].append(
                self.tools[0]["function"]["description"] if self.tools else None
            )
            final_reply = reply if reply is not None else "收到"
            self.session_db.append_message(
                session_id=self.session_id,
                role="user",
                content=user_message,
                platform_message_id=platform_message_id,
            )
            self.session_db.append_message(
                session_id=self.session_id,
                role="assistant",
                content=final_reply,
            )
            return {"final_response": final_reply}

        def close(self):
            pass

    return FakeAgent


def _patch_profile_env(monkeypatch, tmp_path, profile_name):
    home = tmp_path / "home"
    hermes_home = home / ".hermes" / profile_name
    hermes_home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(Path, "home", lambda: home)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    return hermes_home


@pytest.mark.skip(reason="turn guard review 已停用（实测全局 0% 成功率）；continuity 现由主 AI 直接写（memory_tool 解封 target=continuity），不再由 review 写。见 companion_api.py: style_guard_enabled=False")
def test_companion_continuity_memory_persists_and_replays_across_requests(monkeypatch, tmp_path):
    from tools import memory_tool

    profile_dir = _patch_profile_env(monkeypatch, tmp_path, "continuity-profile")
    monkeypatch.setattr("hermes_cli.companion_api.get_profile_path", lambda _sid: str(profile_dir))
    monkeypatch.setattr("hermes_cli.companion_api._schedule_unresolved_reviews", lambda **kwargs: None)

    captures = {"overrides": [], "histories": [], "tool_descriptions": []}
    monkeypatch.setattr("run_agent.AIAgent", _build_fake_agent(captures, reply=ASSISTANT_REPLY))

    def boundary_review(**kwargs):
        memory_store = kwargs.get("memory_store") or memory_tool.MemoryStore()
        if kwargs.get("memory_store") is None:
            memory_store.load_from_disk()
        review = real_review_turn(
            profile_dir=kwargs["profile_dir"],
            turn_id=kwargs["turn_id"],
            assistant_text=kwargs["assistant_text"],
            user_message=kwargs["user_message"],
            messages=kwargs["messages"],
            provider=kwargs["provider"],
            model=kwargs["model"],
            base_url=kwargs.get("base_url"),
            api_key=kwargs.get("api_key"),
            memory_store=memory_store,
            call_llm_fn=lambda **_llm_kwargs: _review_result(kwargs["turn_id"], kwargs["assistant_text"]),
        )
        return {
            "turn_id": review["turn_id"],
            "review_status": review["review_status"],
            "memory_status": review["memory_status"],
            "memory_modifications": review["memory_modifications"],
            "review_result": review.get("review_result"),
        }

    monkeypatch.setattr("hermes_cli.companion_api.review_turn", boundary_review)

    client = TestClient(app)
    first = client.post("/companion/v1/chat", json=_payload("msg-1", "继续吧"))

    continuity_path = Path(profile_dir) / "memories" / "CONTINUITY.md"
    assert first.status_code == 200
    assert continuity_path.exists()
    assert continuity_path.read_text(encoding="utf-8") == CONTINUITY_FACT
    assert first.json()["memory_status"] == "applied"
    assert first.json()["memory_modifications"][0]["success"] is True
    assert first.json()["memory_modifications"][0]["target"] == "continuity"
    assert CONTINUITY_FACT in first.json()["memory_modifications"][0]["entries"]

    state_db = Path(profile_dir) / "state.db"
    if state_db.exists():
        state_db.unlink()

    second_payload = _payload("msg-2", "新请求", stream=False)
    second = client.post("/companion/v1/chat", json=second_payload)

    assert second.status_code == 200
    assert captures["histories"][1] == []
    assert "角色连续性记忆" in captures["overrides"][1]
    assert CONTINUITY_FACT in captures["overrides"][1]


@pytest.mark.skip(reason="turn guard review 已停用（实测全局 0% 成功率）；continuity 现由主 AI 直接写（memory_tool 解封 target=continuity），不再由 review 写。见 companion_api.py: style_guard_enabled=False")
def test_companion_user_memory_behavior_remains_user_only(monkeypatch, tmp_path):
    profile_dir = _patch_profile_env(monkeypatch, tmp_path, "user-profile")
    memories_dir = profile_dir / "memories"
    memories_dir.mkdir(parents=True, exist_ok=True)
    (profile_dir / "SOUL.md").write_text("角色 SOUL\n测试角色", encoding="utf-8")
    (memories_dir / "USER.md").write_text("用户喜欢深夜调试", encoding="utf-8")
    (memories_dir / "CONTINUITY.md").write_text(CONTINUITY_FACT, encoding="utf-8")
    monkeypatch.setattr("hermes_cli.companion_api.get_profile_path", lambda _sid: str(profile_dir))
    monkeypatch.setattr("hermes_cli.companion_api._schedule_unresolved_reviews", lambda **kwargs: None)
    monkeypatch.setattr(
        "hermes_cli.companion_api.review_turn",
        lambda **kwargs: {
            "turn_id": kwargs["turn_id"],
            "review_status": "clean",
            "memory_status": "none",
            "memory_modifications": [],
        },
    )

    captures = {"overrides": [], "histories": [], "tool_descriptions": []}
    monkeypatch.setattr(
        "run_agent.AIAgent",
        _build_fake_agent(captures, tool_description=MEMORY_TOOL_DESCRIPTION),
    )

    response = TestClient(app).post("/companion/v1/chat", json=_payload("msg-user-only", "继续吧"))

    user_md = (memories_dir / "USER.md").read_text(encoding="utf-8")
    assert response.status_code == 200
    assert user_md == "用户喜欢深夜调试"
    assert CONTINUITY_FACT not in user_md
    assert "用户喜欢深夜调试" in captures["overrides"][0]
    assert CONTINUITY_FACT in captures["overrides"][0]
    assert captures["tool_descriptions"][0].startswith(MEMORY_TOOL_DESCRIPTION)


def test_style_guard_enabled_remains_hardcoded_false():
    """防回归：companion_api.py 的 chat_endpoint 必须保持 style_guard_enabled = False。

    turn guard review 实测全局 0% 成功率（deepseek-v4-flash thinking 吃光 max_tokens
    致 content 恒空），17034 轮纯空转、浪费 ~43000 次辅助 LLM 调用。已永久停用，
    continuity 改由主 AI 直接写（memory_tool 解封 target=continuity）。

    此测试直接读源码断言，确保 style_guard_enabled 不被改回动态判断——否则会对
    ~10% 的 style_guard_v1 profile 重新启用 0% 成功率的 review 空转。
    修复方式：保持 `style_guard_enabled = False`（硬编码）。
    """
    import inspect
    import hermes_cli.companion_api as companion_api

    src = inspect.getsource(companion_api.chat_endpoint)
    assert "style_guard_enabled = False" in src, (
        "style_guard_enabled 必须硬编码为 False。若此测试失败，说明有人重新启用了 "
        "turn guard review（0% 成功率、~43000 次/周期空转）。修复：改回 style_guard_enabled = False"
    )
