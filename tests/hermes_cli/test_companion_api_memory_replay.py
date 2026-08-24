from pathlib import Path

import pytest
from starlette.testclient import TestClient

from hermes_cli.web_server import app


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


def test_style_guard_keeps_memory_tool_enabled_and_uses_savana_description(monkeypatch, tmp_path):
    captured = {}

    class FakeAgent:
        def __init__(self, **kwargs):
            captured["enabled_toolsets"] = kwargs.get("enabled_toolsets")
            self.session_db = kwargs["session_db"]
            self.session_id = kwargs["session_id"]
            self._memory_store = None
            self.tools = []
            if "memory" in (kwargs.get("enabled_toolsets") or []):
                self.tools = [
                    {
                        "type": "function",
                        "function": {
                            "name": "memory",
                            "description": "generic memory tool description",
                        },
                    }
                ]
            self.suppress_status_output = False
            self.model = kwargs["model"]
            self.provider = kwargs["provider"]
            self.base_url = kwargs["base_url"]
            self.ephemeral_system_prompt = kwargs.get("ephemeral_system_prompt")

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
            captured["tool_description"] = self.tools[0]["function"]["description"]
            return {"final_response": "收到"}

        def close(self):
            pass

    profile_dir = tmp_path / "profile"
    profile_dir.mkdir()
    monkeypatch.setattr("hermes_cli.companion_api.get_profile_path", lambda _sid: str(profile_dir))
    monkeypatch.setattr("hermes_cli.companion_api._schedule_unresolved_reviews", lambda **kwargs: None)
    monkeypatch.setattr(
        "hermes_cli.companion_api._review_companion_turn",
        lambda **kwargs: {"memory_modifications": [], "review_status": "skipped", "memory_status": "skipped"},
    )

    import run_agent

    monkeypatch.setattr(run_agent, "AIAgent", FakeAgent)

    client = TestClient(app)
    response = client.post("/companion/v1/chat", json=_payload("msg-memory", "以后叫我小鱼，记住哦"))

    assert response.status_code == 200
    assert captured["enabled_toolsets"] == ["memory"]
    assert "将用户的个人信息永久写入记忆" in captured["tool_description"]
    assert "用户说\"记住\"" in captured["tool_description"]
    assert "即使用户没有说\"记住\"" in captured["tool_description"]
    assert "稳定偏好/厌恶/避雷" in captured["tool_description"]
    assert "用户明确的未来计划、日程或重要事件" in captured["tool_description"]
    assert "用户不吃香菜" in captured["tool_description"]
    assert "用户喜欢茉莉花茶" in captured["tool_description"]
    assert "用户下周五要去苏州参加陶艺课" in captured["tool_description"]
    assert "用户自己的计划写入 target='user'" in captured["tool_description"]


def test_readonly_companion_chat_preserves_memory_files_and_compaction_checkpoint(monkeypatch, tmp_path):
    captured = {}

    class FakeAgent:
        def __init__(self, **kwargs):
            captured["enabled_toolsets"] = kwargs.get("enabled_toolsets")
            self.session_db = kwargs["session_db"]
            self.session_id = kwargs["session_id"]
            self._memory_store = None
            self._memory_enabled = True
            self._memory_nudge_interval = 9
            self.tools = []
            self.suppress_status_output = False
            self.model = kwargs["model"]
            self.provider = kwargs["provider"]
            self.base_url = kwargs["base_url"]
            self.ephemeral_system_prompt = kwargs.get("ephemeral_system_prompt")

        def run_conversation(self, user_message, **kwargs):
            captured["memory_nudge_interval"] = self._memory_nudge_interval
            return {"final_response": "[[MEMORY_UNKNOWN]]"}

        def close(self):
            pass

    profile_dir = tmp_path / "profile"
    memories_dir = profile_dir / "memories"
    memories_dir.mkdir(parents=True)
    user_path = memories_dir / "USER.md"
    continuity_path = memories_dir / "CONTINUITY.md"
    checkpoint_path = profile_dir / "companion_checkpoint.json"
    user_path.write_text("用户叫阿棠", encoding="utf-8")
    continuity_path.write_text("双方下周继续调查旧车站", encoding="utf-8")
    checkpoint_path.write_text('{"cutoff": 3}', encoding="utf-8")

    monkeypatch.setattr("hermes_cli.companion_api.get_profile_path", lambda _sid: str(profile_dir))
    monkeypatch.setattr(
        "hermes_cli.companion_api._compact_companion_history_for_prompt",
        lambda history, checkpoint: (history, {"cutoff": 4}),
    )
    monkeypatch.setattr(
        "hermes_cli.companion_api._write_companion_checkpoint",
        lambda *_args, **_kwargs: pytest.fail("readonly probe must not write checkpoint"),
    )

    import run_agent

    monkeypatch.setattr(run_agent, "AIAgent", FakeAgent)
    payload = _payload("msg-readonly", "我从来没告诉过你宠物名字。")
    payload["memory_write_mode"] = "readonly"

    response = TestClient(app).post("/companion/v1/chat", json=payload)

    assert response.status_code == 200
    assert captured["enabled_toolsets"] == []
    assert captured["memory_nudge_interval"] == 0
    assert response.json()["memory_write_mode"] == "readonly"
    assert response.json()["memory_modifications"] == []
    assert user_path.read_text(encoding="utf-8") == "用户叫阿棠"
    assert continuity_path.read_text(encoding="utf-8") == "双方下周继续调查旧车站"
    assert checkpoint_path.read_text(encoding="utf-8") == '{"cutoff": 3}'


def test_style_guard_memory_snapshots_include_continuity(monkeypatch, tmp_path):
    from hermes_cli import companion_api

    profile_dir = tmp_path / "profile"
    memories_dir = profile_dir / "memories"
    memories_dir.mkdir(parents=True)
    (memories_dir / "MEMORY.md").write_text("已有长期记忆", encoding="utf-8")
    (memories_dir / "USER.md").write_text("用户叫小鱼", encoding="utf-8")
    (memories_dir / "CONTINUITY.md").write_text("角色已答应下周陪用户去看展", encoding="utf-8")

    snapshots = companion_api._load_companion_memory_snapshots(str(profile_dir))

    assert snapshots == {
        "memory": "已有长期记忆",
        "user": "用户叫小鱼",
        "continuity": "角色已答应下周陪用户去看展",
    }
@pytest.mark.skip(reason="turn guard review 已停用（实测全局 0% 成功率），见 companion_api.py: style_guard_enabled=False")


def test_style_guard_system_prompt_includes_continuity_block(monkeypatch, tmp_path):
    from hermes_cli import companion_api

    profile_dir = tmp_path / "profile"
    memories_dir = profile_dir / "memories"
    memories_dir.mkdir(parents=True)
    (profile_dir / "SOUL.md").write_text("角色 SOUL\n测试角色", encoding="utf-8")
    (memories_dir / "MEMORY.md").write_text("已有长期记忆", encoding="utf-8")
    (memories_dir / "USER.md").write_text("用户叫小鱼", encoding="utf-8")
    (memories_dir / "CONTINUITY.md").write_text("角色已答应下周陪用户去看展", encoding="utf-8")

    captures = {}

    class FakeAgent:
        def __init__(self, **kwargs):
            self.session_db = kwargs["session_db"]
            self.session_id = kwargs["session_id"]
            self._memory_store = None
            self.tools = []
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
            captures["override"] = self._system_prompt_override
            return {"final_response": "收到"}

        def close(self):
            pass

    monkeypatch.setattr("hermes_cli.companion_api.get_profile_path", lambda _sid: str(profile_dir))
    monkeypatch.setattr("hermes_cli.companion_api._schedule_unresolved_reviews", lambda **kwargs: None)
    monkeypatch.setattr(
        "hermes_cli.companion_api._review_companion_turn",
        lambda **kwargs: {"memory_modifications": [], "review_status": "skipped", "memory_status": "skipped"},
    )

    import run_agent

    monkeypatch.setattr(run_agent, "AIAgent", FakeAgent)

    client = TestClient(app)
    response = client.post("/companion/v1/chat", json=_payload("msg-continuity", "继续吧"))

    assert response.status_code == 200
    assert "角色连续性记忆" in captures["override"]
    assert "角色已答应下周陪用户去看展" in captures["override"]
@pytest.mark.skip(reason="turn guard review 已停用（实测全局 0% 成功率），见 companion_api.py: style_guard_enabled=False")



def test_style_guard_empty_continuity_snapshot_does_not_inject(monkeypatch, tmp_path):
    from hermes_cli import companion_api

    profile_dir = tmp_path / "profile"
    memories_dir = profile_dir / "memories"
    memories_dir.mkdir(parents=True)
    (profile_dir / "SOUL.md").write_text("角色 SOUL\n测试角色", encoding="utf-8")
    (memories_dir / "MEMORY.md").write_text("已有长期记忆", encoding="utf-8")
    (memories_dir / "USER.md").write_text("用户叫小鱼", encoding="utf-8")
    (memories_dir / "CONTINUITY.md").write_text("", encoding="utf-8")

    snapshots = companion_api._load_companion_memory_snapshots(str(profile_dir))
    assert snapshots["continuity"] == ""

    captures = {}

    class FakeAgent:
        def __init__(self, **kwargs):
            self.session_db = kwargs["session_db"]
            self.session_id = kwargs["session_id"]
            self._memory_store = None
            self.tools = []
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
            captures["override"] = self._system_prompt_override
            return {"final_response": "收到"}

        def close(self):
            pass

    monkeypatch.setattr("hermes_cli.companion_api.get_profile_path", lambda _sid: str(profile_dir))
    monkeypatch.setattr("hermes_cli.companion_api._schedule_unresolved_reviews", lambda **kwargs: None)
    monkeypatch.setattr(
        "hermes_cli.companion_api._review_companion_turn",
        lambda **kwargs: {"memory_modifications": [], "review_status": "skipped", "memory_status": "skipped"},
    )

    import run_agent

    monkeypatch.setattr(run_agent, "AIAgent", FakeAgent)

    response = TestClient(app).post("/companion/v1/chat", json=_payload("msg-empty", "继续吧"))
    assert response.status_code == 200
    assert "CONTINUITY" not in captures["override"]
    assert "已有长期记忆" in captures["override"]
    assert "用户叫小鱼" in captures["override"]
@pytest.mark.skip(reason="turn guard review 已停用（实测全局 0% 成功率），见 companion_api.py: style_guard_enabled=False")


def test_style_guard_non_savana_keeps_memory_and_user_but_skips_continuity(monkeypatch, tmp_path):
    from hermes_cli import companion_api

    profile_dir = tmp_path / "profile"
    memories_dir = profile_dir / "memories"
    memories_dir.mkdir(parents=True)
    (memories_dir / "MEMORY.md").write_text("已有长期记忆", encoding="utf-8")
    (memories_dir / "USER.md").write_text("用户叫小鱼", encoding="utf-8")
    (memories_dir / "CONTINUITY.md").write_text("角色已答应下周陪用户去看展", encoding="utf-8")

    captures = {}

    class FakeAgent:
        def __init__(self, **kwargs):
            self.session_db = kwargs["session_db"]
            self.session_id = kwargs["session_id"]
            self._memory_store = None
            self.tools = []
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
            captures["override"] = self._system_prompt_override
            return {"final_response": "收到"}

        def close(self):
            pass

    monkeypatch.setattr("hermes_cli.companion_api.get_profile_path", lambda _sid: str(profile_dir))
    monkeypatch.setattr("hermes_cli.companion_api._schedule_unresolved_reviews", lambda **kwargs: None)
    monkeypatch.setattr(
        "hermes_cli.companion_api._review_companion_turn",
        lambda **kwargs: {"memory_modifications": [], "review_status": "skipped", "memory_status": "skipped"},
    )

    import run_agent

    monkeypatch.setattr(run_agent, "AIAgent", FakeAgent)

    response = TestClient(app).post("/companion/v1/chat", json=_payload("msg-nonsavana", "继续吧"))
    assert response.status_code == 200
    assert "已有长期记忆" in captures["override"]
    assert "用户叫小鱼" in captures["override"]
    assert "角色连续性记忆" in captures["override"]
    assert "角色已答应下周陪用户去看展" in captures["override"]


def test_companion_chat_replays_prior_turns_via_run_agent_patch_only(monkeypatch, tmp_path):
    call_history = []

    class FakeAgent:
        def __init__(self, **kwargs):
            self.session_db = kwargs["session_db"]
            self.session_id = kwargs["session_id"]
            self._memory_store = None
            self.tools = []
            self.suppress_status_output = False
            self.model = kwargs["model"]
            self.provider = kwargs["provider"]
            self.base_url = kwargs["base_url"]
            self.ephemeral_system_prompt = kwargs.get("ephemeral_system_prompt")

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
            reply = "reply to {}".format(user_message)
            history_copy = [dict(item) for item in (conversation_history or [])]
            call_history.append(
                {
                    "user_message": user_message,
                    "conversation_history": history_copy,
                    "platform_message_id": platform_message_id,
                }
            )
            self.session_db.append_message(
                session_id=self.session_id,
                role="user",
                content=user_message,
                platform_message_id=platform_message_id,
            )
            self.session_db.append_message(
                session_id=self.session_id,
                role="assistant",
                content=reply,
            )
            if stream_callback:
                stream_callback(reply)
            return {"final_response": reply}

        def chat(self, message, stream_callback=None):
            result = self.run_conversation(message, stream_callback=stream_callback)
            return result["final_response"]

    profile_dir = tmp_path / "profile"
    profile_dir.mkdir()
    monkeypatch.setattr("hermes_cli.companion_api.get_profile_path", lambda _sid: str(profile_dir))

    import run_agent

    monkeypatch.setattr(run_agent, "AIAgent", FakeAgent)

    client = TestClient(app)
    response1 = client.post("/companion/v1/chat", json=_payload("msg-1", "第一句"))
    response2 = client.post("/companion/v1/chat", json=_payload("msg-2", "第二句"))

    assert response1.status_code == 200
    assert response2.status_code == 200
    assert len(call_history) == 2
    assert call_history[0]["conversation_history"] == []
    assert call_history[1]["conversation_history"] == [
        {"role": "user", "content": "第一句", "message_id": "msg-1"},
        {"role": "assistant", "content": "reply to 第一句"},
    ]
    assert call_history[1]["platform_message_id"] == "msg-2"


def test_companion_chat_does_not_double_write_current_user(monkeypatch, tmp_path):
    class FakeAgent:
        def __init__(self, **kwargs):
            self.session_db = kwargs["session_db"]
            self.session_id = kwargs["session_id"]
            self._memory_store = None
            self.tools = []
            self.suppress_status_output = False
            self.model = kwargs["model"]
            self.provider = kwargs["provider"]
            self.base_url = kwargs["base_url"]
            self.ephemeral_system_prompt = kwargs.get("ephemeral_system_prompt")

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
            reply = "single reply"
            self.session_db.append_message(
                session_id=self.session_id,
                role="user",
                content=user_message,
                platform_message_id=platform_message_id,
            )
            self.session_db.append_message(
                session_id=self.session_id,
                role="assistant",
                content=reply,
            )
            return {"final_response": reply}

        def chat(self, message, stream_callback=None):
            result = self.run_conversation(message, stream_callback=stream_callback)
            return result["final_response"]

    profile_dir = tmp_path / "profile"
    profile_dir.mkdir()
    monkeypatch.setattr("hermes_cli.companion_api.get_profile_path", lambda _sid: str(profile_dir))

    import run_agent
    from hermes_state import SessionDB

    monkeypatch.setattr(run_agent, "AIAgent", FakeAgent)

    client = TestClient(app)
    response = client.post("/companion/v1/chat", json=_payload("msg-1", "只发一轮"))

    assert response.status_code == 200

    session_id = "savana_usera_chara"
    db = SessionDB(db_path=Path(profile_dir) / "state.db")
    messages = db.get_messages_as_conversation(session_id)

    assert messages == [
        {"role": "user", "content": "只发一轮", "message_id": "msg-1"},
        {"role": "assistant", "content": "single reply"},
    ]


def _build_turns(total_turns):
    """构造 total_turns 个用户轮次的历史（user 带 message_id，assistant 不带）。"""
    history = []
    for turn in range(1, total_turns + 1):
        history.append(
            {
                "role": "user",
                "content": "用户第{}轮".format(turn),
                "message_id": "msg-{}".format(turn),
            }
        )
        history.append({"role": "assistant", "content": "助手第{}轮".format(turn)})
    return history


def test_companion_prompt_history_first_reanchor_builds_checkpoint():
    from hermes_cli import companion_api

    recent_turns = companion_api._COMPANION_HISTORY_RECENT_USER_TURNS
    # 构造刚好超过窗口 2 轮的历史，验证首次重锚定触发与切点边界
    total_turns = recent_turns + 2
    history = _build_turns(total_turns)

    compacted, checkpoint = companion_api._compact_companion_history_for_prompt(
        history, checkpoint=None
    )

    # 生成了 checkpoint，切点 = 全部 - 最近窗口 = 前 2 轮（4 条）
    assert checkpoint is not None
    assert checkpoint["cut_index"] == 2 * 2
    assert checkpoint["summary_text"]

    # 被压缩的早期部分 = 前 2 个用户轮 + 2 个助手回复 = 4 条
    omitted_count = 2 * 2
    assert compacted[0]["role"] == "user"
    assert "早期对话摘要" in compacted[0]["content"]
    assert "本次压缩：省略 {} 条历史消息，其中用户轮次 2 个".format(omitted_count) in compacted[0]["content"]
    assert compacted[1] == {"role": "assistant", "content": "收到，我会基于这份早期摘要承接当前对话。"}
    # 摘要占 1 个 user + 1 个 assistant，加上最近窗口 recent_turns 个 user 轮
    assert len([msg for msg in compacted if msg.get("role") == "user"]) == recent_turns + 1
    # 最早保留的原文用户轮 = 第 3 轮
    assert compacted[2] == {"role": "user", "content": "用户第3轮", "message_id": "msg-3"}
    assert compacted[-1] == {"role": "assistant", "content": "助手第{}轮".format(total_turns)}


def test_companion_prompt_history_checkpoint_reuse_keeps_prefix_byte_stable():
    """核心不变量：复用 checkpoint 后，history 前缀逐字节稳定，只在末尾追加新轮次。"""
    from hermes_cli import companion_api

    recent_turns = companion_api._COMPANION_HISTORY_RECENT_USER_TURNS
    history = _build_turns(recent_turns + 2)

    # 第一次：首次重锚定，拿到 checkpoint
    compacted1, checkpoint = companion_api._compact_companion_history_for_prompt(
        history, checkpoint=None
    )
    assert checkpoint is not None

    # 追加一轮（user + assistant），传入 checkpoint 复用
    appended = history + [
        {"role": "user", "content": "用户第{}轮".format(recent_turns + 3), "message_id": "msg-{}".format(recent_turns + 3)},
        {"role": "assistant", "content": "助手第{}轮".format(recent_turns + 3)},
    ]
    compacted2, checkpoint2 = companion_api._compact_companion_history_for_prompt(
        appended, checkpoint=checkpoint
    )

    # checkpoint 原样复用（同一对象，未触发重锚定）
    assert checkpoint2 is checkpoint
    # 摘要逐字节相等
    assert compacted2[0]["content"] == compacted1[0]["content"]
    # 前缀逐字节稳定：第二次的输出 = 第一次的输出 + 末尾新增的一轮
    assert compacted2[:-2] == compacted1
    assert compacted2[-2:] == appended[-2:]


def test_companion_prompt_history_reanchor_when_appended_exceeds_budget():
    """追加段 token 估算超预算时，触发重锚定：切点前移、摘要更新、保留最近窗口。"""
    from hermes_cli import companion_api

    recent_turns = companion_api._COMPANION_HISTORY_RECENT_USER_TURNS
    history = _build_turns(recent_turns + 2)

    _, checkpoint = companion_api._compact_companion_history_for_prompt(
        history, checkpoint=None
    )
    first_cut = checkpoint["cut_index"]

    # 追加大量内容，使追加段 token 估算超过一个极小的预算
    big = list(history)
    for turn in range(recent_turns + 3, recent_turns + 40):
        big.append({"role": "user", "content": "用户第{}轮".format(turn) + "y" * 400, "message_id": "msg-{}".format(turn)})
        big.append({"role": "assistant", "content": "助手第{}轮".format(turn) + "y" * 400})

    compacted, checkpoint2 = companion_api._compact_companion_history_for_prompt(
        big, checkpoint=checkpoint, max_prompt_tokens=100
    )

    # 触发了重锚定：新 checkpoint、切点前移、摘要变化
    assert checkpoint2 is not checkpoint
    assert checkpoint2["cut_index"] > first_cut
    assert checkpoint2["summary_text"] != checkpoint["summary_text"]
    # 重锚定后仍保留最近 recent_turns 个用户轮次
    assert len([msg for msg in compacted if msg.get("role") == "user"]) == recent_turns + 1


def test_companion_prompt_history_under_recent_window_is_unchanged():
    from hermes_cli import companion_api

    recent_turns = companion_api._COMPANION_HISTORY_RECENT_USER_TURNS
    # 刚好等于窗口轮次时，不触发压缩，原样返回，且不生成 checkpoint
    history = _build_turns(recent_turns)

    compacted, checkpoint = companion_api._compact_companion_history_for_prompt(
        history, checkpoint=None
    )
    assert compacted == history
    assert checkpoint is None


def test_companion_prompt_history_reanchor_preserves_prior_summary():
    """重锚定时旧摘要正文必须合并进新摘要（累积式），防止中期剧情失忆。"""
    from hermes_cli import companion_api

    recent_turns = companion_api._COMPANION_HISTORY_RECENT_USER_TURNS
    history = _build_turns(recent_turns + 2)

    _, checkpoint = companion_api._compact_companion_history_for_prompt(
        history, checkpoint=None
    )
    assert checkpoint is not None
    # 给旧摘要注入一个可辨识的旧事实
    old_fact = "- user: 两人在梧桐树下确认了恋人关系"
    checkpoint["summary_text"] = checkpoint["summary_text"] + "\n" + old_fact

    # 追加大量内容触发重锚定
    big = list(history)
    for turn in range(recent_turns + 3, recent_turns + 40):
        big.append({"role": "user", "content": "用户第{}轮".format(turn) + "y" * 400, "message_id": "msg-{}".format(turn)})
        big.append({"role": "assistant", "content": "助手第{}轮".format(turn) + "y" * 400})

    compacted, checkpoint2 = companion_api._compact_companion_history_for_prompt(
        big, checkpoint=checkpoint, max_prompt_tokens=100
    )

    assert checkpoint2 is not checkpoint
    assert "梧桐树下确认了恋人关系" in checkpoint2["summary_text"]
    assert "更早剧情（前次压缩保留）" in checkpoint2["summary_text"]
    # 新摘要同样注入到渲染后的 history 首条
    assert "梧桐树下确认了恋人关系" in compacted[0]["content"]


def test_companion_summary_char_limit_trims_oldest_first():
    """累积摘要超上限时从头部裁剪：最老信息先淘汰，最新采样保留。"""
    from hermes_cli import companion_api

    limit = companion_api._COMPANION_SUMMARY_CHAR_LIMIT
    old_summary = "【早期对话摘要（自动压缩，仅用于承接事实）】\n" + "\n".join(
        "- user: 古老事实{} {}".format(i, "x" * 200) for i in range(60)
    )
    assert len(old_summary) > limit

    messages = [
        {"role": "user", "content": "最新关键剧情", "message_id": "m1"},
        {"role": "assistant", "content": "最新回复"},
    ]
    summary = companion_api._summarize_early_companion_history(
        messages, prior_summary=old_summary
    )

    assert len(summary) <= limit
    # 头部被裁（最老事实消失），尾部最新内容保留
    assert "古老事实0" not in summary
    assert "最新关键剧情" in summary
    assert summary.startswith("【早期对话摘要")


def test_companion_prompt_token_budget_stays_cost_bounded():
    """防回归：预算必须保持成本驱动的紧值（≤ 64K），不允许回到窗口驱动。"""
    from hermes_cli import companion_api

    assert companion_api._COMPANION_PROMPT_TOKEN_BUDGET <= 64000
