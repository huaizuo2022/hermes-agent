from pathlib import Path

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


def test_companion_prompt_history_keeps_recent_window_with_early_summary():
    from hermes_cli import companion_api

    recent_turns = companion_api._COMPANION_HISTORY_RECENT_USER_TURNS
    # 构造刚好超过窗口 2 轮的历史，验证压缩触发与边界
    total_turns = recent_turns + 2
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

    compacted = companion_api._compact_companion_history_for_prompt(history)

    # 被压缩的早期部分 = 全部 - 最近窗口；其中用户轮次 = 2 个
    omitted_count = 2 * 2  # 2 个用户轮 + 2 个助手回复
    assert compacted[0]["role"] == "user"
    assert "早期对话摘要" in compacted[0]["content"]
    assert "此前省略了 {} 条历史消息，其中用户轮次 2 个".format(omitted_count) in compacted[0]["content"]
    assert compacted[1] == {"role": "assistant", "content": "收到，我会基于这份早期摘要承接当前对话。"}
    # 摘要占 1 个 user + 1 个 assistant，加上最近窗口 recent_turns 个 user 轮
    assert len([msg for msg in compacted if msg.get("role") == "user"]) == recent_turns + 1
    # 最早保留的原文用户轮 = 第 3 轮
    assert compacted[2] == {"role": "user", "content": "用户第3轮", "message_id": "msg-3"}
    assert compacted[-1] == {"role": "assistant", "content": "助手第{}轮".format(total_turns)}


def test_companion_prompt_history_under_recent_window_is_unchanged():
    from hermes_cli import companion_api

    recent_turns = companion_api._COMPANION_HISTORY_RECENT_USER_TURNS
    # 刚好等于窗口轮次时，不触发压缩，原样返回
    history = []
    for turn in range(1, recent_turns + 1):
        history.append(
            {
                "role": "user",
                "content": "用户第{}轮".format(turn),
                "message_id": "msg-{}".format(turn),
            }
        )
        history.append({"role": "assistant", "content": "助手第{}轮".format(turn)})

    assert companion_api._compact_companion_history_for_prompt(history) == history
