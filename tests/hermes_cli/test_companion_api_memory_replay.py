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
