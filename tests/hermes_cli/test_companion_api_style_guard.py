import json
from pathlib import Path

from starlette.testclient import TestClient

from hermes_cli.companion_profile_policy import ensure_companion_profile
from hermes_cli.companion_turn_guard import TurnReviewStore, assistant_sha256
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
        "directives": "legacy directives",
        "companion_directives": "natural companion directives",
        "character_profile": {
            "name": "测试角色",
            "personality": "温柔",
            "speaking_style": "自然",
        },
    }


def _install_fake_agent(monkeypatch, events=None):
    captures = {
        "init_kwargs": [],
        "history": [],
        "overrides": [],
    }

    class FakeAgent:
        def __init__(self, **kwargs):
            self.session_db = kwargs["session_db"]
            self.session_id = kwargs["session_id"]
            self._memory_store = type("MemoryStore", (), {"modifications": [{"kind": "legacy-memory"}]})()
            self.tools = []
            for tool_name in list(kwargs.get("enabled_toolsets") or []):
                self.tools.append(
                    {
                        "function": {
                            "name": tool_name,
                            "description": tool_name,
                        }
                    }
                )
            self.suppress_status_output = False
            self.model = kwargs["model"]
            self.provider = kwargs["provider"]
            self.base_url = kwargs["base_url"]
            self.ephemeral_system_prompt = kwargs.get("ephemeral_system_prompt")
            self._system_prompt_override = None
            captures["init_kwargs"].append(dict(kwargs))

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
            if "drift" in user_message:
                reply = "系统持续保持会话状态并执行依赖检查。"
            history_copy = [dict(item) for item in (conversation_history or [])]
            captures["history"].append(history_copy)
            captures["overrides"].append(self._system_prompt_override)
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
                for chunk in ["part-1", "part-2", "part-3"]:
                    if events is not None:
                        events.append("token:" + chunk)
                    stream_callback(chunk)
            return {"final_response": reply}

    import run_agent

    monkeypatch.setattr(run_agent, "AIAgent", FakeAgent)
    return captures


def test_legacy_profile_keeps_memory_tools_and_directives_without_sidecar(monkeypatch, tmp_path):
    profile_dir = tmp_path / "legacy-profile"
    profile_dir.mkdir()
    captures = _install_fake_agent(monkeypatch)
    monkeypatch.setattr("hermes_cli.companion_api.get_profile_path", lambda _sid: str(profile_dir))

    response = TestClient(app).post(
        "/companion/v1/chat",
        json=_payload("msg-1", "记住我喜欢攀岩", stream=False),
    )

    assert response.status_code == 200
    assert captures["init_kwargs"][0]["enabled_toolsets"] == ["memory"]
    assert captures["init_kwargs"][0]["ephemeral_system_prompt"] == "legacy directives"
    assert response.json()["memory_modifications"] == [{"kind": "legacy-memory"}]
    assert not (profile_dir / "companion_guard.db").exists()


def test_style_guard_profile_uses_companion_prompt_override_without_memory_tools(monkeypatch, tmp_path):
    profile_dir = tmp_path / "style-guard-profile"
    captures = _install_fake_agent(monkeypatch)
    monkeypatch.setattr("hermes_cli.companion_api.get_profile_path", lambda _sid: str(profile_dir))
    monkeypatch.setattr(
        "hermes_cli.companion_api.review_turn",
        lambda **kwargs: {
            "turn_id": kwargs["turn_id"],
            "review_status": "clean",
            "memory_status": "none",
            "memory_modifications": [],
        },
    )

    response = TestClient(app).post(
        "/companion/v1/chat",
        json=_payload("msg-1", "第一句", stream=False),
    )

    assert response.status_code == 200
    assert captures["init_kwargs"][0]["enabled_toolsets"] == []
    assert captures["init_kwargs"][0]["ephemeral_system_prompt"] == "natural companion directives"
    assert "角色 SOUL" in captures["overrides"][0]
    assert "测试角色" in captures["overrides"][0]
    assert captures["init_kwargs"][0]["enabled_toolsets"] == []


def test_style_guard_stream_emits_multiple_tokens_before_review(monkeypatch, tmp_path):
    profile_dir = tmp_path / "stream-profile"
    events = []
    _install_fake_agent(monkeypatch, events=events)
    monkeypatch.setattr("hermes_cli.companion_api.get_profile_path", lambda _sid: str(profile_dir))

    def fake_review_turn(**kwargs):
        events.append("review_started")
        return {
            "turn_id": kwargs["turn_id"],
            "review_status": "pending",
            "memory_status": "pending",
            "memory_modifications": [{"kind": "queued"}],
        }

    monkeypatch.setattr("hermes_cli.companion_api.review_turn", fake_review_turn)

    with TestClient(app).stream(
        "POST",
        "/companion/v1/chat",
        json=_payload("msg-1", "流式第一句", stream=True),
    ) as response:
        body = "".join(response.iter_text())

    assert response.status_code == 200
    assert body.count("event: token") == 3
    assert events[:3] == ["token:part-1", "token:part-2", "token:part-3"]
    assert events[3] == "review_started"
    assert '"review_status": "pending"' in body
    assert '"memory_modifications": [{"kind": "queued"}]' in body


def test_style_guard_drift_replays_summary_but_keeps_raw_history(monkeypatch, tmp_path):
    profile_dir = tmp_path / "drift-profile"
    ensure_companion_profile(profile_dir)
    captures = _install_fake_agent(monkeypatch)
    monkeypatch.setattr("hermes_cli.companion_api.get_profile_path", lambda _sid: str(profile_dir))
    store = TurnReviewStore(profile_dir)

    def fake_review_turn(**kwargs):
        if kwargs["turn_id"] == "msg-1":
            assistant_text = kwargs["assistant_text"]
            store.begin("msg-1", assistant_text)
            store.commit(
                {
                    "turn_id": "msg-1",
                    "assistant_sha256": assistant_sha256(assistant_text),
                    "style_decision": "drift",
                    "style_reason": "drift",
                    "continuity_summary": "她答应会留下，继续把剧情说完。",
                    "memory_operations": [],
                    "self_review": {
                        "fits_character_and_scene": "pass",
                        "no_technical_false_positive": "pass",
                        "summary_preserves_facts": "pass",
                        "summary_adds_no_new_facts": "pass",
                    },
                    "verdict": "pass",
                },
                "judge",
            )
            return {
                "turn_id": "msg-1",
                "review_status": "drift",
                "memory_status": "none",
                "memory_modifications": [],
                "review_result": {
                    "continuity_summary": "她答应会留下，继续把剧情说完。",
                },
            }
        return {
            "turn_id": kwargs["turn_id"],
            "review_status": "clean",
            "memory_status": "none",
            "memory_modifications": [],
        }

    monkeypatch.setattr("hermes_cli.companion_api.review_turn", fake_review_turn)
    client = TestClient(app)

    response1 = client.post("/companion/v1/chat", json=_payload("msg-1", "drift first", stream=False))
    response2 = client.post("/companion/v1/chat", json=_payload("msg-2", "第二句", stream=False))

    from hermes_state import SessionDB

    session_db = SessionDB(db_path=Path(profile_dir) / "state.db")
    messages = session_db.get_messages_as_conversation("savana_usera_chara")

    assert response1.status_code == 200
    assert response2.status_code == 200
    assert messages[1]["content"] == "系统持续保持会话状态并执行依赖检查。"
    assert captures["history"][1][1]["content"].startswith("【上一轮事实摘要，仅用于承接事实】\n")
    assert "系统持续保持会话状态并执行依赖检查。" not in captures["history"][1][1]["content"]


def test_style_guard_pending_and_invalid_do_not_replay_raw_assistant(monkeypatch, tmp_path):
    profile_dir = tmp_path / "pending-profile"
    ensure_companion_profile(profile_dir)
    monkeypatch.setattr("hermes_cli.companion_api.get_profile_path", lambda _sid: str(profile_dir))

    from hermes_state import SessionDB

    session_db = SessionDB(db_path=Path(profile_dir) / "state.db")
    session_db.create_session("savana_usera_chara", "savana")
    session_db.append_message(
        session_id="savana_usera_chara",
        role="user",
        content="第一句",
        platform_message_id="msg-1",
    )
    raw_assistant = "系统持续保持会话状态并执行依赖检查。"
    session_db.append_message(
        session_id="savana_usera_chara",
        role="assistant",
        content=raw_assistant,
    )

    store = TurnReviewStore(profile_dir)
    pending_record = store.begin("msg-1", raw_assistant)
    store.mark_invalid("msg-2", pending_record["assistant_sha256"], "bad", {"error": "bad"}, "judge")

    captures = _install_fake_agent(monkeypatch)
    monkeypatch.setattr(
        "hermes_cli.companion_api.review_turn",
        lambda **kwargs: {
            "turn_id": kwargs["turn_id"],
            "review_status": "clean",
            "memory_status": "none",
            "memory_modifications": [],
        },
    )

    response = TestClient(app).post(
        "/companion/v1/chat",
        json=_payload("msg-3", "第三句", stream=False),
    )

    assert response.status_code == 200
    history = captures["history"][0]
    assert history[1]["content"] != raw_assistant
    assert "等待风格审查结果" in history[1]["content"]
