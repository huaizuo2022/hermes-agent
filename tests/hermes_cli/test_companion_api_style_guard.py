import asyncio
import json
import threading
import time
from pathlib import Path

import httpx
import pytest
from starlette.testclient import TestClient

from hermes_cli.companion_profile_policy import ensure_companion_profile
from hermes_cli.companion_turn_guard import TurnReviewStore, assistant_sha256
from hermes_cli.web_server import app
from run_agent import AIAgent


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


class CountingLock:
    def __init__(self):
        self._lock = threading.Lock()
        self.acquire_count = 0
        self.release_count = 0

    def acquire(self):
        self.acquire_count += 1
        self._lock.acquire()
        return True

    def release(self):
        self.release_count += 1
        self._lock.release()


class ObservableCountingLock:
    def __init__(self):
        self._lock = threading.Lock()
        self.acquire_count = 0
        self.release_count = 0
        self.acquire_started = threading.Event()
        self.acquire_completed = threading.Event()
        self.release_observed = threading.Event()

    def hold_for_test(self):
        self._lock.acquire()

    def acquire(self):
        self.acquire_count += 1
        self.acquire_started.set()
        self._lock.acquire()
        self.acquire_completed.set()
        return True

    def release(self):
        self.release_count += 1
        self.release_observed.set()
        self._lock.release()


async def _invoke_asgi_app(scope, receive, send):
    await app(scope, receive, send)


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
    assert "review_status" not in response.json()
    assert "memory_status" not in response.json()
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
    assert captures["init_kwargs"][0]["ephemeral_system_prompt"] in ("", None)
    assert "角色 SOUL" in captures["overrides"][0]
    assert "测试角色" in captures["overrides"][0]
    assert "natural companion directives" in captures["overrides"][0]
    assert "legacy directives" not in captures["overrides"][0]

    agent = object.__new__(AIAgent)
    agent._system_prompt_override = captures["overrides"][0]
    assert "natural companion directives" in agent._build_system_prompt()


def test_style_guard_stream_emits_multiple_tokens_before_review(monkeypatch, tmp_path):
    profile_dir = tmp_path / "stream-profile"
    events = []
    _install_fake_agent(monkeypatch, events=events)
    monkeypatch.setattr("hermes_cli.companion_api.get_profile_path", lambda _sid: str(profile_dir))
    import hermes_cli.companion_api as companion_api

    monkeypatch.setattr(
        companion_api,
        "_stream_event_hook",
        lambda name, payload=None: events.append("yield:" + name),
        raising=False,
    )

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
        consumed = []
        chunks = []
        for chunk in response.iter_text():
            chunks.append(chunk)
            if "event: token" in chunk:
                consumed.append(chunk)
        body = "".join(chunks)

    assert response.status_code == 200
    assert body.count("event: token") == 3
    assert events[:3] == ["token:part-1", "token:part-2", "token:part-3"]
    assert events[3:6] == ["yield:token", "yield:token", "yield:token"]
    review_index = events.index("review_started")
    assert review_index >= 6
    assert "yield:metadata" in events[review_index + 1 :]
    assert '"review_status": "pending"' in body
    assert '"memory_status": "pending"' in body
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
    def fake_review_turn(**kwargs):
        if kwargs["turn_id"] == "msg-1":
            return {
                "turn_id": kwargs["turn_id"],
                "review_status": "invalid",
                "memory_status": "none",
                "memory_modifications": [],
            }
        profile_path = Path(kwargs["profile_dir"])
        store = TurnReviewStore(profile_path)
        store.begin(kwargs["turn_id"], kwargs["assistant_text"])
        store.commit(
            {
                "turn_id": kwargs["turn_id"],
                "assistant_sha256": assistant_sha256(kwargs["assistant_text"]),
                "style_decision": "clean",
                "style_reason": "ok",
                "continuity_summary": "",
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
            "turn_id": kwargs["turn_id"],
            "review_status": "clean",
            "memory_status": "none",
            "memory_modifications": [],
        }

    monkeypatch.setattr("hermes_cli.companion_api.review_turn", fake_review_turn)

    response = TestClient(app).post(
        "/companion/v1/chat",
        json=_payload("msg-3", "第三句", stream=False),
    )

    assert response.status_code == 200
    history = captures["history"][0]
    assert history[1]["content"] != raw_assistant
    assert "等待风格审查结果" in history[1]["content"]


def test_style_guard_restores_unresolved_turns_before_current_generation(monkeypatch, tmp_path):
    profile_dir = tmp_path / "restore-profile"
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
    store.begin("msg-1", raw_assistant)
    captures = _install_fake_agent(monkeypatch)
    review_calls = []

    def fake_review_turn(**kwargs):
        review_calls.append(kwargs["turn_id"])
        if kwargs["turn_id"] == "msg-1":
            store.begin("msg-1", kwargs["assistant_text"])
            store.commit(
                {
                    "turn_id": "msg-1",
                    "assistant_sha256": assistant_sha256(kwargs["assistant_text"]),
                    "style_decision": "drift",
                    "style_reason": "restored",
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
            "turn_id": kwargs["turn_id"],
            "review_status": "clean",
            "memory_status": "none",
            "memory_modifications": [],
        }

    monkeypatch.setattr("hermes_cli.companion_api.review_turn", fake_review_turn)

    response = TestClient(app).post(
        "/companion/v1/chat",
        json=_payload("msg-2", "第二句", stream=False),
    )

    assert response.status_code == 200
    assert review_calls[:2] == ["msg-1", "msg-2"]
    assert captures["history"][0][1]["content"].startswith("【上一轮事实摘要，仅用于承接事实】\n")


def test_style_guard_nonstream_metadata_includes_review_and_memory_status(monkeypatch, tmp_path):
    profile_dir = tmp_path / "metadata-profile"
    captures = _install_fake_agent(monkeypatch)
    monkeypatch.setattr("hermes_cli.companion_api.get_profile_path", lambda _sid: str(profile_dir))
    monkeypatch.setattr(
        "hermes_cli.companion_api.review_turn",
        lambda **kwargs: {
            "turn_id": kwargs["turn_id"],
            "review_status": "clean",
            "memory_status": "applied",
            "memory_modifications": [{"kind": "applied"}],
        },
    )

    response = TestClient(app).post(
        "/companion/v1/chat",
        json=_payload("msg-1", "第一句", stream=False),
    )

    assert response.status_code == 200
    assert captures["init_kwargs"][0]["enabled_toolsets"] == []
    assert response.json()["review_status"] == "clean"
    assert response.json()["memory_status"] == "applied"
    assert response.json()["memory_modifications"] == [{"kind": "applied"}]


def test_legacy_stream_metadata_shape_remains_base_compatible(monkeypatch, tmp_path):
    profile_dir = tmp_path / "legacy-stream-profile"
    profile_dir.mkdir()
    _install_fake_agent(monkeypatch)
    monkeypatch.setattr("hermes_cli.companion_api.get_profile_path", lambda _sid: str(profile_dir))

    with TestClient(app).stream(
        "POST",
        "/companion/v1/chat",
        json=_payload("msg-1", "第一句", stream=True),
    ) as response:
        body = "".join(response.iter_text())

    assert response.status_code == 200
    assert '"memory_modifications": [{"kind": "legacy-memory"}]' in body
    assert '"review_status"' not in body
    assert '"memory_status"' not in body


def test_legacy_stream_error_still_emits_base_metadata(monkeypatch, tmp_path):
    profile_dir = tmp_path / "legacy-stream-error-profile"
    profile_dir.mkdir()

    class FailingAgent:
        def __init__(self, **kwargs):
            self.session_db = kwargs["session_db"]
            self.session_id = kwargs["session_id"]
            self._memory_store = type("MemoryStore", (), {"modifications": [{"kind": "legacy-memory"}]})()
            self.tools = []
            self.suppress_status_output = False
            self.model = kwargs["model"]
            self.provider = kwargs["provider"]
            self.base_url = kwargs["base_url"]
            self.ephemeral_system_prompt = kwargs.get("ephemeral_system_prompt")

        def run_conversation(self, *args, **kwargs):
            raise RuntimeError("boom")

    import run_agent

    monkeypatch.setattr(run_agent, "AIAgent", FailingAgent)
    monkeypatch.setattr("hermes_cli.companion_api.get_profile_path", lambda _sid: str(profile_dir))

    with TestClient(app).stream(
        "POST",
        "/companion/v1/chat",
        json=_payload("msg-1", "第一句", stream=True),
    ) as response:
        body = "".join(response.iter_text())

    assert response.status_code == 200
    assert body.index("event: error") < body.index("event: metadata")
    assert '"session_id": "savana_usera_chara"' in body
    assert '"status": "completed"' in body
    assert '"memory_modifications": [{"kind": "legacy-memory"}]' in body
    assert '"review_status"' not in body
    assert '"memory_status"' not in body


def test_style_guard_prompt_cache_overrides_stored_system_prompt_each_turn(monkeypatch, tmp_path):
    profile_dir = tmp_path / "prompt-cache-profile"
    ensure_companion_profile(profile_dir)
    monkeypatch.setattr("hermes_cli.companion_api.get_profile_path", lambda _sid: str(profile_dir))

    from hermes_state import SessionDB

    session_id = "savana_usera_chara"
    session_db = SessionDB(db_path=Path(profile_dir) / "state.db")
    session_db.create_session(session_id, "savana")
    session_db._conn.execute(
        "UPDATE sessions SET system_prompt = ? WHERE id = ?",
        ("OLD SYSTEM PROMPT legacy directives", session_id),
    )
    session_db._conn.commit()
    session_db.close()

    captured_prompts = []

    def fake_run_conversation(
        self,
        user_message,
        system_message=None,
        conversation_history=None,
        task_id=None,
        stream_callback=None,
        persist_user_message=None,
        platform_message_id=None,
    ):
        captured_prompts.append(self._cached_system_prompt)
        return {"final_response": "reply to {}".format(user_message)}

    monkeypatch.setattr(AIAgent, "run_conversation", fake_run_conversation)
    monkeypatch.setattr(
        "hermes_cli.companion_api.review_turn",
        lambda **kwargs: {
            "turn_id": kwargs["turn_id"],
            "review_status": "clean",
            "memory_status": "none",
            "memory_modifications": [],
        },
    )

    payload1 = _payload("msg-1", "第一句", stream=False)
    payload1["companion_directives"] = "NATURAL NEW 1"
    payload2 = _payload("msg-2", "第二句", stream=False)
    payload2["companion_directives"] = "NATURAL NEW 2"

    response1 = TestClient(app).post("/companion/v1/chat", json=payload1)
    response2 = TestClient(app).post("/companion/v1/chat", json=payload2)

    assert response1.status_code == 200
    assert response2.status_code == 200
    assert "NATURAL NEW 1" in captured_prompts[0]
    assert "NATURAL NEW 2" in captured_prompts[1]
    assert "OLD SYSTEM PROMPT" not in captured_prompts[0]
    assert "OLD SYSTEM PROMPT" not in captured_prompts[1]
    assert "legacy directives" not in captured_prompts[0]
    assert "legacy directives" not in captured_prompts[1]


def test_same_session_requests_are_serialized_but_different_sessions_can_overlap(monkeypatch, tmp_path):
    state = {
        "active": 0,
        "max_active": 0,
        "entered": {},
    }
    state_lock = threading.Lock()
    release_event = threading.Event()

    class BlockingAgent:
        def __init__(self, **kwargs):
            self.session_db = kwargs["session_db"]
            self.session_id = kwargs["session_id"]
            self._memory_store = None
            self.tools = []
            self.suppress_status_output = False
            self.model = kwargs["model"]
            self.provider = kwargs["provider"]
            self.base_url = kwargs["base_url"]

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
            with state_lock:
                state["active"] += 1
                state["max_active"] = max(state["max_active"], state["active"])
                state["entered"].setdefault(self.session_id, []).append(
                    [user_message, [dict(item) for item in (conversation_history or [])]]
                )
            release_event.wait(2.0)
            self.session_db.append_message(
                session_id=self.session_id,
                role="user",
                content=user_message,
                platform_message_id=platform_message_id,
            )
            self.session_db.append_message(
                session_id=self.session_id,
                role="assistant",
                content="reply to {}".format(user_message),
            )
            with state_lock:
                state["active"] -= 1
            return {"final_response": "reply to {}".format(user_message)}

    import run_agent

    monkeypatch.setattr(run_agent, "AIAgent", BlockingAgent)
    def fake_review_turn(**kwargs):
        store = TurnReviewStore(Path(kwargs["profile_dir"]))
        store.begin(kwargs["turn_id"], kwargs["assistant_text"])
        store.commit(
            {
                "turn_id": kwargs["turn_id"],
                "assistant_sha256": assistant_sha256(kwargs["assistant_text"]),
                "style_decision": "clean",
                "style_reason": "ok",
                "continuity_summary": "",
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
            "turn_id": kwargs["turn_id"],
            "review_status": "clean",
            "memory_status": "none",
            "memory_modifications": [],
        }

    monkeypatch.setattr("hermes_cli.companion_api.review_turn", fake_review_turn)

    def path_for(session_id):
        return str(tmp_path / session_id)

    monkeypatch.setattr("hermes_cli.companion_api.get_profile_path", path_for)
    client = TestClient(app)
    same_results = []

    def same_request(message_id, text):
        same_results.append(
            client.post("/companion/v1/chat", json=_payload(message_id, text, stream=False))
        )

    thread_a = threading.Thread(target=same_request, args=("msg-1", "第一句"))
    thread_b = threading.Thread(target=same_request, args=("msg-2", "第二句"))
    thread_a.start()
    time.sleep(0.2)
    thread_b.start()
    time.sleep(0.2)

    with state_lock:
        same_entered = len(state["entered"].get("savana_usera_chara", []))
    assert same_entered == 1

    release_event.set()
    thread_a.join()
    thread_b.join()

    assert [resp.status_code for resp in same_results] == [200, 200]
    same_history = state["entered"]["savana_usera_chara"][1][1]
    assert same_history[-1]["role"] == "assistant"
    assert same_history[-1]["content"] == "reply to 第一句"

    state["active"] = 0
    state["max_active"] = 0
    state["entered"] = {}
    release_event.clear()
    different_results = []

    def request_for(user_id):
        payload = _payload("msg-1", "你好", stream=False)
        payload["user_id"] = user_id
        different_results.append(client.post("/companion/v1/chat", json=payload))

    thread_c = threading.Thread(target=request_for, args=("UserB",))
    thread_d = threading.Thread(target=request_for, args=("UserC",))
    thread_c.start()
    thread_d.start()
    time.sleep(0.2)
    with state_lock:
        assert state["max_active"] == 2
    release_event.set()
    thread_c.join()
    thread_d.join()
    assert [resp.status_code for resp in different_results] == [200, 200]


def test_stream_requests_hold_same_session_lock_until_stream_completes(monkeypatch, tmp_path):
    state = {
        "active": 0,
        "max_active": 0,
        "entered": {},
    }
    state_lock = threading.Lock()
    release_event = threading.Event()

    class BlockingStreamAgent:
        def __init__(self, **kwargs):
            self.session_db = kwargs["session_db"]
            self.session_id = kwargs["session_id"]
            self._memory_store = None
            self.tools = []
            self.suppress_status_output = False
            self.model = kwargs["model"]
            self.provider = kwargs["provider"]
            self.base_url = kwargs["base_url"]

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
            with state_lock:
                state["active"] += 1
                state["max_active"] = max(state["max_active"], state["active"])
                state["entered"].setdefault(self.session_id, []).append(
                    [user_message, [dict(item) for item in (conversation_history or [])]]
                )
            if stream_callback:
                stream_callback("chunk-a")
            release_event.wait(2.0)
            self.session_db.append_message(
                session_id=self.session_id,
                role="user",
                content=user_message,
                platform_message_id=platform_message_id,
            )
            self.session_db.append_message(
                session_id=self.session_id,
                role="assistant",
                content="reply to {}".format(user_message),
            )
            if stream_callback:
                stream_callback("chunk-b")
            with state_lock:
                state["active"] -= 1
            return {"final_response": "reply to {}".format(user_message)}

    import run_agent

    monkeypatch.setattr(run_agent, "AIAgent", BlockingStreamAgent)

    def fake_review_turn(**kwargs):
        store = TurnReviewStore(Path(kwargs["profile_dir"]))
        store.begin(kwargs["turn_id"], kwargs["assistant_text"])
        store.commit(
            {
                "turn_id": kwargs["turn_id"],
                "assistant_sha256": assistant_sha256(kwargs["assistant_text"]),
                "style_decision": "clean",
                "style_reason": "ok",
                "continuity_summary": "",
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
            "turn_id": kwargs["turn_id"],
            "review_status": "clean",
            "memory_status": "none",
            "memory_modifications": [],
        }

    monkeypatch.setattr("hermes_cli.companion_api.review_turn", fake_review_turn)
    monkeypatch.setattr("hermes_cli.companion_api.get_profile_path", lambda sid: str(tmp_path / sid))

    same_results = []

    def stream_request(results, user_id, message_id, text):
        payload = _payload(message_id, text, stream=True)
        payload["user_id"] = user_id
        with TestClient(app).stream("POST", "/companion/v1/chat", json=payload) as response:
            body = "".join(response.iter_text())
        results.append((response.status_code, body))

    thread_a = threading.Thread(target=stream_request, args=(same_results, "UserA", "msg-1", "第一句"))
    thread_b = threading.Thread(target=stream_request, args=(same_results, "UserA", "msg-2", "第二句"))
    thread_a.start()
    time.sleep(0.2)
    thread_b.start()
    time.sleep(0.2)

    with state_lock:
        assert len(state["entered"].get("savana_usera_chara", [])) == 1

    release_event.set()
    thread_a.join()
    thread_b.join()

    assert [status for status, body in same_results] == [200, 200]
    same_history = state["entered"]["savana_usera_chara"][1][1]
    assert same_history[-1]["role"] == "assistant"
    assert same_history[-1]["content"] == "reply to 第一句"

    state["active"] = 0
    state["max_active"] = 0
    state["entered"] = {}
    release_event.clear()
    different_results = []

    thread_c = threading.Thread(target=stream_request, args=(different_results, "UserB", "msg-1", "你好"))
    thread_d = threading.Thread(target=stream_request, args=(different_results, "UserC", "msg-1", "你好"))
    thread_c.start()
    thread_d.start()
    time.sleep(0.2)
    with state_lock:
        assert state["max_active"] >= 2
    release_event.set()
    thread_c.join()
    thread_d.join()
    assert [status for status, body in different_results] == [200, 200]


def test_style_guard_does_not_touch_session_lock_for_legacy(monkeypatch, tmp_path):
    profile_dir = tmp_path / "legacy-no-lock-profile"
    profile_dir.mkdir()
    _install_fake_agent(monkeypatch)
    monkeypatch.setattr("hermes_cli.companion_api.get_profile_path", lambda _sid: str(profile_dir))
    class SentinelLock:
        def acquire(self):
            raise AssertionError("legacy should not acquire session lock")

    monkeypatch.setattr("hermes_cli.companion_api._get_session_lock", lambda _sid: SentinelLock())

    response = TestClient(app).post(
        "/companion/v1/chat",
        json=_payload("msg-1", "第一句", stream=False),
    )

    assert response.status_code == 200


def test_style_guard_stream_counting_lock_releases_on_success_and_disconnect(monkeypatch, tmp_path):
    profile_dir = tmp_path / "counting-lock-profile"
    ensure_companion_profile(profile_dir)
    monkeypatch.setattr("hermes_cli.companion_api.get_profile_path", lambda _sid: str(profile_dir))
    counting_lock = CountingLock()
    monkeypatch.setattr("hermes_cli.companion_api._get_session_lock", lambda _sid: counting_lock)

    class StreamingAgent:
        def __init__(self, **kwargs):
            self.session_db = kwargs["session_db"]
            self.session_id = kwargs["session_id"]
            self._memory_store = None
            self.tools = []
            self.suppress_status_output = False
            self.model = kwargs["model"]
            self.provider = kwargs["provider"]
            self.base_url = kwargs["base_url"]

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
            if stream_callback:
                stream_callback("chunk-a")
            time.sleep(0.1)
            self.session_db.append_message(
                session_id=self.session_id,
                role="user",
                content=user_message,
                platform_message_id=platform_message_id,
            )
            self.session_db.append_message(
                session_id=self.session_id,
                role="assistant",
                content="reply to {}".format(user_message),
            )
            if stream_callback:
                stream_callback("chunk-b")
            return {"final_response": "reply to {}".format(user_message)}

    import run_agent

    monkeypatch.setattr(run_agent, "AIAgent", StreamingAgent)
    monkeypatch.setattr(
        "hermes_cli.companion_api.review_turn",
        lambda **kwargs: {
            "turn_id": kwargs["turn_id"],
            "review_status": "clean",
            "memory_status": "none",
            "memory_modifications": [],
        },
    )

    with TestClient(app).stream(
        "POST",
        "/companion/v1/chat",
        json=_payload("msg-1", "第一句", stream=True),
    ) as response:
        _ = "".join(response.iter_text())

    assert counting_lock.acquire_count == 1
    assert counting_lock.release_count == 1

    with TestClient(app).stream(
        "POST",
        "/companion/v1/chat",
        json=_payload("msg-2", "第二句", stream=True),
    ) as response:
        iterator = response.iter_text()
        next(iterator)

    assert counting_lock.acquire_count == 2
    assert counting_lock.release_count == 2


def test_style_guard_stream_counting_lock_releases_when_thread_start_fails(monkeypatch, tmp_path):
    profile_dir = tmp_path / "thread-start-fail-profile"
    ensure_companion_profile(profile_dir)
    monkeypatch.setattr("hermes_cli.companion_api.get_profile_path", lambda _sid: str(profile_dir))
    counting_lock = CountingLock()
    monkeypatch.setattr("hermes_cli.companion_api._get_session_lock", lambda _sid: counting_lock)
    _install_fake_agent(monkeypatch)

    monkeypatch.setattr(
        "hermes_cli.companion_api._start_background_thread",
        lambda target: (_ for _ in ()).throw(RuntimeError("start failed")),
    )

    response = TestClient(app, raise_server_exceptions=False).post(
        "/companion/v1/chat",
        json=_payload("msg-1", "第一句", stream=True),
    )

    assert response.status_code == 500
    assert counting_lock.acquire_count == 1
    assert counting_lock.release_count == 1


@pytest.mark.asyncio
async def test_style_guard_stream_lock_wait_does_not_block_event_loop(monkeypatch, tmp_path):
    state = {
        "active": 0,
        "max_active": 0,
        "entered": {},
    }
    state_lock = threading.Lock()
    release_event = threading.Event()
    heartbeat = {"ticks": 0}
    stop_heartbeat = asyncio.Event()

    class AsyncBlockingStreamAgent:
        def __init__(self, **kwargs):
            self.session_db = kwargs["session_db"]
            self.session_id = kwargs["session_id"]
            self._memory_store = None
            self.tools = []
            self.suppress_status_output = False
            self.model = kwargs["model"]
            self.provider = kwargs["provider"]
            self.base_url = kwargs["base_url"]

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
            with state_lock:
                state["active"] += 1
                state["max_active"] = max(state["max_active"], state["active"])
                state["entered"].setdefault(self.session_id, []).append(
                    [user_message, [dict(item) for item in (conversation_history or [])]]
                )
            if stream_callback:
                stream_callback("chunk-a")
            release_event.wait(2.0)
            self.session_db.append_message(
                session_id=self.session_id,
                role="user",
                content=user_message,
                platform_message_id=platform_message_id,
            )
            self.session_db.append_message(
                session_id=self.session_id,
                role="assistant",
                content="reply to {}".format(user_message),
            )
            if stream_callback:
                stream_callback("chunk-b")
            with state_lock:
                state["active"] -= 1
            return {"final_response": "reply to {}".format(user_message)}

    import run_agent

    monkeypatch.setattr(run_agent, "AIAgent", AsyncBlockingStreamAgent)
    monkeypatch.setattr("hermes_cli.companion_api.get_profile_path", lambda sid: str(tmp_path / sid))
    def fake_review_turn(**kwargs):
        store = TurnReviewStore(Path(kwargs["profile_dir"]))
        store.begin(kwargs["turn_id"], kwargs["assistant_text"])
        store.commit(
            {
                "turn_id": kwargs["turn_id"],
                "assistant_sha256": assistant_sha256(kwargs["assistant_text"]),
                "style_decision": "clean",
                "style_reason": "ok",
                "continuity_summary": "",
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
            "turn_id": kwargs["turn_id"],
            "review_status": "clean",
            "memory_status": "none",
            "memory_modifications": [],
        }

    monkeypatch.setattr("hermes_cli.companion_api.review_turn", fake_review_turn)

    async def heartbeat_task():
        while not stop_heartbeat.is_set():
            heartbeat["ticks"] += 1
            await asyncio.sleep(0.01)

    async def consume_stream(user_id, message_id, text):
        payload = _payload(message_id, text, stream=True)
        payload["user_id"] = user_id
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            async with client.stream("POST", "/companion/v1/chat", json=payload) as response:
                body = ""
                async for chunk in response.aiter_text():
                    body += chunk
                return response.status_code, body

    hb_task = asyncio.create_task(heartbeat_task())
    task1 = asyncio.create_task(consume_stream("UserA", "msg-1", "第一句"))
    await asyncio.sleep(0.1)
    task2 = asyncio.create_task(consume_stream("UserA", "msg-2", "第二句"))
    before = heartbeat["ticks"]
    await asyncio.sleep(0.1)
    after = heartbeat["ticks"]
    assert after > before
    with state_lock:
        assert len(state["entered"].get("savana_usera_chara", [])) == 1

    release_event.set()
    result1, result2 = await asyncio.gather(task1, task2)
    assert result1[0] == 200
    assert result2[0] == 200
    with state_lock:
        assert state["entered"]["savana_usera_chara"][1][1][-1]["content"] == "reply to 第一句"

    state["active"] = 0
    state["max_active"] = 0
    state["entered"] = {}
    release_event.clear()
    task3 = asyncio.create_task(consume_stream("UserB", "msg-1", "你好"))
    task4 = asyncio.create_task(consume_stream("UserC", "msg-1", "你好"))
    await asyncio.sleep(0.1)
    with state_lock:
        assert state["max_active"] >= 2
    release_event.set()
    result3, result4 = await asyncio.gather(task3, task4)
    assert result3[0] == 200
    assert result4[0] == 200
    stop_heartbeat.set()
    await hb_task


@pytest.mark.asyncio
async def test_style_guard_cancelled_lock_wait_releases_orphan_acquire(monkeypatch, tmp_path):
    profile_dir = tmp_path / "cancelled-lock-wait-profile"
    ensure_companion_profile(profile_dir)
    monkeypatch.setattr("hermes_cli.companion_api.get_profile_path", lambda _sid: str(profile_dir))
    lock = ObservableCountingLock()
    lock.hold_for_test()
    monkeypatch.setattr("hermes_cli.companion_api._get_session_lock", lambda _sid: lock)
    _install_fake_agent(monkeypatch)
    monkeypatch.setattr(
        "hermes_cli.companion_api.review_turn",
        lambda **kwargs: {
            "turn_id": kwargs["turn_id"],
            "review_status": "clean",
            "memory_status": "none",
            "memory_modifications": [],
        },
    )

    async def post_message(message_id):
        payload = _payload(message_id, "第一句", stream=False)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.post("/companion/v1/chat", json=payload)

    waiting_task = asyncio.create_task(post_message("msg-1"))
    assert await asyncio.to_thread(lock.acquire_started.wait, 1.0)

    waiting_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(waiting_task, 1.0)

    lock.release()
    assert await asyncio.to_thread(lock.acquire_completed.wait, 1.0)
    await asyncio.sleep(0.1)
    assert lock.acquire_count == 1
    assert lock.release_count == 2

    follow_up = await asyncio.wait_for(post_message("msg-2"), 2.0)
    assert follow_up.status_code == 200
    assert lock.acquire_count == 2
    assert lock.release_count == 3


def test_style_guard_disconnect_keeps_lock_until_slow_producer_finishes(monkeypatch, tmp_path):
    profile_dir = tmp_path / "slow-producer-disconnect-profile"
    ensure_companion_profile(profile_dir)
    monkeypatch.setattr("hermes_cli.companion_api.get_profile_path", lambda _sid: str(profile_dir))
    lock = ObservableCountingLock()
    monkeypatch.setattr("hermes_cli.companion_api._get_session_lock", lambda _sid: lock)
    producer_may_finish = threading.Event()
    first_producer_entered = threading.Event()
    second_producer_entered = threading.Event()

    class SlowStreamingAgent:
        def __init__(self, **kwargs):
            self.session_db = kwargs["session_db"]
            self.session_id = kwargs["session_id"]
            self._memory_store = None
            self.tools = []
            self.suppress_status_output = False
            self.model = kwargs["model"]
            self.provider = kwargs["provider"]
            self.base_url = kwargs["base_url"]

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
            if platform_message_id == "msg-1":
                first_producer_entered.set()
            else:
                second_producer_entered.set()
            if stream_callback:
                stream_callback("chunk-a")
            producer_may_finish.wait(5.0)
            self.session_db.append_message(
                session_id=self.session_id,
                role="user",
                content=user_message,
                platform_message_id=platform_message_id,
            )
            self.session_db.append_message(
                session_id=self.session_id,
                role="assistant",
                content="reply to {}".format(user_message),
            )
            if stream_callback:
                stream_callback("chunk-b")
            return {"final_response": "reply to {}".format(user_message)}

    import run_agent

    monkeypatch.setattr(run_agent, "AIAgent", SlowStreamingAgent)
    monkeypatch.setattr(
        "hermes_cli.companion_api.review_turn",
        lambda **kwargs: {
            "turn_id": kwargs["turn_id"],
            "review_status": "clean",
            "memory_status": "none",
            "memory_modifications": [],
        },
    )

    first_finished = threading.Event()
    second_finished = threading.Event()
    results = {}

    def first_request():
        try:
            with TestClient(app).stream(
                "POST",
                "/companion/v1/chat",
                json=_payload("msg-1", "第一句", stream=True),
            ) as response:
                iterator = response.iter_text()
                first_chunk = next(iterator)
                results["first_status"] = response.status_code
                results["first_chunk"] = first_chunk
        finally:
            first_finished.set()

    def second_request():
        try:
            response = TestClient(app).post(
                "/companion/v1/chat",
                json=_payload("msg-2", "第二句", stream=False),
            )
            results["second_status"] = response.status_code
        finally:
            second_finished.set()

    thread_a = threading.Thread(target=first_request)
    thread_a.start()
    assert first_producer_entered.wait(1.0)
    thread_b = threading.Thread(target=second_request)
    thread_b.start()
    time.sleep(0.2)

    assert not second_producer_entered.is_set()
    assert not second_finished.is_set()
    assert lock.acquire_count == 2
    assert lock.release_count == 0

    producer_may_finish.set()
    assert first_finished.wait(2.0)
    assert second_finished.wait(2.0)
    thread_a.join(timeout=2.0)
    thread_b.join(timeout=2.0)

    assert results["first_status"] == 200
    assert "event: token" in results["first_chunk"]
    assert results["second_status"] == 200
    assert second_producer_entered.is_set()
    assert lock.acquire_count == 2
    assert lock.release_count == 2


@pytest.mark.asyncio
async def test_style_guard_asgi_send_start_failure_cleans_up_without_iterating_body(monkeypatch, tmp_path):
    profile_dir = tmp_path / "asgi-send-start-failure-profile"
    ensure_companion_profile(profile_dir)
    monkeypatch.setattr("hermes_cli.companion_api.get_profile_path", lambda _sid: str(profile_dir))
    lock = ObservableCountingLock()
    monkeypatch.setattr("hermes_cli.companion_api._get_session_lock", lambda _sid: lock)
    final_marker_seen = threading.Event()
    producer_done_seen = threading.Event()
    lock_released_seen = threading.Event()

    class FastStreamAgent:
        def __init__(self, **kwargs):
            self.session_db = kwargs["session_db"]
            self.session_id = kwargs["session_id"]
            self._memory_store = None
            self.tools = []
            self.suppress_status_output = False
            self.model = kwargs["model"]
            self.provider = kwargs["provider"]
            self.base_url = kwargs["base_url"]

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
            if stream_callback:
                stream_callback("chunk-a")
            self.session_db.append_message(
                session_id=self.session_id,
                role="user",
                content=user_message,
                platform_message_id=platform_message_id,
            )
            self.session_db.append_message(
                session_id=self.session_id,
                role="assistant",
                content="reply to {}".format(user_message),
            )
            return {"final_response": "reply to {}".format(user_message)}

    import run_agent

    monkeypatch.setattr(run_agent, "AIAgent", FastStreamAgent)
    monkeypatch.setattr(
        "hermes_cli.companion_api.review_turn",
        lambda **kwargs: {
            "turn_id": kwargs["turn_id"],
            "review_status": "clean",
            "memory_status": "none",
            "memory_modifications": [],
        },
    )

    def stream_hook(name, payload=None):
        if name == "final_marker_set":
            final_marker_seen.set()
        if name == "producer_done":
            producer_done_seen.set()
        if name == "lock_released":
            lock_released_seen.set()

    monkeypatch.setattr("hermes_cli.companion_api._stream_event_hook", stream_hook, raising=False)

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def fail_on_start(message):
        if message["type"] == "http.response.start":
            raise OSError("send start failed")

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.4"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/companion/v1/chat",
        "raw_path": b"/companion/v1/chat",
        "query_string": b"",
        "headers": [(b"content-type", b"application/json")],
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 80),
    }
    body = json.dumps(_payload("msg-1", "第一句", stream=True)).encode("utf-8")
    sent_body = {"done": False}

    async def receive_once():
        if sent_body["done"]:
            await asyncio.sleep(0)
            return {"type": "http.disconnect"}
        sent_body["done"] = True
        return {"type": "http.request", "body": body, "more_body": False}

    with pytest.raises(OSError):
        await asyncio.wait_for(_invoke_asgi_app(scope, receive_once, fail_on_start), 2.0)

    assert await asyncio.to_thread(final_marker_seen.wait, 1.0)
    assert await asyncio.to_thread(producer_done_seen.wait, 1.0)
    assert await asyncio.to_thread(lock_released_seen.wait, 1.0)
    assert lock.acquire_count == 1
    assert lock.release_count == 1

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        follow_up = await asyncio.wait_for(
            client.post("/companion/v1/chat", json=_payload("msg-2", "第二句", stream=False)),
            2.0,
        )
    assert follow_up.status_code == 200
    assert lock.acquire_count == 2
    assert lock.release_count == 2


@pytest.mark.asyncio
async def test_style_guard_asgi_disconnect_cleanup_wait_does_not_block_event_loop(monkeypatch, tmp_path):
    profile_dir = tmp_path / "asgi-disconnect-cleanup-profile"
    ensure_companion_profile(profile_dir)
    monkeypatch.setattr("hermes_cli.companion_api.get_profile_path", lambda _sid: str(profile_dir))
    lock = ObservableCountingLock()
    monkeypatch.setattr("hermes_cli.companion_api._get_session_lock", lambda _sid: lock)
    producer_may_finish = threading.Event()
    first_producer_entered = threading.Event()
    second_producer_entered = threading.Event()
    final_marker_seen = threading.Event()
    producer_done_seen = threading.Event()
    lock_released_seen = threading.Event()
    heartbeat = {"ticks": 0}
    stop_heartbeat = asyncio.Event()

    class SlowStreamingAgent:
        def __init__(self, **kwargs):
            self.session_db = kwargs["session_db"]
            self.session_id = kwargs["session_id"]
            self._memory_store = None
            self.tools = []
            self.suppress_status_output = False
            self.model = kwargs["model"]
            self.provider = kwargs["provider"]
            self.base_url = kwargs["base_url"]

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
            if platform_message_id == "msg-1":
                first_producer_entered.set()
            else:
                second_producer_entered.set()
            if stream_callback:
                stream_callback("chunk-a")
            producer_may_finish.wait(5.0)
            self.session_db.append_message(
                session_id=self.session_id,
                role="user",
                content=user_message,
                platform_message_id=platform_message_id,
            )
            self.session_db.append_message(
                session_id=self.session_id,
                role="assistant",
                content="reply to {}".format(user_message),
            )
            if stream_callback:
                stream_callback("chunk-b")
            return {"final_response": "reply to {}".format(user_message)}

    import run_agent

    monkeypatch.setattr(run_agent, "AIAgent", SlowStreamingAgent)
    monkeypatch.setattr(
        "hermes_cli.companion_api.review_turn",
        lambda **kwargs: {
            "turn_id": kwargs["turn_id"],
            "review_status": "clean",
            "memory_status": "none",
            "memory_modifications": [],
        },
    )

    def stream_hook(name, payload=None):
        if name == "final_marker_set":
            final_marker_seen.set()
        if name == "producer_done":
            producer_done_seen.set()
        if name == "lock_released":
            lock_released_seen.set()

    monkeypatch.setattr("hermes_cli.companion_api._stream_event_hook", stream_hook, raising=False)

    async def heartbeat_task():
        while not stop_heartbeat.is_set():
            heartbeat["ticks"] += 1
            await asyncio.sleep(0.01)

    body = json.dumps(_payload("msg-1", "第一句", stream=True)).encode("utf-8")
    receive_calls = {"count": 0}

    async def receive_disconnect():
        if receive_calls["count"] == 0:
            receive_calls["count"] += 1
            return {"type": "http.request", "body": body, "more_body": False}
        await asyncio.sleep(0.01)
        return {"type": "http.disconnect"}

    sent_messages = []
    first_cleanup_done = asyncio.Event()

    async def send_and_disconnect(message):
        sent_messages.append(message["type"])
        if message["type"] == "http.response.body":
            raise OSError("client disconnected")

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.4"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/companion/v1/chat",
        "raw_path": b"/companion/v1/chat",
        "query_string": b"",
        "headers": [(b"content-type", b"application/json")],
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 80),
    }

    async def first_request():
        try:
            await _invoke_asgi_app(scope, receive_disconnect, send_and_disconnect)
        except OSError:
            pass
        finally:
            first_cleanup_done.set()

    async def second_request():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.post("/companion/v1/chat", json=_payload("msg-2", "第二句", stream=False))

    hb_task = asyncio.create_task(heartbeat_task())
    first_task = asyncio.create_task(first_request())
    assert await asyncio.to_thread(first_producer_entered.wait, 1.0)
    await asyncio.sleep(0.1)
    second_task = asyncio.create_task(second_request())
    before = heartbeat["ticks"]
    await asyncio.sleep(0.1)
    after = heartbeat["ticks"]

    assert after > before
    assert not second_producer_entered.is_set()
    assert not second_task.done()
    assert not first_cleanup_done.is_set()
    assert lock.acquire_count == 2
    assert lock.release_count == 0

    producer_may_finish.set()
    await asyncio.wait_for(first_task, 2.0)
    response = await asyncio.wait_for(second_task, 2.0)
    assert response.status_code == 200
    assert await asyncio.to_thread(final_marker_seen.wait, 1.0)
    assert await asyncio.to_thread(producer_done_seen.wait, 1.0)
    assert await asyncio.to_thread(lock_released_seen.wait, 1.0)
    assert second_producer_entered.is_set()
    assert lock.acquire_count == 2
    assert lock.release_count == 2
    stop_heartbeat.set()
    await hb_task
