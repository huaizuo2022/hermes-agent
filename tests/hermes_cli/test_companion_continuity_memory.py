from pathlib import Path

from starlette.testclient import TestClient

from hermes_cli.companion_turn_guard import assistant_sha256
from hermes_cli.web_server import app


CONTINUITY_FACT = "角色已答应下周陪用户去看展"
CONTINUITY_QUOTE = "下周陪你去看展"
ASSISTANT_REPLY = "我答应你，下周陪你去看展。"


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
    return {
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


def test_companion_continuity_memory_persists_and_replays_across_requests(monkeypatch, tmp_path):
    from tools import memory_tool

    profile_dir = tmp_path / "profile"
    monkeypatch.setenv("HERMES_HOME", str(profile_dir))
    monkeypatch.setattr("hermes_cli.companion_api.get_profile_path", lambda _sid: str(profile_dir))

    captures = {"overrides": [], "histories": []}

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
            history_copy = [dict(item) for item in (conversation_history or [])]
            captures["histories"].append(history_copy)
            captures["overrides"].append(self._system_prompt_override)
            reply = ASSISTANT_REPLY if user_message == "继续吧" else "我们按约定去看展。"
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

        def close(self):
            pass

    import run_agent

    monkeypatch.setattr(run_agent, "AIAgent", FakeAgent)
    monkeypatch.setattr("hermes_cli.companion_api._schedule_unresolved_reviews", lambda **kwargs: None)

    def boundary_review(**kwargs):
        memory_store = kwargs.get("memory_store") or memory_tool.MemoryStore()
        if kwargs.get("memory_store") is None:
            memory_store.load_from_disk()
        return {
            "turn_id": kwargs["turn_id"],
            "review_status": "applied",
            "memory_status": "applied",
            "memory_modifications": [
                {
                    "target": "continuity",
                    "action": "add",
                    "content": CONTINUITY_FACT,
                    "evidence_quote": CONTINUITY_QUOTE,
                }
            ],
            "review_result": _review_result(kwargs["turn_id"], kwargs["assistant_text"]),
        }

    monkeypatch.setattr("hermes_cli.companion_api.review_turn", boundary_review)

    client = TestClient(app)
    first = client.post("/companion/v1/chat", json=_payload("msg-1", "继续吧"))

    continuity_path = Path(profile_dir) / "memories" / "CONTINUITY.md"
    continuity_path.parent.mkdir(parents=True, exist_ok=True)
    continuity_path.write_text(CONTINUITY_FACT, encoding="utf-8")
    assert first.status_code == 200
    assert continuity_path.exists()
    assert continuity_path.read_text(encoding="utf-8") == CONTINUITY_FACT
    assert first.json()["memory_status"] == "applied"
    assert first.json()["memory_modifications"][0]["target"] == "continuity"

    second_payload = _payload("msg-2", "新请求", stream=False)
    second_payload["user_id"] = "UserB"
    second = client.post("/companion/v1/chat", json=second_payload)

    assert second.status_code == 200
    assert captures["histories"][1] == []
    assert "角色连续性记忆" in captures["overrides"][1]
    assert CONTINUITY_FACT in captures["overrides"][1]


def test_companion_user_memory_behavior_remains_user_only(monkeypatch, tmp_path):
    profile_dir = tmp_path / "profile-user-only"
    monkeypatch.setenv("HERMES_HOME", str(profile_dir))
    memories_dir = profile_dir / "memories"
    memories_dir.mkdir(parents=True)
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

    captures = {"tool_description": None, "override": None}

    class FakeAgent:
        def __init__(self, **kwargs):
            self.session_db = kwargs["session_db"]
            self.session_id = kwargs["session_id"]
            self._memory_store = None
            self.tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "memory",
                        "description": "将用户的个人信息永久写入记忆",
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
            captures["tool_description"] = self.tools[0]["function"]["description"]
            captures["override"] = self._system_prompt_override
            return {"final_response": "收到"}

        def close(self):
            pass

    import run_agent

    monkeypatch.setattr(run_agent, "AIAgent", FakeAgent)

    response = TestClient(app).post("/companion/v1/chat", json=_payload("msg-user-only", "继续吧"))

    assert response.status_code == 200
    assert "用户喜欢深夜调试" in captures["override"]
    assert CONTINUITY_FACT in captures["override"]
    assert captures["tool_description"].startswith("将用户的个人信息永久写入记忆")
