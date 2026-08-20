from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from hermes_state import SessionDB
from hermes_cli.web_server import app
from tests.hermes_cli.story_projection_fixtures import (
    DEFAULT_APPLIED_AT,
    PROJECTION_PROTOCOL_VERSION,
    create_story_projection_fixture,
)


def _projection_columns(db_path: Path) -> dict[str, dict[str, int | str | None]]:
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "PRAGMA table_info(companion_memory_projections)"
        ).fetchall()
    return {
        row[1]: {
            "type": row[2],
            "notnull": row[3],
            "default": row[4],
            "pk": row[5],
        }
        for row in rows
    }


def _sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _projection_payload(
    *,
    user_id: str = "5a7ba070-8650-4f95-af38-3673f7bc5f18",
    character_id: str = "57ffc9e5-9b4a-48bb-b597-fbdfdd47f9ee",
    projection_id: str = "784b9d19-1ce8-4d43-a366-60f4aa5b67c1",
    memory_version: int = 42,
    content: str = "角色答应在11月7日给用户准备低糖生日蛋糕",
    protocol_version: int = 2,
    fact_subject: str = "character",
    story_state: str = "active",
    story_kind: str = "commitment",
) -> dict[str, object]:
    entries = [
        {
            "memory_id": "4a52c78f-d07b-55e1-b1fd-5073bffb9e3e",
            "story_kind": story_kind,
            "story_state": story_state,
            "fact_subject": fact_subject,
            "content": content,
            "content_hash": _sha256_hex(content),
            "importance": 9,
            "event_time": "11月7日",
            "scope_kind": "pair",
            "scope_id": None,
        }
    ]
    payload = {
        "protocol_version": protocol_version,
        "projection_id": projection_id,
        "target": "continuity",
        "user_id": user_id,
        "character_id": character_id,
        "memory_version": memory_version,
        "snapshot_checksum": "",
        "generated_at": "2026-08-20T14:03:12.123456+08:00",
        "budget": {
            "rendered_char_limit": 4000,
            "omitted_count": 0,
            "omitted_kinds": [],
        },
        "entries": entries,
    }
    checksum_document = {
        "protocol_version": payload["protocol_version"],
        "target": payload["target"],
        "user_id": payload["user_id"],
        "character_id": payload["character_id"],
        "budget": payload["budget"],
        "entries": payload["entries"],
    }
    if protocol_version == 1:
        payload["entries"][0].pop("fact_subject", None)
        checksum_document["entries"] = payload["entries"]
    payload["snapshot_checksum"] = hashlib.sha256(
        json.dumps(
            checksum_document,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return payload


def test_projection_v2_renders_subject_and_pending_state():
    payload = _projection_payload(
        content="用户提议周日和角色一起去看展",
        fact_subject="user",
        story_state="pending_confirmation",
        story_kind="agreement",
    )

    from hermes_cli.companion_api import _normalize_projection_payload, ContinuityProjectionRequest
    from tools.memory_tool import render_continuity_projection_text

    normalized = _normalize_projection_payload(ContinuityProjectionRequest(**payload))
    assert render_continuity_projection_text(normalized["entries"]) == (
        "[用户提议·待确认] 用户提议周日和角色一起去看展"
    )


def test_projection_receiver_keeps_v1_compatibility():
    payload = _projection_payload(protocol_version=1)

    from hermes_cli.companion_api import _normalize_projection_payload, ContinuityProjectionRequest
    from tools.memory_tool import render_continuity_projection_text

    normalized = _normalize_projection_payload(ContinuityProjectionRequest(**payload))
    assert "fact_subject" not in normalized["entries"][0]
    assert render_continuity_projection_text(normalized["entries"]).startswith("[承诺·进行中]")


@pytest.mark.parametrize(
    ("story_kind", "story_state", "fact_subject"),
    [
        ("agreement", "pending_confirmation", "mutual"),
        ("agreement", "completed", "user"),
        ("relationship_milestone", "active", "character"),
    ],
)
def test_projection_v2_rejects_invalid_subject_state_combinations(
    story_kind: str,
    story_state: str,
    fact_subject: str,
):
    from hermes_cli.companion_api import _normalize_projection_payload, ContinuityProjectionRequest

    payload = _projection_payload(
        story_kind=story_kind,
        story_state=story_state,
        fact_subject=fact_subject,
    )

    with pytest.raises(ValueError, match="invalid_continuity_snapshot"):
        _normalize_projection_payload(ContinuityProjectionRequest(**payload))


def test_chat_request_accepts_readonly_mode_only():
    from pydantic import ValidationError
    from hermes_cli.companion_api import ChatRequest

    request = ChatRequest(
        user_id="user",
        character_id="character",
        message_id="message",
        user_message="probe",
        character_profile={},
        memory_write_mode="readonly",
    )
    assert request.memory_write_mode == "readonly"
    with pytest.raises(ValidationError):
        ChatRequest(
            user_id="user",
            character_id="character",
            message_id="message",
            user_message="probe",
            character_profile={},
            memory_write_mode="off",
        )


def _projection_headers(payload: dict[str, object]) -> dict[str, str]:
    return {
        "X-Projection-Id": str(payload["projection_id"]),
        "X-Projection-Protocol": str(payload["protocol_version"]),
    }


def _make_legacy_profile(
    tmp_path: Path,
    monkeypatch,
    *,
    profile_name: str = "legacy-story-profile",
    continuity_text: str = "旧剧情还在这里",
) -> tuple[Path, Path]:
    isolated_home = tmp_path / "home"
    hermes_root = isolated_home / ".hermes"
    profile_dir = hermes_root / "profiles" / profile_name
    memories_dir = profile_dir / "memories"
    db_path = profile_dir / "state.db"

    memories_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(Path, "home", lambda: isolated_home)
    monkeypatch.setenv("HERMES_HOME", str(profile_dir))

    (memories_dir / "CONTINUITY.md").write_text(continuity_text, encoding="utf-8")
    (memories_dir / "USER.md").write_text("", encoding="utf-8")
    (memories_dir / "MEMORY.md").write_text("", encoding="utf-8")

    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE schema_version (version INTEGER NOT NULL)")
        conn.execute("INSERT INTO schema_version(version) VALUES (14)")
        conn.execute(
            """
            CREATE TABLE sessions (
                id TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                started_at REAL NOT NULL
            )
            """
        )
        conn.execute(
            "INSERT INTO sessions(id, source, started_at) VALUES (?, ?, ?)",
            ("legacy-session", "test", 0.0),
        )
        conn.commit()

    return profile_dir, db_path


class TestCompanionProjectionSchema:
    def test_creates_projection_table_with_session_target_primary_key(self, tmp_path, monkeypatch):
        fixture = create_story_projection_fixture(
            tmp_path,
            monkeypatch,
            create_session=False,
        )

        with sqlite3.connect(fixture.db_path) as conn:
            conn.execute("DROP TABLE companion_memory_projections")
            conn.commit()

        session_db = SessionDB(db_path=fixture.db_path)
        try:
            columns = _projection_columns(fixture.db_path)
        finally:
            session_db.close()

        assert columns["session_id"]["pk"] == 1
        assert columns["target"]["pk"] == 2
        assert columns["protocol_version"]["notnull"] == 1
        assert columns["snapshot_json"]["notnull"] == 1
        assert columns["materialized_at"]["type"] == "TEXT"

    def test_reconciles_missing_projection_columns_on_legacy_table(self, tmp_path, monkeypatch):
        fixture = create_story_projection_fixture(
            tmp_path,
            monkeypatch,
            create_session=False,
        )

        with sqlite3.connect(fixture.db_path) as conn:
            conn.execute("DROP TABLE companion_memory_projections")
            conn.execute(
                """
                CREATE TABLE companion_memory_projections (
                    session_id TEXT NOT NULL,
                    target TEXT NOT NULL,
                    projection_id TEXT NOT NULL,
                    PRIMARY KEY (session_id, target)
                )
                """
            )
            conn.commit()

        session_db = SessionDB(db_path=fixture.db_path)
        try:
            columns = _projection_columns(fixture.db_path)
        finally:
            session_db.close()

        assert "protocol_version" in columns
        assert "memory_version" in columns
        assert "snapshot_checksum" in columns
        assert "snapshot_json" in columns
        assert "rendered_checksum" in columns
        assert "applied_at" in columns
        assert "materialized_at" in columns

    def test_upgrades_legacy_profile_without_touching_existing_continuity_file(self, tmp_path, monkeypatch):
        profile_dir, db_path = _make_legacy_profile(tmp_path, monkeypatch)
        continuity_path = profile_dir / "memories" / "CONTINUITY.md"
        original_text = continuity_path.read_text(encoding="utf-8")

        session_db = SessionDB(db_path=db_path)
        try:
            projection = session_db.get_companion_memory_projection(
                session_id="legacy-session",
                target="continuity",
            )
        finally:
            session_db.close()

        assert continuity_path.read_text(encoding="utf-8") == original_text
        assert projection is None


class TestCompanionProjectionStateApi:
    def test_stores_independent_rows_for_same_session_across_targets(self, tmp_path, monkeypatch):
        fixture = create_story_projection_fixture(
            tmp_path,
            monkeypatch,
            continuity_text="原始剧情",
        )

        session_db = SessionDB(db_path=fixture.db_path)
        try:
            session_db.upsert_companion_memory_projection(
                session_id=fixture.session_id,
                target="memory",
                protocol_version=PROJECTION_PROTOCOL_VERSION,
                projection_id="memory-projection-id",
                memory_version=3,
                snapshot_checksum="memory-checksum",
                snapshot_json='{"target":"memory"}',
                rendered_checksum="memory-rendered",
                applied_at=DEFAULT_APPLIED_AT,
                materialized_at=DEFAULT_APPLIED_AT,
            )
            continuity_projection = session_db.get_companion_memory_projection(
                session_id=fixture.session_id,
                target="continuity",
            )
            memory_projection = session_db.get_companion_memory_projection(
                session_id=fixture.session_id,
                target="memory",
            )
        finally:
            session_db.close()

        assert continuity_projection["projection_id"] == fixture.projection_id
        assert memory_projection["projection_id"] == "memory-projection-id"
        assert memory_projection["target"] == "memory"

    def test_replaces_existing_snapshot_for_same_session_and_target(self, tmp_path, monkeypatch):
        fixture = create_story_projection_fixture(
            tmp_path,
            monkeypatch,
            continuity_text="第一版剧情",
            memory_version=1,
        )

        session_db = SessionDB(db_path=fixture.db_path)
        try:
            session_db.upsert_companion_memory_projection(
                session_id=fixture.session_id,
                target="continuity",
                protocol_version=PROJECTION_PROTOCOL_VERSION,
                projection_id="replacement-projection-id",
                memory_version=2,
                snapshot_checksum="replacement-snapshot-checksum",
                snapshot_json='{"target":"continuity","memory_version":2}',
                rendered_checksum="replacement-rendered-checksum",
                applied_at=DEFAULT_APPLIED_AT,
                materialized_at=None,
            )
            projection = session_db.get_companion_memory_projection(
                session_id=fixture.session_id,
                target="continuity",
            )
            with sqlite3.connect(fixture.db_path) as conn:
                row_count = conn.execute(
                    """
                    SELECT COUNT(*)
                    FROM companion_memory_projections
                    WHERE session_id = ? AND target = 'continuity'
                    """,
                    (fixture.session_id,),
                ).fetchone()[0]
        finally:
            session_db.close()

        assert row_count == 1
        assert projection["projection_id"] == "replacement-projection-id"
        assert projection["memory_version"] == 2
        assert projection["snapshot_checksum"] == "replacement-snapshot-checksum"
        assert projection["materialized_at"] is None


class TestContinuityProjectionRendering:
    def test_renders_all_story_kinds_with_chinese_labels_and_hides_scope_metadata(self):
        payload = _projection_payload()
        payload["entries"] = [
            {
                "memory_id": "11111111-1111-1111-1111-111111111111",
                "story_kind": "relationship_milestone",
                "story_state": "active",
                "fact_subject": "mutual",
                "content": "两人已经正式确认恋人关系",
                "content_hash": _sha256_hex("两人已经正式确认恋人关系"),
                "importance": 10,
                "event_time": None,
                "scope_kind": "pair",
                "scope_id": None,
            },
            {
                "memory_id": "22222222-2222-2222-2222-222222222222",
                "story_kind": "commitment",
                "story_state": "active",
                "fact_subject": "character",
                "content": "角色答应在11月7日准备低糖生日蛋糕",
                "content_hash": _sha256_hex("角色答应在11月7日准备低糖生日蛋糕"),
                "importance": 9,
                "event_time": "11月7日",
                "scope_kind": "pair",
                "scope_id": None,
            },
            {
                "memory_id": "33333333-3333-3333-3333-333333333333",
                "story_kind": "agreement",
                "story_state": "cancelled",
                "fact_subject": "mutual",
                "content": "原定周六一起去看展的约定已取消",
                "content_hash": _sha256_hex("原定周六一起去看展的约定已取消"),
                "importance": 8,
                "event_time": None,
                "scope_kind": "story_branch",
                "scope_id": "branch-secret",
            },
            {
                "memory_id": "44444444-4444-4444-4444-444444444444",
                "story_kind": "unresolved_thread",
                "story_state": "active",
                "fact_subject": "mutual",
                "content": "下周继续调查旧车站线索",
                "content_hash": _sha256_hex("下周继续调查旧车站线索"),
                "importance": 8,
                "event_time": None,
                "scope_kind": "story_branch",
                "scope_id": "branch-secret",
            },
            {
                "memory_id": "55555555-5555-5555-5555-555555555555",
                "story_kind": "shared_event",
                "story_state": "completed",
                "fact_subject": "mutual",
                "content": "两人已经找回遗失的相册",
                "content_hash": _sha256_hex("两人已经找回遗失的相册"),
                "importance": 7,
                "event_time": None,
                "scope_kind": "pair",
                "scope_id": None,
            },
            {
                "memory_id": "66666666-6666-6666-6666-666666666666",
                "story_kind": "gift_or_secret",
                "story_state": "active",
                "fact_subject": "mutual",
                "content": "角色把旧怀表交给用户保管",
                "content_hash": _sha256_hex("角色把旧怀表交给用户保管"),
                "importance": 7,
                "event_time": None,
                "scope_kind": "pair",
                "scope_id": None,
            },
        ]
        payload["budget"]["rendered_char_limit"] = 6000
        payload["snapshot_checksum"] = _sha256_hex(
            json.dumps(
                {
                    "protocol_version": payload["protocol_version"],
                    "target": payload["target"],
                    "user_id": payload["user_id"],
                    "character_id": payload["character_id"],
                    "budget": payload["budget"],
                    "entries": payload["entries"],
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )

        from tools.memory_tool import render_continuity_projection_text

        rendered = render_continuity_projection_text(payload["entries"])

        assert rendered.split("\n§\n") == [
            "[双方关系里程碑·进行中] 两人已经正式确认恋人关系",
            "[角色承诺·进行中] 角色答应在11月7日准备低糖生日蛋糕",
            "[双方约定·已取消] 原定周六一起去看展的约定已取消",
            "[双方未完剧情·进行中] 下周继续调查旧车站线索",
            "[双方共同经历·已完成] 两人已经找回遗失的相册",
            "[双方礼物或秘密·进行中] 角色把旧怀表交给用户保管",
        ]
        assert "branch-secret" not in rendered
        assert "scope_kind" not in rendered
        assert "importance" not in rendered

    def test_put_projection_rejects_budget_above_6000_without_echoing_content(self, tmp_path, monkeypatch):
        payload = _projection_payload(content="绝对不能回显的超长剧情")
        session_id = "savana_{0}_{1}".format(payload["user_id"], payload["character_id"])
        create_story_projection_fixture(
            tmp_path,
            monkeypatch,
            profile_name=session_id,
            session_id=session_id,
        )
        payload["budget"]["rendered_char_limit"] = 6001
        payload["snapshot_checksum"] = _sha256_hex(
            json.dumps(
                {
                    "protocol_version": payload["protocol_version"],
                    "target": payload["target"],
                    "user_id": payload["user_id"],
                    "character_id": payload["character_id"],
                    "budget": payload["budget"],
                    "entries": payload["entries"],
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )

        response = TestClient(app).put(
            f"/companion/v1/sessions/{session_id}/memory-projections/continuity",
            json=payload,
            headers=_projection_headers(payload),
        )

        assert response.status_code == 422
        assert response.json()["error_code"] == "projection_budget_exceeded"
        assert "绝对不能回显的超长剧情" not in response.text


class TestCompanionProjectionEndpoint:
    def test_put_projection_applies_snapshot_and_persists_watermark(self, tmp_path, monkeypatch):
        payload = _projection_payload()
        session_id = "savana_{0}_{1}".format(payload["user_id"], payload["character_id"])
        fixture = create_story_projection_fixture(
            tmp_path,
            monkeypatch,
            profile_name=session_id,
            session_id=session_id,
            continuity_text="旧剧情",
            projection_id="old-projection-id",
            memory_version=5,
        )

        response = TestClient(app).put(
            f"/companion/v1/sessions/{session_id}/memory-projections/continuity",
            json=payload,
            headers=_projection_headers(payload),
        )

        assert response.status_code == 200
        assert response.json()["status"] == "applied"
        assert fixture.continuity_path.read_text(encoding="utf-8") == (
            "[角色承诺·进行中] 角色答应在11月7日给用户准备低糖生日蛋糕"
        )

        row = fixture.read_projection_row()
        assert row["projection_id"] == payload["projection_id"]
        assert row["memory_version"] == payload["memory_version"]
        assert row["snapshot_checksum"] == payload["snapshot_checksum"]
        assert row["materialized_at"]

    def test_put_projection_rejects_identity_mismatch_without_touching_profile(self, tmp_path, monkeypatch):
        payload = _projection_payload()
        session_id = "savana_other_user_other_character"
        fixture = create_story_projection_fixture(
            tmp_path,
            monkeypatch,
            profile_name=session_id,
            session_id=session_id,
            continuity_text="原始剧情",
            projection_id="fixture-projection-id",
            memory_version=7,
        )

        response = TestClient(app).put(
            f"/companion/v1/sessions/{session_id}/memory-projections/continuity",
            json=payload,
            headers=_projection_headers(payload),
        )

        assert response.status_code == 422
        assert response.json()["error_code"] == "projection_identity_mismatch"
        assert fixture.continuity_path.read_text(encoding="utf-8") == "原始剧情"
        row = fixture.read_projection_row()
        assert row["projection_id"] == "fixture-projection-id"
        assert row["memory_version"] == 7

    def test_put_projection_repairs_missing_file_on_same_version_retry(self, tmp_path, monkeypatch):
        payload = _projection_payload()
        session_id = "savana_{0}_{1}".format(payload["user_id"], payload["character_id"])
        fixture = create_story_projection_fixture(
            tmp_path,
            monkeypatch,
            profile_name=session_id,
            session_id=session_id,
            continuity_text="[角色承诺·进行中] 角色答应在11月7日给用户准备低糖生日蛋糕",
            projection_id=str(payload["projection_id"]),
            memory_version=int(payload["memory_version"]),
            snapshot_payload={
                "protocol_version": payload["protocol_version"],
                "target": payload["target"],
                "user_id": payload["user_id"],
                "character_id": payload["character_id"],
                "budget": payload["budget"],
                "entries": payload["entries"],
            },
        )
        fixture.continuity_path.unlink()

        response = TestClient(app).put(
            f"/companion/v1/sessions/{session_id}/memory-projections/continuity",
            json=payload,
            headers=_projection_headers(payload),
        )

        assert response.status_code == 200
        assert response.json()["status"] == "already_applied"
        assert fixture.continuity_path.read_text(encoding="utf-8") == (
            "[角色承诺·进行中] 角色答应在11月7日给用户准备低糖生日蛋糕"
        )

    def test_put_projection_repairs_drifted_file_on_same_version_retry(self, tmp_path, monkeypatch):
        payload = _projection_payload()
        session_id = "savana_{0}_{1}".format(payload["user_id"], payload["character_id"])
        fixture = create_story_projection_fixture(
            tmp_path,
            monkeypatch,
            profile_name=session_id,
            session_id=session_id,
            continuity_text="[角色承诺·进行中] 角色答应在11月7日给用户准备低糖生日蛋糕",
            projection_id=str(payload["projection_id"]),
            memory_version=int(payload["memory_version"]),
            snapshot_payload={
                "protocol_version": payload["protocol_version"],
                "target": payload["target"],
                "user_id": payload["user_id"],
                "character_id": payload["character_id"],
                "budget": payload["budget"],
                "entries": payload["entries"],
            },
        )
        fixture.continuity_path.write_text("被手工改坏的剧情内容", encoding="utf-8")

        response = TestClient(app).put(
            f"/companion/v1/sessions/{session_id}/memory-projections/continuity",
            json=payload,
            headers=_projection_headers(payload),
        )

        assert response.status_code == 200
        assert response.json()["status"] == "already_applied"
        assert fixture.continuity_path.read_text(encoding="utf-8") == (
            "[角色承诺·进行中] 角色答应在11月7日给用户准备低糖生日蛋糕"
        )

    def test_put_projection_rejects_invalid_generated_at_without_echoing_content(self, tmp_path, monkeypatch):
        payload = _projection_payload(content="绝对不能在报错里回显的剧情")
        session_id = "savana_{0}_{1}".format(payload["user_id"], payload["character_id"])
        create_story_projection_fixture(
            tmp_path,
            monkeypatch,
            profile_name=session_id,
            session_id=session_id,
        )
        payload["generated_at"] = "not-a-timestamp"
        payload["snapshot_checksum"] = _sha256_hex(
            json.dumps(
                {
                    "protocol_version": payload["protocol_version"],
                    "target": payload["target"],
                    "user_id": payload["user_id"],
                    "character_id": payload["character_id"],
                    "budget": payload["budget"],
                    "entries": payload["entries"],
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )

        response = TestClient(app).put(
            f"/companion/v1/sessions/{session_id}/memory-projections/continuity",
            json=payload,
            headers=_projection_headers(payload),
        )

        assert response.status_code == 422
        assert response.json()["error_code"] == "invalid_continuity_snapshot"
        assert "绝对不能在报错里回显的剧情" not in response.text

    def test_put_projection_returns_retryable_503_for_sqlite_busy(self, tmp_path, monkeypatch):
        payload = _projection_payload()
        session_id = "savana_{0}_{1}".format(payload["user_id"], payload["character_id"])
        create_story_projection_fixture(
            tmp_path,
            monkeypatch,
            profile_name=session_id,
            session_id=session_id,
            continuity_text="旧剧情",
            projection_id="old-projection-id",
            memory_version=1,
        )

        class BusySessionDB:
            def __init__(self, *args, **kwargs):
                pass

            def get_companion_memory_projection(self, *args, **kwargs):
                return None

            def close(self):
                return None

            def upsert_companion_memory_projection(self, *args, **kwargs):
                raise sqlite3.OperationalError("database is locked")

        monkeypatch.setattr("hermes_state.SessionDB", BusySessionDB)

        response = TestClient(app).put(
            f"/companion/v1/sessions/{session_id}/memory-projections/continuity",
            json=payload,
            headers=_projection_headers(payload),
        )

        assert response.status_code == 503
        assert response.json() == {
            "status": "projection_apply_failed",
            "error_code": "projection_sqlite_busy",
            "retryable": True,
        }

    def test_load_from_disk_repairs_missing_continuity_from_projection_snapshot(self, tmp_path, monkeypatch):
        from tools.memory_tool import MemoryStore

        payload = _projection_payload()
        session_id = "savana_{0}_{1}".format(payload["user_id"], payload["character_id"])
        fixture = create_story_projection_fixture(
            tmp_path,
            monkeypatch,
            profile_name=session_id,
            session_id=session_id,
            continuity_text="",
            projection_id=str(payload["projection_id"]),
            memory_version=int(payload["memory_version"]),
            materialized_at=None,
            snapshot_payload={
                "protocol_version": payload["protocol_version"],
                "target": payload["target"],
                "user_id": payload["user_id"],
                "character_id": payload["character_id"],
                "budget": payload["budget"],
                "entries": payload["entries"],
            },
        )
        fixture.continuity_path.unlink()

        store = MemoryStore()
        store.load_from_disk()

        continuity_text = fixture.continuity_path.read_text(encoding="utf-8")
        assert continuity_text == "[角色承诺·进行中] 角色答应在11月7日给用户准备低糖生日蛋糕"
        assert store.format_for_system_prompt("continuity")
        assert "角色答应在11月7日给用户准备低糖生日蛋糕" in store.format_for_system_prompt("continuity")

    def test_load_from_disk_repairs_drifted_continuity_from_projection_snapshot(self, tmp_path, monkeypatch):
        from tools.memory_tool import MemoryStore

        payload = _projection_payload()
        session_id = "savana_{0}_{1}".format(payload["user_id"], payload["character_id"])
        fixture = create_story_projection_fixture(
            tmp_path,
            monkeypatch,
            profile_name=session_id,
            session_id=session_id,
            continuity_text="完全错误的旧文件内容",
            projection_id=str(payload["projection_id"]),
            memory_version=int(payload["memory_version"]),
            materialized_at=DEFAULT_APPLIED_AT,
            snapshot_payload={
                "protocol_version": payload["protocol_version"],
                "target": payload["target"],
                "user_id": payload["user_id"],
                "character_id": payload["character_id"],
                "budget": payload["budget"],
                "entries": payload["entries"],
            },
        )
        fixture.continuity_path.write_text("被外部漂移污染的文件", encoding="utf-8")

        store = MemoryStore()
        store.load_from_disk()

        continuity_text = fixture.continuity_path.read_text(encoding="utf-8")
        assert continuity_text == "[角色承诺·进行中] 角色答应在11月7日给用户准备低糖生日蛋糕"
        assert store.format_for_system_prompt("continuity")
