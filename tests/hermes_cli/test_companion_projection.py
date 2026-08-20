from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from hermes_state import SessionDB
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
