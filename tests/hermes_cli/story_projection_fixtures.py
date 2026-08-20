from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from hermes_state import SessionDB


LEGACY_PROJECTION_PROTOCOL_VERSION = 1
PROJECTION_PROTOCOL_VERSION = 2
DEFAULT_APPLIED_AT = "2026-08-20T00:00:00Z"
DEFAULT_SESSION_ID = "savana_user_char"
DEFAULT_PROFILE_NAME = "story-projection"
DEFAULT_PROJECTION_ID = "00000000-0000-0000-0000-000000000041"


def _sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _stable_snapshot_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _default_snapshot_payload(
    *,
    session_id: str,
    projection_id: str,
    memory_version: int,
    continuity_text: str,
) -> dict[str, Any]:
    return {
        "session_id": session_id,
        "target": "continuity",
        "projection_id": projection_id,
        "memory_version": memory_version,
        "entries": (
            [
                {
                    "memory_id": "fixture-memory-1",
                    "story_kind": "commitment",
                    "story_state": "active",
                    "fact_subject": "character",
                    "content": continuity_text,
                    "content_hash": _sha256_hex(continuity_text),
                    "importance": 10,
                    "scope_kind": "pair",
                    "scope_id": None,
                    "event_time": None,
                }
            ]
            if continuity_text
            else []
        ),
        "omitted": {"count": 0, "story_kinds": []},
    }


def _ensure_projection_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS companion_memory_projections (
            session_id TEXT NOT NULL,
            target TEXT NOT NULL,
            protocol_version INTEGER NOT NULL,
            projection_id TEXT NOT NULL,
            memory_version INTEGER NOT NULL,
            snapshot_checksum TEXT NOT NULL,
            snapshot_json TEXT NOT NULL,
            rendered_checksum TEXT,
            applied_at TEXT NOT NULL,
            materialized_at TEXT,
            PRIMARY KEY (session_id, target)
        )
        """
    )


@dataclass(frozen=True)
class StoryProjectionFixture:
    hermes_root: Path
    profile_dir: Path
    memories_dir: Path
    continuity_path: Path
    db_path: Path
    session_id: str
    profile_name: str
    projection_id: str
    memory_version: int
    snapshot_json: str
    snapshot_checksum: str
    rendered_checksum: str

    def read_projection_row(self) -> dict[str, Any]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                """
                SELECT session_id, target, protocol_version, projection_id, memory_version,
                       snapshot_checksum, snapshot_json, rendered_checksum,
                       applied_at, materialized_at
                FROM companion_memory_projections
                WHERE session_id = ? AND target = 'continuity'
                """,
                (self.session_id,),
            ).fetchone()
        if row is None:
            raise AssertionError("projection row missing from fixture state.db")
        return dict(row)


def create_story_projection_fixture(
    tmp_path: Path,
    monkeypatch,
    *,
    profile_name: str = DEFAULT_PROFILE_NAME,
    session_id: str = DEFAULT_SESSION_ID,
    continuity_text: str = "",
    projection_id: str = DEFAULT_PROJECTION_ID,
    memory_version: int = 1,
    snapshot_payload: Mapping[str, Any] | None = None,
    applied_at: str = DEFAULT_APPLIED_AT,
    materialized_at: str | None = DEFAULT_APPLIED_AT,
    create_session: bool = True,
) -> StoryProjectionFixture:
    """Create an isolated Hermes profile fixture for story projection tests.

    The fixture always points ``HERMES_HOME`` at a temporary profile directory
    under ``tmp_path`` and never touches ``~/.hermes``.
    """

    isolated_home = tmp_path / "home"
    hermes_root = isolated_home / ".hermes"
    profile_dir = hermes_root / "profiles" / profile_name
    memories_dir = profile_dir / "memories"
    continuity_path = memories_dir / "CONTINUITY.md"
    db_path = profile_dir / "state.db"

    memories_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(Path, "home", lambda: isolated_home)
    monkeypatch.setenv("HERMES_HOME", str(profile_dir))

    continuity_path.write_text(continuity_text, encoding="utf-8")
    (memories_dir / "USER.md").write_text("", encoding="utf-8")
    (memories_dir / "MEMORY.md").write_text("", encoding="utf-8")

    snapshot_payload = dict(snapshot_payload) if snapshot_payload is not None else _default_snapshot_payload(
        session_id=session_id,
        projection_id=projection_id,
        memory_version=memory_version,
        continuity_text=continuity_text,
    )
    snapshot_json = _stable_snapshot_json(snapshot_payload)
    snapshot_checksum = _sha256_hex(snapshot_json)
    rendered_checksum = _sha256_hex(continuity_text)

    session_db = SessionDB(db_path=db_path)
    try:
        if create_session:
            session_db.create_session(
                session_id=session_id,
                source="test",
                model="test-model",
            )
        _ensure_projection_table(session_db._conn)
        session_db._conn.execute(
            """
            INSERT OR REPLACE INTO companion_memory_projections (
                session_id,
                target,
                protocol_version,
                projection_id,
                memory_version,
                snapshot_checksum,
                snapshot_json,
                rendered_checksum,
                applied_at,
                materialized_at
            ) VALUES (?, 'continuity', ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                PROJECTION_PROTOCOL_VERSION,
                projection_id,
                memory_version,
                snapshot_checksum,
                snapshot_json,
                rendered_checksum,
                applied_at,
                materialized_at,
            ),
        )
        session_db._conn.commit()
    finally:
        session_db.close()

    return StoryProjectionFixture(
        hermes_root=hermes_root,
        profile_dir=profile_dir,
        memories_dir=memories_dir,
        continuity_path=continuity_path,
        db_path=db_path,
        session_id=session_id,
        profile_name=profile_name,
        projection_id=projection_id,
        memory_version=memory_version,
        snapshot_json=snapshot_json,
        snapshot_checksum=snapshot_checksum,
        rendered_checksum=rendered_checksum,
    )
