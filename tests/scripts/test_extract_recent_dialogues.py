import importlib.util
import json
import sqlite3
from pathlib import Path


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "_extract_recent_dialogues_under_test",
        Path(__file__).resolve().parents[2] / "scripts" / "extract_recent_dialogues.py",
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_profile(root, profile_id, character_name, intimacy, messages, policy=None):
    profile_dir = root / "profiles" / profile_id
    profile_dir.mkdir(parents=True)
    (profile_dir / "SOUL.md").write_text(
        "# {0}\n\n## Evolved Persona\n基础状态\n\n## Relationship with User\n"
        "- Intimacy level: {1}/10\n".format(character_name, intimacy),
        encoding="utf-8",
    )
    if policy:
        (profile_dir / "profile.yaml").write_text(
            "evolution_policy: {0}\n".format(policy),
            encoding="utf-8",
        )

    conn = sqlite3.connect(str(profile_dir / "state.db"))
    try:
        conn.execute(
            "CREATE TABLE messages (role TEXT, content TEXT, timestamp REAL)"
        )
        conn.executemany(
            "INSERT INTO messages (role, content, timestamp) VALUES (?, ?, ?)",
            messages,
        )
        conn.commit()
    finally:
        conn.close()


def _read_manifest(tmp_path, source_date):
    manifest_path = (
        tmp_path / "cron" / "state" / "savana-self-evolution" / (source_date + ".json")
    )
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def test_manifest_uses_previous_natural_day_and_outputs_only_current_batch(
    monkeypatch, tmp_path, capsys
):
    module = _load_module()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("SAVANA_EVOLUTION_TARGET_DATE", "2026-06-15")
    monkeypatch.setenv("SAVANA_EVOLUTION_SOURCE_DATE", "2026-06-14")
    monkeypatch.setenv("SAVANA_EVOLUTION_BATCH_SIZE", "2")

    # Two profiles chatted on 2026-06-14, one only on 2026-06-15.
    _write_profile(
        tmp_path,
        "savana_user_a_character",
        "角色A",
        3,
        [
            ("user", "yesterday a", 1781402400),      # 2026-06-14 10:00:00 +0800
            ("assistant", "reply a", 1781402460),
        ],
    )
    _write_profile(
        tmp_path,
        "savana_user_b_character",
        "角色B",
        4,
        [
            ("user", "yesterday b", 1781420400),      # 2026-06-14 15:00:00 +0800
            ("assistant", "reply b", 1781420460),
        ],
    )
    _write_profile(
        tmp_path,
        "savana_user_c_character",
        "角色C",
        5,
        [
            ("user", "today c", 1781492400),          # 2026-06-15 11:00:00 +0800
            ("assistant", "reply c", 1781492460),
        ],
    )

    module.main()

    output = capsys.readouterr().out
    manifest = _read_manifest(tmp_path, "2026-06-14")

    assert manifest["source_date"] == "2026-06-14"
    assert manifest["total_profiles"] == 2
    assert manifest["cursor"] == 0
    assert manifest["batches"] == [["savana_user_b_character", "savana_user_a_character"]]
    assert "- Review Date: 2026-06-14" in output
    assert "- Profiles Included In Batch: 2" in output
    assert "角色B" in output
    assert "角色A" in output
    assert "角色C" not in output


def test_manifest_splits_into_batches_and_advance_moves_cursor(monkeypatch, tmp_path):
    module = _load_module()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("SAVANA_EVOLUTION_TARGET_DATE", "2026-06-15")
    monkeypatch.setenv("SAVANA_EVOLUTION_SOURCE_DATE", "2026-06-14")
    monkeypatch.setenv("SAVANA_EVOLUTION_BATCH_SIZE", "2")

    for index in range(5):
        _write_profile(
            tmp_path,
            "savana_user_{0}_character".format(index),
            "角色{0}".format(index),
            2 + index,
            [
                ("user", "message {0}".format(index), 1781402400 + index * 60),
                ("assistant", "reply {0}".format(index), 1781402430 + index * 60),
            ],
        )

    module.main()

    manifest = _read_manifest(tmp_path, "2026-06-14")
    assert manifest["total_profiles"] == 5
    assert len(manifest["batches"]) == 3
    assert manifest["cursor"] == 0

    advanced = module.advance_manifest_cursor(str(tmp_path), "2026-06-14")
    assert advanced["cursor"] == 1
    advanced = module.advance_manifest_cursor(str(tmp_path), "2026-06-14")
    assert advanced["cursor"] == 2
    advanced = module.advance_manifest_cursor(str(tmp_path), "2026-06-14")
    assert advanced["cursor"] == 3
    assert module.manifest_complete(advanced) is True


def test_manifest_separates_legacy_and_guarded_profiles(tmp_path):
    module = _load_module()
    messages = [
        ("user", "yesterday", 1781402400),
        ("assistant", "reply", 1781402460),
    ]
    _write_profile(
        tmp_path,
        "savana_legacy_character",
        "旧角色",
        3,
        messages,
    )
    _write_profile(
        tmp_path,
        "savana_guarded_character",
        "新角色",
        3,
        messages,
        policy="guarded_v1",
    )

    manifest = module.build_manifest(str(tmp_path), "2026-06-14", 10, 30)

    assert manifest["batches"] == [
        ["savana_legacy_character"],
        ["savana_guarded_character"],
    ]
    assert manifest["batch_policies"] == ["legacy", "guarded_v1"]


def test_legacy_report_does_not_include_guarded_fields(monkeypatch, tmp_path, capsys):
    module = _load_module()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("SAVANA_EVOLUTION_TARGET_DATE", "2026-06-15")
    monkeypatch.setenv("SAVANA_EVOLUTION_SOURCE_DATE", "2026-06-14")
    _write_profile(
        tmp_path,
        "savana_legacy_character",
        "旧角色",
        3,
        [
            ("user", "yesterday", 1781402400),
            ("assistant", "reply", 1781402460),
        ],
    )

    module.main()
    output = capsys.readouterr().out

    assert "Evolution Batch Policy" not in output
    assert "SOUL.md SHA-256" not in output
    assert "Base Persona Snapshot" not in output


def test_guarded_report_contains_policy_base_and_hash(monkeypatch, tmp_path, capsys):
    module = _load_module()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("SAVANA_EVOLUTION_TARGET_DATE", "2026-06-15")
    monkeypatch.setenv("SAVANA_EVOLUTION_SOURCE_DATE", "2026-06-14")
    _write_profile(
        tmp_path,
        "savana_guarded_character",
        "新角色",
        3,
        [
            ("user", "yesterday", 1781402400),
            ("assistant", "reply", 1781402460),
        ],
        policy="guarded_v1",
    )

    module.main()
    output = capsys.readouterr().out

    assert "- Evolution Batch Policy: guarded_v1" in output
    assert "- SOUL.md SHA-256:" in output
    assert "### Base Persona Snapshot" in output
    base_snapshot = output.split("### Base Persona Snapshot", 1)[1].split(
        "### Dialogue History",
        1,
    )[0]
    assert "# 新角色" in base_snapshot
    assert "## Relationship with User" in base_snapshot
    assert "## Evolved Persona" not in base_snapshot
