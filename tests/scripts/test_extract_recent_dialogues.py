import importlib.util
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


def _write_profile(root, profile_id, character_name, intimacy, messages):
    profile_dir = root / "profiles" / profile_id
    profile_dir.mkdir(parents=True)
    (profile_dir / "SOUL.md").write_text(
        "# {0}\n\n## Evolved Persona\n基础状态\n\n## Relationship with User\n"
        "- Intimacy level: {1}/10\n".format(character_name, intimacy),
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


def test_recent_dialogue_report_includes_all_recent_profiles_without_top_five_cap(
    monkeypatch, tmp_path, capsys
):
    module = _load_module()
    now = 1000000
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("SAVANA_EVOLUTION_MAX_PROFILES", raising=False)
    monkeypatch.setattr(module.time, "time", lambda: now)

    for index in range(6):
        _write_profile(
            tmp_path,
            "savana_user_{0}_character".format(index),
            "角色{0}".format(index),
            0,
            [
                ("user", "hello {0}".format(index), now - 60 - index),
                ("assistant", "reply {0}".format(index), now - 30 - index),
            ],
        )

    module.main()

    output = capsys.readouterr().out
    assert "- Profiles Included: 6" in output
    assert output.count("## Character:") == 6
    assert "Current Intimacy Level: 0/10" in output


def test_recent_dialogue_report_uses_explicit_max_profiles_only_when_set(
    monkeypatch, tmp_path, capsys
):
    module = _load_module()
    now = 1000000
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("SAVANA_EVOLUTION_MAX_PROFILES", "2")
    monkeypatch.setattr(module.time, "time", lambda: now)

    for index in range(3):
        _write_profile(
            tmp_path,
            "savana_user_{0}_character".format(index),
            "角色{0}".format(index),
            5,
            [
                ("user", "hello {0}".format(index), now - 60 - index),
                ("assistant", "reply {0}".format(index), now - 30 - index),
            ],
        )

    module.main()

    output = capsys.readouterr().out
    assert "- Profiles Included: 2" in output
    assert output.count("## Character:") == 2
