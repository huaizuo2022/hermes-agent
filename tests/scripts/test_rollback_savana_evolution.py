import hashlib
import json

import pytest

from hermes_cli import savana_evolution_guard as guard
from scripts import rollback_savana_evolution as rollback_script


def _guarded_profile(tmp_path, policy="guarded_v1"):
    profile_dir = tmp_path / "profiles" / "savana_u_c"
    profile_dir.mkdir(parents=True)
    (profile_dir / "profile.yaml").write_text(
        "evolution_policy: {0}\n".format(policy),
        encoding="utf-8",
    )
    original = (
        "# 沈越\n\n## Personality\n克制、敏锐\n\n"
        "## Evolved Persona\n基础状态\n\n## Speaking Style\n简短自然\n"
    )
    (profile_dir / "SOUL.md").write_text(original, encoding="utf-8")
    return profile_dir, original


def _passed_result(profile_dir):
    soul = (profile_dir / "SOUL.md").read_text(encoding="utf-8")
    return {
        "profile_id": profile_dir.name,
        "expected_soul_sha256": hashlib.sha256(soul.encode("utf-8")).hexdigest(),
        "decision": "evolve",
        "reason": "自然形成了更坦率的表达",
        "candidate_evolved_persona": "更愿意简短承认自己的担心。",
        "self_review": {
            "necessary": "pass",
            "preserves_identity": "pass",
            "no_unfounded_jump": "pass",
            "no_error_solidification": "pass",
            "no_base_override": "pass",
        },
        "verdict": "pass",
    }


def test_rollback_restores_only_evolved_persona_and_audits_action(tmp_path):
    profile_dir, original = _guarded_profile(tmp_path)
    committed = guard.apply_guarded_result(
        tmp_path,
        _passed_result(profile_dir),
        "test-model",
    )
    changed = (profile_dir / "SOUL.md").read_text(encoding="utf-8")

    rollback = guard.rollback_guarded_evolution(
        tmp_path,
        profile_dir.name,
        committed["audit_id"],
    )

    restored = (profile_dir / "SOUL.md").read_text(encoding="utf-8")
    assert guard.extract_evolved_persona(restored) == guard.extract_evolved_persona(original)
    assert guard.strip_evolved_persona(restored) == guard.strip_evolved_persona(changed)
    rollback_audit = json.loads(
        (profile_dir / "evolution_audit" / (rollback["audit_id"] + ".json")).read_text(
            encoding="utf-8"
        )
    )
    assert rollback_audit["status"] == "committed"
    assert rollback_audit["action"] == "rollback"
    assert rollback_audit["source_audit_id"] == committed["audit_id"]


def test_rollback_refuses_when_current_persona_no_longer_matches_audit(tmp_path):
    profile_dir, unused_original = _guarded_profile(tmp_path)
    committed = guard.apply_guarded_result(
        tmp_path,
        _passed_result(profile_dir),
        "test-model",
    )
    current = (profile_dir / "SOUL.md").read_text(encoding="utf-8")
    (profile_dir / "SOUL.md").write_text(
        guard.replace_evolved_persona(current, "另一次进化"),
        encoding="utf-8",
    )

    with pytest.raises(guard.StaleEvolutionError):
        guard.rollback_guarded_evolution(
            tmp_path,
            profile_dir.name,
            committed["audit_id"],
        )


def test_rollback_cli_prints_new_audit_id(tmp_path, capsys):
    profile_dir, unused_original = _guarded_profile(tmp_path)
    committed = guard.apply_guarded_result(
        tmp_path,
        _passed_result(profile_dir),
        "test-model",
    )

    exit_code = rollback_script.main(
        [
            "--hermes-home",
            str(tmp_path),
            "--profile-id",
            profile_dir.name,
            "--audit-id",
            committed["audit_id"],
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "committed"
    assert payload["audit_id"] != committed["audit_id"]


def test_guarded_v2_rollback_restores_persona_and_preserves_v2_policy_in_audit(tmp_path):
    profile_dir, original = _guarded_profile(tmp_path, policy="guarded_v2")
    committed = guard.apply_guarded_result(
        tmp_path,
        _passed_result(profile_dir),
        "test-model",
        policy="guarded_v2",
    )

    rollback = guard.rollback_guarded_evolution(
        tmp_path,
        profile_dir.name,
        committed["audit_id"],
    )

    restored = (profile_dir / "SOUL.md").read_text(encoding="utf-8")
    assert guard.extract_evolved_persona(restored) == guard.extract_evolved_persona(original)
    rollback_audit = json.loads(
        (profile_dir / "evolution_audit" / (rollback["audit_id"] + ".json")).read_text(
            encoding="utf-8"
        )
    )
    assert rollback_audit["status"] == "committed"
    assert rollback_audit["action"] == "rollback"
    assert rollback_audit["policy"] == "guarded_v2"


def test_rollback_rejects_legacy_profile(tmp_path):
    profile_dir, original = _guarded_profile(tmp_path)
    (profile_dir / "profile.yaml").unlink()

    with pytest.raises(guard.EvolutionPolicyError):
        guard.rollback_guarded_evolution(
            tmp_path,
            profile_dir.name,
            "20260721T000000000000Z-demo",
        )

    assert (profile_dir / "SOUL.md").read_text(encoding="utf-8") == original
