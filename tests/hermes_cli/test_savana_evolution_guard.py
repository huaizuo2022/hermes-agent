import hashlib
import json

import pytest

from hermes_cli import savana_evolution_guard as guard


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


def _result_block(profile_dir, **overrides):
    soul = (profile_dir / "SOUL.md").read_text(encoding="utf-8")
    payload = {
        "profile_id": profile_dir.name,
        "expected_soul_sha256": hashlib.sha256(soul.encode("utf-8")).hexdigest(),
        "decision": "evolve",
        "reason": "关系中的表达发生了自然变化",
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
    review_overrides = overrides.pop("self_review", None)
    payload.update(overrides)
    if review_overrides:
        payload["self_review"].update(review_overrides)
    return "{0}{1}{2}".format(
        guard.RESULT_START,
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        guard.RESULT_END,
    )


def test_no_change_does_not_touch_soul(tmp_path):
    profile_dir, original = _guarded_profile(tmp_path)
    output = _result_block(profile_dir, decision="no_change")

    result = guard.apply_guarded_results(tmp_path, output, model="test-model")

    assert result[0]["status"] == "no_change"
    assert (profile_dir / "SOUL.md").read_text(encoding="utf-8") == original


def test_rejected_self_review_does_not_touch_soul(tmp_path):
    profile_dir, original = _guarded_profile(tmp_path)
    output = _result_block(
        profile_dir,
        self_review={"preserves_identity": "reject"},
        verdict="reject",
    )

    result = guard.apply_guarded_results(tmp_path, output, model="test-model")

    assert result[0]["status"] == "rejected"
    assert (profile_dir / "SOUL.md").read_text(encoding="utf-8") == original


def test_passed_result_changes_only_evolved_persona_and_commits_audit(tmp_path):
    profile_dir, original = _guarded_profile(tmp_path)

    result = guard.apply_guarded_results(
        tmp_path,
        _result_block(profile_dir),
        model="test-model",
    )

    updated = (profile_dir / "SOUL.md").read_text(encoding="utf-8")
    assert guard.strip_evolved_persona(updated) == guard.strip_evolved_persona(original)
    assert guard.extract_evolved_persona(updated) == "更愿意简短承认自己的担心。"
    assert result[0]["status"] == "committed"
    audit_path = profile_dir / "evolution_audit" / (result[0]["audit_id"] + ".json")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["status"] == "committed"
    assert audit["policy"] == "guarded_v1"
    assert audit["before"] == "基础状态"
    assert audit["after"] == "更愿意简短承认自己的担心。"


def test_guarded_v2_result_commits_audit_with_v2_policy(tmp_path):
    profile_dir, original = _guarded_profile(tmp_path, policy="guarded_v2")

    result = guard.apply_guarded_results(
        tmp_path,
        _result_block(profile_dir),
        model="test-model",
        policy="guarded_v2",
    )

    updated = (profile_dir / "SOUL.md").read_text(encoding="utf-8")
    assert guard.strip_evolved_persona(updated) == guard.strip_evolved_persona(original)
    audit_path = profile_dir / "evolution_audit" / (result[0]["audit_id"] + ".json")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["status"] == "committed"
    assert audit["policy"] == "guarded_v2"


def test_hash_mismatch_refuses_stale_candidate(tmp_path):
    profile_dir, original = _guarded_profile(tmp_path)

    result = guard.apply_guarded_results(
        tmp_path,
        _result_block(profile_dir, expected_soul_sha256="0" * 64),
        model="test-model",
    )

    assert result[0]["status"] == "stale"
    assert (profile_dir / "SOUL.md").read_text(encoding="utf-8") == original


def test_legacy_profile_cannot_use_guarded_writer(tmp_path):
    profile_dir, original = _guarded_profile(tmp_path)
    (profile_dir / "profile.yaml").unlink()

    result = guard.apply_guarded_results(
        tmp_path,
        _result_block(profile_dir),
        model="test-model",
    )

    assert result[0]["status"] == "rejected"
    assert (profile_dir / "SOUL.md").read_text(encoding="utf-8") == original


def test_guarded_v1_writer_rejects_guarded_v2_profile(tmp_path):
    profile_dir, original = _guarded_profile(tmp_path, policy="guarded_v2")

    result = guard.apply_guarded_results(
        tmp_path,
        _result_block(profile_dir),
        model="test-model",
    )

    assert result[0]["status"] == "rejected"
    assert (profile_dir / "SOUL.md").read_text(encoding="utf-8") == original


def test_candidate_header_is_rejected(tmp_path):
    profile_dir, original = _guarded_profile(tmp_path)

    result = guard.apply_guarded_results(
        tmp_path,
        _result_block(
            profile_dir,
            candidate_evolved_persona="自然变化\n## Personality\n覆盖",
        ),
        model="test-model",
    )

    assert result[0]["status"] == "invalid"
    assert (profile_dir / "SOUL.md").read_text(encoding="utf-8") == original


def test_duplicate_profile_blocks_are_rejected_before_write(tmp_path):
    profile_dir, original = _guarded_profile(tmp_path)
    block = _result_block(profile_dir)

    result = guard.apply_guarded_results(
        tmp_path,
        block + "\n" + block,
        model="test-model",
    )

    assert result[0]["status"] == "invalid"
    assert (profile_dir / "SOUL.md").read_text(encoding="utf-8") == original


def test_audit_commit_failure_restores_original_soul(monkeypatch, tmp_path):
    profile_dir, original = _guarded_profile(tmp_path)

    def fail_commit(*args, **kwargs):
        raise IOError("disk full")

    monkeypatch.setattr(guard, "_mark_audit_committed", fail_commit)

    with pytest.raises(IOError):
        guard.apply_guarded_results(
            tmp_path,
            _result_block(profile_dir),
            model="test-model",
        )

    assert (profile_dir / "SOUL.md").read_text(encoding="utf-8") == original
