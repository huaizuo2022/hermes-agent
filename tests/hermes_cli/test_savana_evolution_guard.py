import hashlib
import json
import os
from pathlib import Path

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


def _result_payload(profile_dir, **overrides):
    block = _result_block(profile_dir, **overrides)
    return json.loads(block[len(guard.RESULT_START):-len(guard.RESULT_END)])


def _second_guarded_profile(tmp_path):
    profile_dir = tmp_path / "profiles" / "savana_u_d"
    profile_dir.mkdir(parents=True)
    (profile_dir / "profile.yaml").write_text(
        "evolution_policy: guarded_v1\n",
        encoding="utf-8",
    )
    original = (
        "# 林岚\n\n## Personality\n安静、敏锐\n\n"
        "## Evolved Persona\n基础状态\n\n## Speaking Style\n轻声慢语\n"
    )
    (profile_dir / "SOUL.md").write_text(original, encoding="utf-8")
    return profile_dir, original


def _recovery_journals(tmp_path):
    return sorted((tmp_path / "savana_evolution_recovery").glob("*.json"))


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
    assert _recovery_journals(tmp_path) == []


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


def test_unexpected_profile_result_is_rejected_before_any_write(tmp_path):
    profile_dir, original = _guarded_profile(tmp_path)

    result = guard.apply_guarded_results(
        tmp_path,
        _result_block(profile_dir),
        model="test-model",
        expected_profile_ids=["savana_someone_else"],
    )

    assert result == [{
        "profile_id": profile_dir.name,
        "status": "invalid",
        "error": "unexpected profile result",
    }, {
        "profile_id": "savana_someone_else",
        "status": "invalid",
        "error": "missing structured result",
    }]
    assert (profile_dir / "SOUL.md").read_text(encoding="utf-8") == original


def test_mixed_committable_and_invalid_results_do_not_partially_write(tmp_path):
    profile_dir, original = _guarded_profile(tmp_path)
    valid_block = _result_block(profile_dir)
    invalid_block = _result_block(profile_dir, candidate_evolved_persona="自然变化\n## Personality\n覆盖")

    result = guard.apply_guarded_results(
        tmp_path,
        valid_block + "\n" + invalid_block,
        model="test-model",
        expected_profile_ids=[profile_dir.name],
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
    assert list((profile_dir / "evolution_audit").glob("*.json")) == []
    assert _recovery_journals(tmp_path) == []


def test_single_audit_commit_failure_cleans_pending_audit_and_journal(monkeypatch, tmp_path):
    profile_dir, original = _guarded_profile(tmp_path)

    def fail_commit(*args, **kwargs):
        raise IOError("single audit commit failed")

    monkeypatch.setattr(guard, "_mark_audit_committed", fail_commit)

    with pytest.raises(IOError, match="single audit commit failed"):
        guard.apply_guarded_result(
            tmp_path,
            _result_payload(profile_dir),
            model="test-model",
        )

    assert (profile_dir / "SOUL.md").read_text(encoding="utf-8") == original
    assert list((profile_dir / "evolution_audit").glob("*.json")) == []
    assert _recovery_journals(tmp_path) == []


def test_single_double_failure_keeps_recovery_evidence_and_blocks_later_writes(
    monkeypatch,
    tmp_path,
):
    profile_dir, original = _guarded_profile(tmp_path)
    soul_path = profile_dir / "SOUL.md"
    original_atomic_write = guard._atomic_write_text

    def fail_commit(*args, **kwargs):
        raise IOError("single audit commit failed")

    def fail_restore(path, content):
        if Path(path) == soul_path and content == original:
            raise IOError("single SOUL restore failed")
        return original_atomic_write(path, content)

    monkeypatch.setattr(guard, "_mark_audit_committed", fail_commit)
    monkeypatch.setattr(guard, "_atomic_write_text", fail_restore)

    with pytest.raises(guard.BatchRecoveryError) as exc_info:
        guard.apply_guarded_result(
            tmp_path,
            _result_payload(profile_dir),
            model="test-model",
        )

    assert "single audit commit failed" in str(exc_info.value)
    assert "single SOUL restore failed" in str(exc_info.value)
    assert soul_path.read_text(encoding="utf-8") != original
    journals = _recovery_journals(tmp_path)
    assert len(journals) == 1
    journal = json.loads(journals[0].read_text(encoding="utf-8"))
    assert journal["status"] == "recovery_required"
    entry = journal["entries"][0]
    assert entry["soul_state"] == "rollback_failed"
    assert entry["audit_state"] == "retained_for_recovery"
    audit_path = Path(entry["audit_path"])
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["status"] == "pending"
    assert audit["batch_id"] == journal["id"]

    with pytest.raises(guard.BatchRecoveryError, match=str(journals[0])):
        guard.apply_guarded_results(
            tmp_path,
            _result_block(profile_dir),
            model="test-model",
        )


def test_batch_second_profile_write_failure_restores_first_profile(monkeypatch, tmp_path):
    first_dir, first_original = _guarded_profile(tmp_path)
    second_dir, second_original = _second_guarded_profile(tmp_path)

    blocks = []
    for profile_dir in (first_dir, second_dir):
        blocks.append(_result_block(profile_dir))

    original_atomic_write = guard._atomic_write_text
    failure_target = str(second_dir / "SOUL.md")

    def fail_second_write(path, content):
        if str(path) == failure_target and "更愿意简短承认自己的担心。" in content:
            raise IOError("disk full on second profile")
        return original_atomic_write(path, content)

    monkeypatch.setattr(guard, "_atomic_write_text", fail_second_write)

    with pytest.raises(IOError):
        guard.apply_guarded_results(
            tmp_path,
            "\n".join(blocks),
            model="test-model",
            expected_profile_ids=sorted([first_dir.name, second_dir.name]),
        )

    assert (first_dir / "SOUL.md").read_text(encoding="utf-8") == first_original
    assert (second_dir / "SOUL.md").read_text(encoding="utf-8") == second_original
    assert list((first_dir / "evolution_audit").glob("*.json")) == []
    assert list((second_dir / "evolution_audit").glob("*.json")) == []
    assert _recovery_journals(tmp_path) == []


def test_batch_double_failure_keeps_durable_recovery_evidence_and_blocks_next_batch(
    monkeypatch,
    tmp_path,
):
    first_dir, first_original = _guarded_profile(tmp_path)
    second_dir, second_original = _second_guarded_profile(tmp_path)
    output = "\n".join([_result_block(first_dir), _result_block(second_dir)])
    original_atomic_write = guard._atomic_write_text
    first_soul = first_dir / "SOUL.md"
    second_soul = second_dir / "SOUL.md"

    def fail_forward_and_rollback(path, content):
        path = Path(path)
        if path == second_soul and content != second_original:
            raise IOError("forward write failed")
        if path == first_soul and content == first_original:
            raise IOError("rollback write failed")
        return original_atomic_write(path, content)

    monkeypatch.setattr(guard, "_atomic_write_text", fail_forward_and_rollback)

    with pytest.raises(Exception) as exc_info:
        guard.apply_guarded_results(
            tmp_path,
            output,
            model="test-model",
            expected_profile_ids=[first_dir.name, second_dir.name],
        )

    assert exc_info.type.__name__ == "BatchRecoveryError"
    assert "forward write failed" in str(exc_info.value)
    assert "rollback write failed" in str(exc_info.value)
    assert (first_dir / "SOUL.md").read_text(encoding="utf-8") != first_original
    assert (second_dir / "SOUL.md").read_text(encoding="utf-8") == second_original
    journals = _recovery_journals(tmp_path)
    assert len(journals) == 1
    journal = json.loads(journals[0].read_text(encoding="utf-8"))
    assert journal["status"] == "recovery_required"
    first_entry = next(item for item in journal["entries"] if item["profile_id"] == first_dir.name)
    assert first_entry["original_sha256"] == hashlib.sha256(first_original.encode("utf-8")).hexdigest()
    assert first_entry["candidate_sha256"] == hashlib.sha256(
        (first_dir / "SOUL.md").read_text(encoding="utf-8").encode("utf-8")
    ).hexdigest()
    assert first_entry["soul_state"] == "rollback_failed"
    retained_audit = Path(first_entry["audit_path"])
    assert retained_audit.exists()
    audit = json.loads(retained_audit.read_text(encoding="utf-8"))
    assert audit["status"] == "committed"
    assert audit["batch_id"] == journal["id"]
    assert audit["after"] == "更愿意简短承认自己的担心。"

    with pytest.raises(Exception) as blocked_info:
        guard.apply_guarded_results(
            tmp_path,
            output,
            model="test-model",
            expected_profile_ids=[first_dir.name, second_dir.name],
        )
    assert blocked_info.type.__name__ == "BatchRecoveryError"
    assert str(journals[0]) in str(blocked_info.value)


def test_batch_audit_cleanup_failure_is_exposed_and_journaled(monkeypatch, tmp_path):
    first_dir, first_original = _guarded_profile(tmp_path)
    second_dir, second_original = _second_guarded_profile(tmp_path)
    output = "\n".join([_result_block(first_dir), _result_block(second_dir)])
    original_atomic_write = guard._atomic_write_text
    original_unlink = Path.unlink

    def fail_second_write(path, content):
        if Path(path) == second_dir / "SOUL.md" and content != second_original:
            raise IOError("forward write failed")
        return original_atomic_write(path, content)

    def fail_audit_unlink(path, *args, **kwargs):
        if path.parent.name == "evolution_audit" and path.suffix == ".json":
            raise IOError("audit unlink failed")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(guard, "_atomic_write_text", fail_second_write)
    monkeypatch.setattr(Path, "unlink", fail_audit_unlink)

    with pytest.raises(Exception) as exc_info:
        guard.apply_guarded_results(
            tmp_path,
            output,
            model="test-model",
            expected_profile_ids=[first_dir.name, second_dir.name],
        )

    assert exc_info.type.__name__ == "BatchRecoveryError"
    assert "audit unlink failed" in str(exc_info.value)
    assert (first_dir / "SOUL.md").read_text(encoding="utf-8") == first_original
    assert (second_dir / "SOUL.md").read_text(encoding="utf-8") == second_original
    journals = _recovery_journals(tmp_path)
    assert len(journals) == 1
    journal = json.loads(journals[0].read_text(encoding="utf-8"))
    assert journal["status"] == "recovery_required"
    assert any(item["audit_state"] == "cleanup_failed" for item in journal["entries"])
    assert any(Path(item["audit_path"]).exists() for item in journal["entries"])


def test_single_result_writer_blocks_unresolved_batch_journal(tmp_path):
    profile_dir, original = _guarded_profile(tmp_path)
    recovery_dir = tmp_path / "savana_evolution_recovery"
    recovery_dir.mkdir()
    journal_path = recovery_dir / "unresolved.json"
    journal_path.write_text(
        json.dumps({"status": "recovery_required", "entries": []}),
        encoding="utf-8",
    )
    with pytest.raises(guard.BatchRecoveryError, match=str(journal_path)):
        guard.apply_guarded_result(
            tmp_path,
            _result_payload(profile_dir),
            model="test-model",
        )

    assert (profile_dir / "SOUL.md").read_text(encoding="utf-8") == original
    assert not (profile_dir / "evolution_audit").exists()


def test_explicit_empty_expected_profile_ids_fail_closed_before_write(monkeypatch, tmp_path):
    profile_dir, original = _guarded_profile(tmp_path)
    monkeypatch.setattr(
        guard,
        "profile_lock",
        lambda *args, **kwargs: pytest.fail("empty expected set must not acquire locks"),
    )

    result = guard.apply_guarded_results(
        tmp_path,
        _result_block(profile_dir),
        model="test-model",
        expected_profile_ids=[],
    )

    assert result == [{
        "profile_id": "",
        "status": "invalid",
        "error": "expected_profile_ids must not be empty",
    }]
    assert (profile_dir / "SOUL.md").read_text(encoding="utf-8") == original


def test_duplicate_expected_profile_ids_fail_closed_before_lock(monkeypatch, tmp_path):
    profile_dir, original = _guarded_profile(tmp_path)
    monkeypatch.setattr(
        guard,
        "profile_lock",
        lambda *args, **kwargs: pytest.fail("duplicate expected ids must not acquire locks"),
    )

    result = guard.apply_guarded_results(
        tmp_path,
        _result_block(profile_dir),
        model="test-model",
        expected_profile_ids=[profile_dir.name, profile_dir.name],
    )

    assert result == [{
        "profile_id": profile_dir.name,
        "status": "invalid",
        "error": "duplicate expected profile id",
    }]
    assert (profile_dir / "SOUL.md").read_text(encoding="utf-8") == original


def test_profile_symlink_aliases_are_rejected_before_lock(monkeypatch, tmp_path):
    profile_dir, original = _guarded_profile(tmp_path)
    alias_dir = tmp_path / "profiles" / "savana_alias"
    alias_dir.symlink_to(profile_dir.name, target_is_directory=True)
    output = "\n".join([_result_block(profile_dir), _result_block(alias_dir)])
    monkeypatch.setattr(
        guard,
        "profile_lock",
        lambda *args, **kwargs: pytest.fail("canonical aliases must not acquire locks"),
    )

    result = guard.apply_guarded_results(
        tmp_path,
        output,
        model="test-model",
        expected_profile_ids=[profile_dir.name, alias_dir.name],
    )

    assert result == [{
        "profile_id": alias_dir.name,
        "status": "invalid",
        "error": "profile ids resolve to the same canonical directory",
    }]
    assert (profile_dir / "SOUL.md").read_text(encoding="utf-8") == original
    assert not (profile_dir / "evolution_audit").exists()


def test_batch_lock_held_policy_change_fails_before_any_write(monkeypatch, tmp_path):
    profile_dir, original = _guarded_profile(tmp_path)
    original_read_policy = guard.read_evolution_policy
    calls = {"count": 0}

    def flip_policy(profile_path):
        calls["count"] += 1
        if calls["count"] >= 2:
            return "guarded_v2"
        return original_read_policy(profile_path)

    monkeypatch.setattr(guard, "read_evolution_policy", flip_policy)

    result = guard.apply_guarded_results(
        tmp_path,
        _result_block(profile_dir),
        model="test-model",
        expected_profile_ids=[profile_dir.name],
    )

    assert result[0]["status"] == "rejected"
    assert (profile_dir / "SOUL.md").read_text(encoding="utf-8") == original
