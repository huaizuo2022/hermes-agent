from concurrent.futures import ThreadPoolExecutor

import pytest
import yaml

from hermes_cli import companion_profile_policy as policy


def test_existing_profile_without_policy_stays_legacy(tmp_path):
    profile_dir = tmp_path / "profiles" / "savana_u_c"
    profile_dir.mkdir(parents=True)

    assert policy.ensure_companion_profile(profile_dir) == policy.LEGACY_POLICY
    assert policy.read_evolution_policy(profile_dir) == policy.LEGACY_POLICY
    assert not (profile_dir / "profile.yaml").exists()


def test_new_profile_is_marked_guarded_before_use(tmp_path):
    profile_dir = tmp_path / "profiles" / "savana_u_c"

    assert policy.ensure_companion_profile(profile_dir) == policy.GUARDED_POLICY
    assert policy.read_evolution_policy(profile_dir) == policy.GUARDED_POLICY
    payload = yaml.safe_load((profile_dir / "profile.yaml").read_text(encoding="utf-8"))
    assert payload["evolution_policy"] == policy.GUARDED_POLICY


def test_policy_write_failure_removes_partial_new_profile(monkeypatch, tmp_path):
    profile_dir = tmp_path / "profiles" / "savana_u_c"

    def fail_write(*args, **kwargs):
        raise IOError("disk full")

    monkeypatch.setattr(policy, "_atomic_write_yaml", fail_write)

    with pytest.raises(IOError):
        policy.ensure_companion_profile(profile_dir)

    assert not profile_dir.exists()


def test_concurrent_initializers_observe_one_guarded_policy(tmp_path):
    profile_dir = tmp_path / "profiles" / "savana_u_c"

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(
            executor.map(
                lambda unused: policy.ensure_companion_profile(profile_dir),
                range(2),
            )
        )

    assert results == [policy.GUARDED_POLICY, policy.GUARDED_POLICY]
