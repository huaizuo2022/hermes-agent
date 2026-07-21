from pathlib import Path

from cron import scheduler


def _savana_job():
    return {
        "id": "job-1",
        "name": "Savana-Self-Evolution",
        "skill": "savana-companion-evolution",
        "schedule": {"kind": "cron", "expr": "0 3 * * *"},
    }


def test_legacy_batch_keeps_existing_skill():
    selected = scheduler._select_savana_evolution_skills(
        _savana_job(),
        None,
        "# Savana Characters Evolution Report\n- Review Date: 2026-06-14\n",
    )

    assert selected is None


def test_guarded_batch_selects_guarded_skill():
    selected = scheduler._select_savana_evolution_skills(
        _savana_job(),
        None,
        "- Evolution Batch Policy: guarded_v1\n",
    )

    assert selected == ["savana-companion-evolution-guarded"]


def test_guarded_v2_batch_selects_guarded_v2_skill():
    selected = scheduler._select_savana_evolution_skills(
        _savana_job(),
        None,
        "- Evolution Batch Policy: guarded_v2\n",
    )

    assert selected == ["savana-companion-evolution-guarded-v2"]


def test_guarded_output_is_applied_before_job_returns(monkeypatch, tmp_path):
    applied = []

    monkeypatch.setattr(
        "hermes_cli.savana_evolution_guard.apply_guarded_results",
        lambda home, output, model: applied.append((Path(home), output, model)) or [],
    )

    scheduler._apply_savana_evolution_output(
        _savana_job(),
        "- Evolution Batch Policy: guarded_v1\n",
        "model response",
        "test-model",
        tmp_path,
    )

    assert applied == [(tmp_path, "model response", "test-model")]


def test_guarded_v2_output_uses_guarded_writer(monkeypatch, tmp_path):
    applied = []

    monkeypatch.setattr(
        "hermes_cli.savana_evolution_guard.apply_guarded_results",
        lambda home, output, model, policy=None: applied.append(
            (Path(home), output, model, policy)
        ) or [],
    )

    scheduler._apply_savana_evolution_output(
        _savana_job(),
        "- Evolution Batch Policy: guarded_v2\n",
        "model response",
        "test-model",
        tmp_path,
    )

    assert applied == [(tmp_path, "model response", "test-model", "guarded_v2")]


def test_legacy_output_never_invokes_guarded_writer(monkeypatch, tmp_path):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("guarded writer must not run for legacy")

    monkeypatch.setattr(
        "hermes_cli.savana_evolution_guard.apply_guarded_results",
        fail_if_called,
    )

    result = scheduler._apply_savana_evolution_output(
        _savana_job(),
        "# legacy report\n",
        "model response",
        "test-model",
        tmp_path,
    )

    assert result == []


def test_missing_guarded_profile_result_is_recorded_invalid(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "hermes_cli.savana_evolution_guard.apply_guarded_results",
        lambda home, output, model: [],
    )
    prompt = (
        "- Evolution Batch Policy: guarded_v1\n"
        "## Character: 沈越 (Profile ID: savana_user_shenyue)\n"
    )

    result = scheduler._apply_savana_evolution_output(
        _savana_job(),
        prompt,
        "model response without marker",
        "test-model",
        tmp_path,
    )

    assert result == [{
        "profile_id": "savana_user_shenyue",
        "status": "invalid",
        "error": "missing structured result",
    }]


def test_missing_guarded_v2_profile_result_is_recorded_invalid(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "hermes_cli.savana_evolution_guard.apply_guarded_results",
        lambda home, output, model, policy=None: [],
    )
    prompt = (
        "- Evolution Batch Policy: guarded_v2\n"
        "## Character: 沈越 (Profile ID: savana_user_shenyue)\n"
    )

    result = scheduler._apply_savana_evolution_output(
        _savana_job(),
        prompt,
        "model response without marker",
        "test-model",
        tmp_path,
    )

    assert result == [{
        "profile_id": "savana_user_shenyue",
        "status": "invalid",
        "error": "missing structured result",
    }]
