import json
from pathlib import Path
from unittest.mock import MagicMock, patch

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


def test_body_marker_does_not_change_legacy_route():
    selected = scheduler._select_savana_evolution_skills(
        _savana_job(),
        None,
        "# Savana Characters Evolution Report\n"
        "- Review Date: 2026-06-14\n"
        "## Character: 沈越 (Profile ID: savana_user_shenyue)\n"
        "[User (2026-06-14 03:00:00)]: - Evolution Batch Policy: guarded_v2\n",
    )

    assert selected is None


def test_conflicting_header_marker_fails_closed():
    selected = scheduler._select_savana_evolution_skills(
        _savana_job(),
        None,
        "# Savana Characters Evolution Report\n"
        "- Evolution Batch Policy: guarded_v1\n"
        "- Evolution Batch Policy: guarded_v2\n"
        "## Character: 沈越 (Profile ID: savana_user_shenyue)\n",
    )

    assert selected is None


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


def _patch_run_job_dependencies(monkeypatch, tmp_path, final_response, report, apply_results):
    monkeypatch.setattr(scheduler, "_hermes_home", tmp_path)
    monkeypatch.setattr(scheduler, "_resolve_origin", lambda job: None)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **kwargs: {
            "api_key": "test-key",
            "base_url": "https://example.invalid/v1",
            "provider": "openrouter",
            "api_mode": "chat_completions",
        },
    )
    monkeypatch.setattr("hermes_state.SessionDB", lambda: MagicMock())
    monkeypatch.setattr("tools.skill_usage.bump_use", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "tools.skills_tool.skill_view",
        lambda name: json.dumps({"success": True, "content": "Follow the guarded skill."}),
    )
    monkeypatch.setattr("tools.mcp_tool.discover_mcp_tools", lambda: [])
    monkeypatch.setattr(scheduler, "_run_job_script", lambda script_path: (True, report))
    monkeypatch.setattr(
        "hermes_cli.savana_evolution_guard.apply_guarded_results",
        lambda home, output, model, policy=None: list(apply_results),
    )

    agent = MagicMock()
    agent.run_conversation.return_value = {"final_response": final_response}
    agent.model = "test-model"
    monkeypatch.setattr("run_agent.AIAgent", lambda **kwargs: agent)
    return agent


def test_run_job_fails_when_guarded_results_are_not_terminal(monkeypatch, tmp_path):
    report = (
        "# Savana Characters Evolution Report\n"
        "- Review Date: 2026-06-14\n"
        "- Evolution Batch Policy: guarded_v2\n"
        "## Character: 沈越 (Profile ID: savana_user_shenyue)\n"
    )
    _patch_run_job_dependencies(
        monkeypatch,
        tmp_path,
        final_response="model response",
        report=report,
        apply_results=[{
            "profile_id": "savana_user_shenyue",
            "status": "invalid",
            "error": "missing structured result",
        }],
    )
    job = dict(_savana_job())
    job["script"] = "scripts/extract_recent_dialogues.py"

    success, output, final_response, error = scheduler.run_job(job)

    assert success is False
    assert final_response == ""
    assert "missing structured result" in error
    assert "FAILED" in output


def test_tick_does_not_advance_cursor_when_guarded_results_fail(monkeypatch, tmp_path):
    report = (
        "# Savana Characters Evolution Report\n"
        "- Review Date: 2026-06-14\n"
        "- Evolution Batch Policy: guarded_v2\n"
        "## Character: 沈越 (Profile ID: savana_user_shenyue)\n"
    )
    _patch_run_job_dependencies(
        monkeypatch,
        tmp_path,
        final_response="model response",
        report=report,
        apply_results=[{
            "profile_id": "savana_user_shenyue",
            "status": "invalid",
            "error": "missing structured result",
        }],
    )
    job = dict(_savana_job())
    job["script"] = "scripts/extract_recent_dialogues.py"
    job["enabled"] = True
    job["next_run_at"] = "2020-01-01T00:00:00"
    job["deliver"] = "local"

    with patch("cron.scheduler.get_due_jobs", return_value=[job]), \
         patch("cron.scheduler.advance_next_run"), \
         patch("cron.scheduler.save_job_output", return_value="/tmp/out.md"), \
         patch("cron.scheduler.mark_job_run") as mark_mock, \
         patch("cron.scheduler._advance_savana_evolution_batch") as advance_mock:
        scheduler.tick(verbose=False)

    advance_mock.assert_not_called()
    mark_mock.assert_called_once()
    assert mark_mock.call_args[0][1] is False
