import json
from unittest.mock import patch


def test_savana_batch_advance_reschedules_until_manifest_complete(monkeypatch, tmp_path):
    from cron import scheduler

    manifest_dir = tmp_path / "cron" / "state" / "savana-self-evolution"
    manifest_dir.mkdir(parents=True)
    manifest_path = manifest_dir / "2026-06-14.json"
    manifest_path.write_text(
        json.dumps(
            {
                "source_date": "2026-06-14",
                "cursor": 0,
                "batches": [["a", "b"], ["c"]],
            }
        ),
        encoding="utf-8",
    )

    job = {
        "id": "job-1",
        "name": "Savana-Self-Evolution",
        "skill": "savana-companion-evolution",
        "schedule": {"kind": "cron", "expr": "0 3 * * *", "display": "0 3 * * *"},
    }

    updates = []

    def fake_update_job(job_id, payload):
        updates.append((job_id, payload))
        return {"id": job_id, **payload}

    monkeypatch.setattr(scheduler, "_hermes_home", tmp_path)
    monkeypatch.setattr(
        "cron.jobs.update_job",
        fake_update_job,
    )

    class FakeExtractModule(object):
        @staticmethod
        def advance_manifest_cursor(hermes_home, source_date):
            assert hermes_home == str(tmp_path)
            assert source_date == "2026-06-14"
            return {
                "source_date": "2026-06-14",
                "cursor": 1,
                "batches": [["a", "b"], ["c"]],
            }

        @staticmethod
        def manifest_complete(manifest):
            return manifest["cursor"] >= len(manifest["batches"])

    with patch("scripts.extract_recent_dialogues", FakeExtractModule):
        scheduler._advance_savana_evolution_batch(
            job,
            "# Savana Characters Evolution Report\n- Review Date: 2026-06-14\n",
        )

    assert len(updates) == 1
    assert updates[0][0] == "job-1"
    assert "next_run_at" in updates[0][1]


def test_savana_batch_advance_restores_daily_schedule_when_done(monkeypatch, tmp_path):
    from cron import scheduler

    job = {
        "id": "job-1",
        "name": "Savana-Self-Evolution",
        "skill": "savana-companion-evolution",
        "schedule": {"kind": "cron", "expr": "0 3 * * *", "display": "0 3 * * *"},
    }

    updates = []

    def fake_update_job(job_id, payload):
        updates.append((job_id, payload))
        return {"id": job_id, **payload}

    monkeypatch.setattr(scheduler, "_hermes_home", tmp_path)
    monkeypatch.setattr("cron.jobs.update_job", fake_update_job)
    monkeypatch.setattr("cron.jobs.compute_next_run", lambda schedule: "2026-06-16T03:00:00+08:00")

    class FakeExtractModule(object):
        @staticmethod
        def advance_manifest_cursor(hermes_home, source_date):
            return {
                "source_date": "2026-06-14",
                "cursor": 2,
                "batches": [["a"], ["b"]],
            }

        @staticmethod
        def manifest_complete(manifest):
            return True

    with patch("scripts.extract_recent_dialogues", FakeExtractModule):
        scheduler._advance_savana_evolution_batch(
            job,
            "# Savana Characters Evolution Report\n- Review Date: 2026-06-14\n",
        )

    assert updates == [("job-1", {"next_run_at": "2026-06-16T03:00:00+08:00"})]
