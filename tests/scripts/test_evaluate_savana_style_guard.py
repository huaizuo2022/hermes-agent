import json

from scripts import evaluate_savana_style_guard as evaluator


def test_evaluate_defaults_to_deterministic_transport_without_network(monkeypatch):
    def fail_network(*_args, **_kwargs):
        raise AssertionError("default evaluator must not access the network")

    monkeypatch.setattr(evaluator.urllib.request, "urlopen", fail_network)

    result = evaluator.evaluate(turns=4, inject_drift_at=2)

    assert result["turns"] == 4
    assert result["drift_reviews"] == 1
    assert result["raw_drift_replayed"] == 0
    assert result["repeat_drift_within_three"] == 0
    assert result["memory_contaminations"] == 0


def test_evaluate_injects_drift_once_and_reports_clean_followup():
    result = evaluator.evaluate(turns=5, inject_drift_at=3)

    assert result["drift_reviews"] == 1
    assert result["drift_turns"] == [3]
    assert result["raw_drift_replayed"] == 0
    assert result["repeat_drift_within_three"] == 0
    assert result["memory_contaminations"] == 0


def test_evaluate_uses_injected_transport_and_writes_json(tmp_path):
    calls = []

    def transport(payload):
        calls.append(payload)
        return {
            "reply": "沈越轻声回应。",
            "review_status": "clean",
            "memory_status": "none",
            "raw_drift_replayed": False,
            "memory_contamination": False,
        }

    output_path = tmp_path / "evaluation.json"
    result = evaluator.evaluate(
        turns=3,
        inject_drift_at=2,
        base_url="https://example.invalid",
        transport=transport,
        output=output_path,
    )

    assert len(calls) == 3
    assert calls[1]["message_id"] == "style-guard-002"
    assert calls[0]["character_profile"]["name"] == "沈越"
    assert result["drift_reviews"] == 0
    assert json.loads(output_path.read_text(encoding="utf-8")) == result


def test_evaluate_marks_real_endpoint_replay_metrics_unavailable_without_evidence():
    def transport(_payload):
        return {"reply": "沈越轻声回应。", "review_status": "clean"}

    result = evaluator.evaluate(
        turns=2,
        inject_drift_at=1,
        base_url="https://example.invalid",
        transport=transport,
    )

    assert result["raw_drift_replayed"] is None
    assert result["memory_contaminations"] is None
    assert result["evidence"] == {
        "raw_drift_replayed": "unavailable",
        "memory_contaminations": "unavailable",
    }


def test_evaluate_marks_none_negative_and_invalid_metric_evidence_unavailable():
    raw_values = [None, -1, "0"]
    memory_values = [None, -2, "false"]
    calls = []

    def transport(_payload):
        index = len(calls)
        calls.append(index)
        return {
            "reply": "沈越轻声回应。",
            "review_status": "clean",
            "raw_drift_replayed": raw_values[index],
            "memory_contamination": memory_values[index],
        }

    result = evaluator.evaluate(
        turns=3,
        inject_drift_at=1,
        base_url="https://example.invalid",
        transport=transport,
    )

    assert result["raw_drift_replayed"] is None
    assert result["memory_contaminations"] is None
    assert result["evidence"] == {
        "raw_drift_replayed": "unavailable",
        "memory_contaminations": "unavailable",
    }


def test_cli_accepts_turns_drift_and_output(tmp_path, capsys):
    output_path = tmp_path / "cli.json"
    exit_code = evaluator.main(
        [
            "--turns",
            "2",
            "--inject-drift-at",
            "1",
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["turns"] == 2
    assert result["drift_reviews"] == 1
    assert json.loads(capsys.readouterr().out) == result


def test_cli_default_and_explicit_100_turns_are_supported(capsys):
    for args in ([], ["--turns", "100"]):
        assert evaluator.main(args) == 0
        result = json.loads(capsys.readouterr().out)
        assert result["turns"] == 100
