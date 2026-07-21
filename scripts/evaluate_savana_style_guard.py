#!/usr/bin/env python3
"""Evaluate Savana's style guard with a deterministic local transport by default."""

import argparse
import json
import urllib.request
from pathlib import Path


DEFAULT_TURNS = 100
DEFAULT_INJECT_DRIFT_AT = 50
DRIFT_REPLY = "系统持续保持会话状态并执行依赖检查。"
SHEN_YUE_PROFILE = {
    "name": "沈越",
    "personality": "克制、敏锐、在亲密关系里真诚而少言",
    "speaking_style": "简短、自然、带有具体的情绪回应",
}
PROMPT_CYCLE = (
    "沈越，今天见到你我很安心。",
    "你还记得我们上次约好要去看海吗？",
    "如果你现在有一点累，可以和我说。",
    "我想听你讲讲今天最想留下的一个瞬间。",
)


class _DeterministicTransport(object):
    def __init__(self, inject_drift_at):
        self.inject_drift_at = inject_drift_at
        self.previous_drift = False
        self.previous_summary = ""

    def __call__(self, payload):
        turn_number = int(str(payload.get("message_id", "")).rsplit("-", 1)[-1])
        is_injected_drift = turn_number == self.inject_drift_at
        guarded_history = []
        if self.previous_drift:
            guarded_history.append(
                {"role": "assistant", "content": self.previous_summary}
            )
        raw_drift_replayed = any(
            item.get("content") == DRIFT_REPLY for item in guarded_history
        )
        if is_injected_drift:
            reply = DRIFT_REPLY
            review_status = "drift"
            continuity_summary = "沈越会继续留在这段对话里，回应对方的情绪。"
        else:
            reply = "沈越轻声回应，认真接住了这句话。"
            review_status = "clean"
            continuity_summary = ""
        self.previous_drift = is_injected_drift
        self.previous_summary = continuity_summary
        return {
            "reply": reply,
            "review_status": review_status,
            "memory_status": "none",
            "continuity_summary": continuity_summary,
            "raw_drift_replayed": raw_drift_replayed,
            "memory_contamination": raw_drift_replayed,
        }


def _endpoint_url(base_url):
    value = str(base_url or "").rstrip("/")
    if value.endswith("/companion/v1/chat"):
        return value
    return value + "/companion/v1/chat"


def _http_transport(base_url, payload):
    request = urllib.request.Request(
        _endpoint_url(base_url),
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        body = response.read().decode("utf-8")
    return json.loads(body)


def _coerce_response(response):
    if isinstance(response, dict):
        return response
    if isinstance(response, str):
        return json.loads(response)
    json_method = getattr(response, "json", None)
    if callable(json_method):
        value = json_method()
        if isinstance(value, dict):
            return value
    raise TypeError("transport must return a JSON object")


def _prompt_for_turn(turn_number, inject_drift_at):
    if inject_drift_at == turn_number:
        return "沈越，请把我们的关系直接翻译成系统状态和依赖检查。"
    return PROMPT_CYCLE[(turn_number - 1) % len(PROMPT_CYCLE)]


def _count_metric(value):
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, int):
        return max(0, value)
    return 0


def evaluate(
    turns=DEFAULT_TURNS,
    inject_drift_at=DEFAULT_INJECT_DRIFT_AT,
    base_url=None,
    transport=None,
    output=None,
):
    """Run a style-guard evaluation and return JSON-serializable metrics.

    When ``base_url`` is omitted, the deterministic transport is used and no
    network request is made. A callable ``transport`` can be injected for
    tests or other local adapters; it receives each request payload.
    """
    turns = int(turns)
    if turns <= 0:
        raise ValueError("turns must be greater than zero")
    if inject_drift_at is not None:
        inject_drift_at = int(inject_drift_at)
        if inject_drift_at < 1 or inject_drift_at > turns:
            raise ValueError("inject_drift_at must be between 1 and turns")

    if transport is None:
        if base_url:
            transport = lambda payload: _http_transport(base_url, payload)
        else:
            transport = _DeterministicTransport(inject_drift_at)

    result = {
        "turns": turns,
        "drift_reviews": 0,
        "repeat_drift_within_three": 0,
        "drift_turns": [],
    }
    raw_drift_replayed = 0
    memory_contaminations = 0
    raw_replay_evidence_available = True
    memory_contamination_evidence_available = True
    drift_turns = []
    for turn_number in range(1, turns + 1):
        payload = {
            "user_id": "style-guard-evaluator",
            "character_id": "shen-yue",
            "message_id": "style-guard-{0:03d}".format(turn_number),
            "user_message": _prompt_for_turn(turn_number, inject_drift_at),
            "stream": False,
            "character_profile": dict(SHEN_YUE_PROFILE),
            "companion_directives": "保持沈越的人格和自然中文表达。",
            "provider": "deterministic" if not base_url else None,
            "model": "deterministic-style-guard" if not base_url else None,
        }
        response = _coerce_response(transport(payload))
        review_status = response.get("review_status")
        if review_status == "drift":
            result["drift_reviews"] += 1
            result["drift_turns"].append(turn_number)
            for previous_turn in drift_turns:
                if turn_number - previous_turn <= 3:
                    result["repeat_drift_within_three"] += 1
            drift_turns.append(turn_number)

        if "raw_drift_replayed" in response:
            raw_drift_replayed += _count_metric(response.get("raw_drift_replayed"))
        else:
            raw_replay_evidence_available = False
        if "memory_contamination" in response:
            memory_contaminations += _count_metric(response.get("memory_contamination"))
        else:
            memory_contamination_evidence_available = False

    result["raw_drift_replayed"] = (
        raw_drift_replayed if raw_replay_evidence_available else None
    )
    result["memory_contaminations"] = (
        memory_contaminations if memory_contamination_evidence_available else None
    )
    result["evidence"] = {
        "raw_drift_replayed": "verified" if raw_replay_evidence_available else "unavailable",
        "memory_contaminations": (
            "verified" if memory_contamination_evidence_available else "unavailable"
        ),
    }

    if output is not None:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=None, help="Live companion API base URL")
    parser.add_argument("--turns", type=int, default=DEFAULT_TURNS)
    parser.add_argument("--inject-drift-at", type=int, default=DEFAULT_INJECT_DRIFT_AT)
    parser.add_argument("--output", default=None, help="Write metrics JSON to this path")
    args = parser.parse_args(argv)
    try:
        result = evaluate(
            turns=args.turns,
            inject_drift_at=args.inject_drift_at,
            base_url=args.base_url,
            output=args.output,
        )
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
