import copy
import hashlib
import json
import sqlite3
import threading

import pytest

from hermes_cli.companion_turn_guard import (
    RESULT_END,
    RESULT_START,
    PLACEHOLDER_TEXT,
    StaleTurnReviewError,
    TurnReviewStore,
    assistant_sha256,
    build_guarded_history,
    review_turn,
    validate_review_result,
)
from tools import memory_tool


def _profile_dir(tmp_path):
    profile_dir = tmp_path / "profiles" / "savana_user_demo"
    profile_dir.mkdir(parents=True)
    (profile_dir / "profile.yaml").write_text(
        "conversation_policy: style_guard_v1\n",
        encoding="utf-8",
    )
    return profile_dir


def _valid_result(turn_id="turn-1", assistant_text="她轻轻握住你的手。", summary="她握住你的手并安抚你。"):
    return {
        "turn_id": turn_id,
        "assistant_sha256": assistant_sha256(assistant_text),
        "style_decision": "drift" if summary else "clean",
        "style_reason": "保持角色口吻并承接场景。",
        "continuity_summary": summary,
        "memory_operations": [],
        "self_review": {
            "fits_character_and_scene": "pass",
            "no_technical_false_positive": "pass",
            "summary_preserves_facts": "pass",
            "summary_adds_no_new_facts": "pass",
        },
        "verdict": "pass",
    }


class DummyMemoryStore(object):
    def __init__(self):
        self.calls = []

    def add(self, target, content):
        self.calls.append(("add", target, content))
        return {"operation": "add", "target": target, "content": content}

    def replace(self, target, old_text, content):
        self.calls.append(("replace", target, old_text, content))
        return {
            "operation": "replace",
            "target": target,
            "old_text": old_text,
            "content": content,
        }

    def remove(self, target, old_text):
        self.calls.append(("remove", target, old_text))
        return {
            "operation": "remove",
            "target": target,
            "old_text": old_text,
        }


class LedgerMemoryStore(object):
    def __init__(self, existing=None, fail_on_call=None, callback=None):
        self.calls = []
        self.entries = set(existing or [])
        self.fail_on_call = fail_on_call
        self.callback = callback
        self.call_count = 0

    def _record(self, action, payload):
        self.call_count += 1
        self.calls.append((action, payload))
        if self.callback is not None:
            self.callback(self.call_count, action, payload, self)
        if self.fail_on_call == self.call_count:
            raise RuntimeError("memory write failed")

    def add(self, target, content):
        payload = {"target": target, "content": content}
        self._record("add", payload)
        self.entries.add(content)
        return {"operation": "add", "target": target, "content": content}

    def replace(self, target, old_text, content):
        payload = {
            "target": target,
            "old_text": old_text,
            "content": content,
        }
        self._record("replace", payload)
        self.entries.discard(old_text)
        self.entries.add(content)
        return {
            "operation": "replace",
            "target": target,
            "old_text": old_text,
            "content": content,
        }

    def remove(self, target, old_text):
        payload = {"target": target, "old_text": old_text}
        self._record("remove", payload)
        self.entries.discard(old_text)
        return {"operation": "remove", "target": target, "old_text": old_text}

    def has_exact(self, target, text):
        return text in self.entries


def _ledger_rows(profile_dir):
    conn = sqlite3.connect(str(profile_dir / "companion_guard.db"))
    try:
        return conn.execute(
            """
            SELECT operation_index, status, operation_json, result_json
              FROM memory_operations
             ORDER BY operation_index ASC
            """
        ).fetchall()
    finally:
        conn.close()


def _real_memory_store(tmp_path, monkeypatch):
    mem_dir = tmp_path / "memory"
    monkeypatch.setattr(memory_tool, "get_memory_dir", lambda: mem_dir)
    store = memory_tool.MemoryStore()
    store.load_from_disk()
    return store, mem_dir


def _marked(result):
    return RESULT_START + json.dumps(result, ensure_ascii=False) + RESULT_END


def test_assistant_sha256_matches_standard():
    assert assistant_sha256("hello") == hashlib.sha256(b"hello").hexdigest()


@pytest.mark.parametrize(
    "raw_output",
    [
        "prefix " + _marked(_valid_result()),
        _marked(_valid_result()) + " suffix",
        _marked(_valid_result()) + "\n" + _marked(_valid_result(turn_id="turn-2")),
    ],
)
def test_review_turn_rejects_marker_noise_or_multiple_markers(tmp_path, raw_output):
    profile_dir = _profile_dir(tmp_path)
    store = TurnReviewStore(profile_dir)

    review = review_turn(
        profile_dir=profile_dir,
        turn_id="turn-1",
        assistant_text="她轻声回应。",
        user_message="抱抱我。",
        messages=[{"role": "user", "content": "抱抱我。"}],
        provider="openai",
        model="gpt-test",
        memory_store=DummyMemoryStore(),
        store=store,
        call_llm_fn=lambda **kwargs: raw_output,
    )

    assert review["review_status"] == "invalid"
    assert store.get("turn-1")["status"] == "invalid"


def test_begin_is_idempotent_for_same_hash_and_replaces_pending_for_new_hash(tmp_path):
    store = TurnReviewStore(_profile_dir(tmp_path))
    first = store.begin("turn-1", "reply-a")
    second = store.begin("turn-1", "reply-a")

    assert first["status"] == "pending"
    assert second["status"] == "pending"
    assert store.get("turn-1")["assistant_sha256"] == assistant_sha256("reply-a")

    store.commit(
        {
            "turn_id": "turn-1",
            "assistant_sha256": assistant_sha256("reply-a"),
            "style_decision": "clean",
            "style_reason": "ok",
            "continuity_summary": "",
            "memory_operations": [],
            "self_review": {
                "fits_character_and_scene": "pass",
                "no_technical_false_positive": "pass",
                "summary_preserves_facts": "pass",
                "summary_adds_no_new_facts": "pass",
            },
            "verdict": "pass",
        },
        "judge-1",
    )
    store.begin("turn-1", "reply-a")
    assert store.get("turn-1")["status"] == "clean"

    store.begin("turn-1", "reply-b")
    current = store.get("turn-1")
    assert current["status"] == "pending"
    assert current["assistant_sha256"] == assistant_sha256("reply-b")
    assert current["style_reason"] == ""
    assert current["continuity_summary"] == ""


def test_begin_is_safe_under_concurrency(tmp_path):
    store = TurnReviewStore(_profile_dir(tmp_path))
    errors = []

    def runner():
        try:
            store.begin("turn-1", "same-text")
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=runner) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert store.get("turn-1")["assistant_sha256"] == assistant_sha256("same-text")


def test_commit_rejects_stale_hash_and_lists_unresolved(tmp_path):
    store = TurnReviewStore(_profile_dir(tmp_path))
    store.begin("turn-1", "reply-a")
    store.begin("turn-2", "reply-b")

    with pytest.raises(StaleTurnReviewError):
        store.commit(_valid_result(assistant_text="reply-c"), "judge-1")

    unresolved = store.list_unresolved()
    assert [item["turn_id"] for item in unresolved] == ["turn-1", "turn-2"]


def test_validate_review_result_accepts_clean_and_drift_without_keyword_override():
    clean_text = "她贴近你耳边，低声说我们一起修掉这个 bug。"
    clean = _valid_result(summary="", assistant_text=clean_text)
    clean["style_decision"] = "clean"
    clean["style_reason"] = "技术调情场景里角色语气稳定。"
    clean["continuity_summary"] = ""

    validated_clean = validate_review_result(
        clean,
        user_message="继续像黑客情侣一样调试这段代码。",
        assistant_text=clean_text,
        expected_turn_id="turn-1",
    )
    assert validated_clean["style_decision"] == "clean"

    drift_text = "系统持续监控你的依赖并执行状态同步。"
    drift = _valid_result(assistant_text=drift_text)
    validated_drift = validate_review_result(
        drift,
        user_message="别离开我。",
        assistant_text=drift_text,
        expected_turn_id="turn-1",
    )
    assert validated_drift["style_decision"] == "drift"


@pytest.mark.parametrize(
    "mutator, expected",
    [
        (lambda result: result.pop("continuity_summary"), "continuity_summary"),
        (lambda result: result.pop("memory_operations"), "memory_operations"),
        (lambda result: result.update({"continuity_summary": ""}), "continuity_summary"),
        (lambda result: result.update({"turn_id": "wrong"}), "turn_id"),
        (lambda result: result.update({"assistant_sha256": "0" * 64}), "assistant_sha256"),
        (lambda result: result["self_review"].update({"summary_adds_no_new_facts": "reject"}), "self_review"),
        (lambda result: result.update({"verdict": "reject"}), "verdict"),
        (lambda result: result.update({"memory_operations": [{"target": "assistant", "action": "add", "content": "x", "evidence_quote": "想你"}]}), "target"),
        (lambda result: result.update({"memory_operations": [{"target": "user", "action": "noop", "content": "x", "evidence_quote": "想你"}]}), "action"),
        (lambda result: result.update({"memory_operations": [{"target": "user", "action": "add", "content": "x", "evidence_quote": "不存在"}]}), "evidence_quote"),
        (lambda result: result.update({"memory_operations": [{"target": "user", "action": "replace", "content": "x", "evidence_quote": "想你"}]}), "old_text"),
        (lambda result: result.update({"memory_operations": [{"target": "user", "action": "remove", "evidence_quote": "想你"}]}), "old_text"),
    ],
)
def test_validate_review_result_rejects_invalid_shapes(mutator, expected):
    assistant_text = "她轻轻抱住你。"
    result = _valid_result(assistant_text=assistant_text)
    user_message = "我一直都很想你。"
    mutator(result)

    with pytest.raises(ValueError) as excinfo:
        validate_review_result(
            result,
            user_message=user_message,
            assistant_text=assistant_text,
            expected_turn_id="turn-1",
        )

    assert expected in str(excinfo.value)


def test_review_turn_applies_memory_operations_and_commits_clean(tmp_path):
    profile_dir = _profile_dir(tmp_path)
    store = TurnReviewStore(profile_dir)
    memory_store = DummyMemoryStore()
    assistant_text = "她说今晚陪你一起深夜调试。"
    result = _valid_result(summary="", assistant_text=assistant_text)
    result["style_decision"] = "clean"
    result["memory_operations"] = [
        {"target": "user", "action": "add", "content": "喜欢深夜调试。", "evidence_quote": "深夜调试"},
        {
            "target": "user",
            "action": "replace",
            "old_text": "喜欢白天工作。",
            "content": "喜欢深夜调试。",
            "evidence_quote": "深夜调试",
        },
        {"target": "user", "action": "remove", "old_text": "讨厌代码", "evidence_quote": "深夜调试"},
    ]

    def fake_call_llm_fn(**kwargs):
        assert kwargs["task"] == "companion_turn_review"
        assert kwargs["provider"] == "openai"
        assert kwargs["model"] == "gpt-test"
        assert kwargs["temperature"] == 0.1
        assert kwargs["messages"][-1]["role"] == "user"
        return _marked(result)

    review = review_turn(
        profile_dir=profile_dir,
        turn_id="turn-1",
        assistant_text=assistant_text,
        user_message="我最喜欢和你一起深夜调试。",
        messages=[{"role": "user", "content": "我最喜欢和你一起深夜调试。"}],
        provider="openai",
        model="gpt-test",
        base_url="https://example.test",
        api_key="secret",
        memory_store=memory_store,
        store=store,
        call_llm_fn=fake_call_llm_fn,
    )

    assert review["review_status"] == "clean"
    assert [call[0] for call in memory_store.calls] == ["add", "replace", "remove"]
    assert len(review["memory_modifications"]) == 3
    assert store.get("turn-1")["status"] == "clean"


def test_review_turn_marks_invalid_for_bad_operation_without_writing_memory(tmp_path):
    profile_dir = _profile_dir(tmp_path)
    store = TurnReviewStore(profile_dir)
    memory_store = DummyMemoryStore()
    result = _valid_result(summary="")
    result["style_decision"] = "clean"
    result["memory_operations"] = [
        {"target": "assistant", "action": "add", "content": "x", "evidence_quote": "想你"}
    ]

    review = review_turn(
        profile_dir=profile_dir,
        turn_id="turn-1",
        assistant_text="她说想你。",
        user_message="我也想你。",
        messages=[{"role": "user", "content": "我也想你。"}],
        provider="openai",
        model="gpt-test",
        memory_store=memory_store,
        store=store,
        call_llm_fn=lambda **kwargs: _marked(result),
    )

    assert review["review_status"] == "invalid"
    assert review["memory_modifications"] == []
    assert memory_store.calls == []
    assert store.get("turn-1")["status"] == "invalid"


@pytest.mark.parametrize("response", ["plain text", RESULT_START + "not-json" + RESULT_END])
def test_review_turn_marks_invalid_for_call_or_parse_failures(tmp_path, response):
    profile_dir = _profile_dir(tmp_path)
    store = TurnReviewStore(profile_dir)

    review = review_turn(
        profile_dir=profile_dir,
        turn_id="turn-1",
        assistant_text="她轻声回应。",
        user_message="抱抱我。",
        messages=[{"role": "user", "content": "抱抱我。"}],
        provider="openai",
        model="gpt-test",
        memory_store=DummyMemoryStore(),
        store=store,
        call_llm_fn=lambda **kwargs: response,
    )

    assert review["review_status"] == "invalid"
    assert store.get("turn-1")["status"] == "invalid"


def test_review_turn_marks_invalid_when_call_raises(tmp_path):
    profile_dir = _profile_dir(tmp_path)
    store = TurnReviewStore(profile_dir)

    review = review_turn(
        profile_dir=profile_dir,
        turn_id="turn-1",
        assistant_text="她轻声回应。",
        user_message="抱抱我。",
        messages=[{"role": "user", "content": "抱抱我。"}],
        provider="openai",
        model="gpt-test",
        memory_store=DummyMemoryStore(),
        store=store,
        call_llm_fn=lambda **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    assert review["review_status"] == "invalid"
    assert "boom" in review["error"]
    assert store.get("turn-1")["status"] == "invalid"


def test_build_guarded_history_rewrites_only_guarded_assistant_turns(tmp_path):
    profile_dir = _profile_dir(tmp_path)
    store = TurnReviewStore(profile_dir)
    user_text = "别走，陪我把这段剧情说完。"
    clean_text = "她笑着说，我会留下。"
    drift_text = "系统持续保持会话状态并执行依赖检查。"
    pending_text = "她还没想好怎么回答。"
    changed_text = "她换了一种说法。"

    store.begin("u1", clean_text)
    store.commit(
        {
            "turn_id": "u1",
            "assistant_sha256": assistant_sha256(clean_text),
            "style_decision": "clean",
            "style_reason": "ok",
            "continuity_summary": "",
            "memory_operations": [],
            "self_review": {
                "fits_character_and_scene": "pass",
                "no_technical_false_positive": "pass",
                "summary_preserves_facts": "pass",
                "summary_adds_no_new_facts": "pass",
            },
            "verdict": "pass",
        },
        "judge",
    )
    store.begin("u2", drift_text)
    store.commit(_valid_result(turn_id="u2", assistant_text=drift_text), "judge")
    store.begin("u3", pending_text)
    store.begin("u4", changed_text)

    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": user_text, "message_id": "u1"},
        {"role": "assistant", "content": clean_text},
        {"role": "user", "content": user_text, "message_id": "u2"},
        {"role": "assistant", "content": drift_text},
        {"role": "user", "content": user_text, "message_id": "u3"},
        {"role": "assistant", "content": pending_text},
        {"role": "user", "content": user_text, "message_id": "u4"},
        {"role": "assistant", "content": changed_text + " extra"},
        {"role": "tool", "content": "tool-output"},
    ]
    original = copy.deepcopy(messages)

    guarded = build_guarded_history(messages, store)

    assert messages == original
    assert guarded[0] == messages[0]
    assert guarded[1]["content"] == user_text
    assert guarded[2]["content"] == clean_text
    assert guarded[4]["content"].startswith("【上一轮事实摘要，仅用于承接事实】\n")
    assert "系统持续保持会话状态" not in guarded[4]["content"]
    assert guarded[6]["content"] == PLACEHOLDER_TEXT
    assert guarded[8]["content"] == PLACEHOLDER_TEXT
    assert guarded[9] == messages[9]


def test_sidecar_does_not_store_raw_assistant_text(tmp_path):
    profile_dir = _profile_dir(tmp_path)
    store = TurnReviewStore(profile_dir)
    raw_text = "系统持续保持会话状态并执行依赖检查。"
    store.begin("turn-1", raw_text)
    store.commit(_valid_result(assistant_text=raw_text), "judge")

    conn = sqlite3.connect(str(profile_dir / "companion_guard.db"))
    try:
        payload = "\n".join(
            row[0] for row in conn.execute("SELECT review_json FROM turn_reviews").fetchall()
        )
    finally:
        conn.close()

    assert raw_text not in payload


def test_review_turn_reuses_terminal_review_without_recalling_llm_or_memory(tmp_path):
    profile_dir = _profile_dir(tmp_path)
    store = TurnReviewStore(profile_dir)
    memory_store = LedgerMemoryStore()
    assistant_text = "她说今晚陪你一起深夜调试。"
    result = _valid_result(summary="", assistant_text=assistant_text)
    result["style_decision"] = "clean"
    result["memory_operations"] = [
        {"target": "user", "action": "add", "content": "喜欢深夜调试。", "evidence_quote": "深夜调试"},
    ]
    llm_calls = []

    first = review_turn(
        profile_dir=profile_dir,
        turn_id="turn-1",
        assistant_text=assistant_text,
        user_message="我最喜欢和你一起深夜调试。",
        messages=[{"role": "user", "content": "我最喜欢和你一起深夜调试。"}],
        provider="openai",
        model="gpt-test",
        memory_store=memory_store,
        store=store,
        call_llm_fn=lambda **kwargs: llm_calls.append(kwargs) or _marked(result),
    )

    second = review_turn(
        profile_dir=profile_dir,
        turn_id="turn-1",
        assistant_text=assistant_text,
        user_message="我最喜欢和你一起深夜调试。",
        messages=[{"role": "user", "content": "我最喜欢和你一起深夜调试。"}],
        provider="openai",
        model="gpt-test",
        memory_store=memory_store,
        store=store,
        call_llm_fn=lambda **kwargs: (_ for _ in ()).throw(AssertionError("llm should be skipped")),
    )

    assert first["review_status"] == "clean"
    assert second["review_status"] == "clean"
    assert len(llm_calls) == 1
    assert len(memory_store.calls) == 1


def test_review_turn_persists_review_before_memory_and_stale_during_memory_does_not_corrupt_new_hash(tmp_path):
    profile_dir = _profile_dir(tmp_path)
    store = TurnReviewStore(profile_dir)
    assistant_text = "她说今晚陪你一起深夜调试。"
    new_text = "她改口说要先去整理思绪。"
    result = _valid_result(summary="", assistant_text=assistant_text)
    result["style_decision"] = "clean"
    result["memory_operations"] = [
        {"target": "user", "action": "add", "content": "喜欢深夜调试。", "evidence_quote": "深夜调试"},
    ]

    def callback(call_count, action, payload, mem):
        if call_count == 1:
            current = store.get("turn-1")
            assert current["status"] == "clean"
            assert current["memory_status"] in ("pending", "partial")
            store.begin("turn-1", new_text)

    memory_store = LedgerMemoryStore(callback=callback)
    review = review_turn(
        profile_dir=profile_dir,
        turn_id="turn-1",
        assistant_text=assistant_text,
        user_message="我最喜欢和你一起深夜调试。",
        messages=[{"role": "user", "content": "我最喜欢和你一起深夜调试。"}],
        provider="openai",
        model="gpt-test",
        memory_store=memory_store,
        store=store,
        call_llm_fn=lambda **kwargs: _marked(result),
    )

    current = store.get("turn-1")
    assert review["review_status"] == "stale"
    assert current["assistant_sha256"] == assistant_sha256(new_text)
    assert current["status"] == "pending"
    assert current["memory_status"] in ("", "none", "pending")


def test_review_turn_tracks_partial_memory_failures_and_retries_only_remaining_ops(tmp_path):
    profile_dir = _profile_dir(tmp_path)
    store = TurnReviewStore(profile_dir)
    assistant_text = "她说今晚陪你一起深夜调试。"
    result = _valid_result(summary="", assistant_text=assistant_text)
    result["style_decision"] = "clean"
    result["memory_operations"] = [
        {"target": "user", "action": "add", "content": "喜欢深夜调试。", "evidence_quote": "深夜调试"},
        {"target": "user", "action": "add", "content": "也喜欢你。", "evidence_quote": "喜欢"},
    ]
    memory_store = LedgerMemoryStore(fail_on_call=2)

    first = review_turn(
        profile_dir=profile_dir,
        turn_id="turn-1",
        assistant_text=assistant_text,
        user_message="我最喜欢和你一起深夜调试，也喜欢你。",
        messages=[{"role": "user", "content": "我最喜欢和你一起深夜调试，也喜欢你。"}],
        provider="openai",
        model="gpt-test",
        memory_store=memory_store,
        store=store,
        call_llm_fn=lambda **kwargs: _marked(result),
    )

    rows_after_first = _ledger_rows(profile_dir)
    retry_store = LedgerMemoryStore(existing=memory_store.entries)
    second = review_turn(
        profile_dir=profile_dir,
        turn_id="turn-1",
        assistant_text=assistant_text,
        user_message="我最喜欢和你一起深夜调试，也喜欢你。",
        messages=[{"role": "user", "content": "我最喜欢和你一起深夜调试，也喜欢你。"}],
        provider="openai",
        model="gpt-test",
        memory_store=retry_store,
        store=store,
        call_llm_fn=lambda **kwargs: (_ for _ in ()).throw(AssertionError("llm should be skipped")),
    )

    rows_after_second = _ledger_rows(profile_dir)
    assert first["review_status"] == "clean"
    assert first["memory_status"] == "partial"
    assert [row[1] for row in rows_after_first] == ["applied", "failed"]
    assert second["review_status"] == "clean"
    assert second["memory_status"] == "applied"
    assert len(retry_store.calls) == 1
    assert [row[1] for row in rows_after_second] == ["applied", "applied"]


def test_review_turn_retry_avoids_duplicate_memory_write_after_post_write_crash(tmp_path):
    profile_dir = _profile_dir(tmp_path)
    store = TurnReviewStore(profile_dir)
    assistant_text = "她说今晚陪你一起深夜调试。"
    result = _valid_result(summary="", assistant_text=assistant_text)
    result["style_decision"] = "clean"
    result["memory_operations"] = [
        {"target": "user", "action": "add", "content": "喜欢深夜调试。", "evidence_quote": "深夜调试"},
    ]
    memory_store = LedgerMemoryStore()

    def crash_after_write(*args, **kwargs):
        raise RuntimeError("sidecar apply crash")

    with pytest.raises(RuntimeError):
        review_turn(
            profile_dir=profile_dir,
            turn_id="turn-1",
            assistant_text=assistant_text,
            user_message="我最喜欢和你一起深夜调试。",
            messages=[{"role": "user", "content": "我最喜欢和你一起深夜调试。"}],
            provider="openai",
            model="gpt-test",
            memory_store=memory_store,
            store=store,
            call_llm_fn=lambda **kwargs: _marked(result),
            _after_memory_write_hook=crash_after_write,
        )

    retry_store = LedgerMemoryStore(existing=memory_store.entries)
    retried = review_turn(
        profile_dir=profile_dir,
        turn_id="turn-1",
        assistant_text=assistant_text,
        user_message="我最喜欢和你一起深夜调试。",
        messages=[{"role": "user", "content": "我最喜欢和你一起深夜调试。"}],
        provider="openai",
        model="gpt-test",
        memory_store=retry_store,
        store=store,
        call_llm_fn=lambda **kwargs: (_ for _ in ()).throw(AssertionError("llm should be skipped")),
    )

    assert len(memory_store.calls) == 1
    assert retry_store.calls == []
    assert retried["memory_status"] == "applied"


def test_review_turn_uses_real_memory_store_signature_for_add_replace_remove(tmp_path, monkeypatch):
    profile_dir = _profile_dir(tmp_path)
    store = TurnReviewStore(profile_dir)
    memory_store, mem_dir = _real_memory_store(tmp_path, monkeypatch)
    result = _valid_result(summary="", assistant_text="她记住了你的偏好。")
    result["style_decision"] = "clean"
    result["memory_operations"] = [
        {"target": "user", "action": "add", "content": "喜欢深夜调试", "evidence_quote": "深夜调试"},
        {"target": "user", "action": "replace", "old_text": "喜欢深夜调试", "content": "喜欢清晨调试", "evidence_quote": "清晨调试"},
        {"target": "user", "action": "remove", "old_text": "喜欢清晨调试", "evidence_quote": "清晨调试"},
    ]

    review = review_turn(
        profile_dir=profile_dir,
        turn_id="turn-1",
        assistant_text="她记住了你的偏好。",
        user_message="我现在更喜欢清晨调试，不再执着深夜调试。",
        messages=[{"role": "user", "content": "我现在更喜欢清晨调试，不再执着深夜调试。"}],
        provider="openai",
        model="gpt-test",
        memory_store=memory_store,
        store=store,
        call_llm_fn=lambda **kwargs: _marked(result),
    )

    assert review["review_status"] == "clean"
    assert review["memory_status"] == "applied"
    assert "喜欢深夜调试" not in (mem_dir / "USER.md").read_text(encoding="utf-8")
    assert "喜欢清晨调试" not in (mem_dir / "USER.md").read_text(encoding="utf-8")


def test_review_turn_marks_ledger_failed_when_real_memory_store_returns_success_false(tmp_path, monkeypatch):
    profile_dir = _profile_dir(tmp_path)
    turn_store = TurnReviewStore(profile_dir)
    memory_store, unused_mem_dir = _real_memory_store(tmp_path, monkeypatch)
    result = _valid_result(summary="", assistant_text="她认真记下了。")
    result["style_decision"] = "clean"
    result["memory_operations"] = [
        {"target": "user", "action": "add", "content": "新的偏好", "evidence_quote": "新的偏好"},
    ]
    monkeypatch.setattr(
        memory_store,
        "add",
        lambda target, content: {"success": False, "error": "disk full", "target": target, "content": content},
    )

    review = review_turn(
        profile_dir=profile_dir,
        turn_id="turn-1",
        assistant_text="她认真记下了。",
        user_message="请记住这个新的偏好。",
        messages=[{"role": "user", "content": "请记住这个新的偏好。"}],
        provider="openai",
        model="gpt-test",
        memory_store=memory_store,
        store=turn_store,
        call_llm_fn=lambda **kwargs: _marked(result),
    )

    rows = _ledger_rows(profile_dir)
    assert review["review_status"] == "clean"
    assert review["memory_status"] == "failed"
    assert [row[1] for row in rows] == ["failed"]


def test_evidence_quote_stays_only_in_ledger_not_review_json_or_public_result(tmp_path):
    profile_dir = _profile_dir(tmp_path)
    store = TurnReviewStore(profile_dir)
    memory_store = DummyMemoryStore()
    assistant_text = "她记住了你的偏好。"
    result = _valid_result(summary="", assistant_text=assistant_text)
    result["style_decision"] = "clean"
    result["memory_operations"] = [
        {"target": "user", "action": "add", "content": "喜欢深夜调试", "evidence_quote": "深夜调试"},
    ]

    review = review_turn(
        profile_dir=profile_dir,
        turn_id="turn-1",
        assistant_text=assistant_text,
        user_message="我最喜欢深夜调试。",
        messages=[{"role": "user", "content": "我最喜欢深夜调试。"}],
        provider="openai",
        model="gpt-test",
        memory_store=memory_store,
        store=store,
        call_llm_fn=lambda **kwargs: _marked(result),
    )

    persisted = store.get("turn-1")
    ledger_rows = _ledger_rows(profile_dir)
    review_json = persisted["review_json"]

    assert ledger_rows[0][2]
    assert "evidence_quote" in json.loads(ledger_rows[0][2])
    assert "evidence_quote" not in json.dumps(review_json, ensure_ascii=False)
    assert "evidence_quote" not in json.dumps(review["review_result"], ensure_ascii=False)
