import copy
import hashlib
import json
import re
import sqlite3
from datetime import datetime
from pathlib import Path

from agent.auxiliary_client import call_llm
from hermes_cli.companion_profile_policy import STYLE_GUARD_V1_POLICY, profile_lock, read_conversation_policy


RESULT_START = "<!-- COMPANION_TURN_REVIEW_RESULT "
RESULT_END = " COMPANION_TURN_REVIEW_RESULT -->"
PLACEHOLDER_TEXT = "【上一轮回复暂不纳入上下文，等待风格审查结果】"
_RESULT_RE = re.compile(
    re.escape(RESULT_START) + r"(.*?)" + re.escape(RESULT_END),
    re.DOTALL,
)
_ALLOWED_STATUS = frozenset(["pending", "clean", "drift", "invalid"])
_ALLOWED_DECISION = frozenset(["clean", "drift"])
_SELF_REVIEW_KEYS = (
    "fits_character_and_scene",
    "no_technical_false_positive",
    "summary_preserves_facts",
    "summary_adds_no_new_facts",
)
_TERMINAL_STYLE_STATUS = frozenset(["clean", "drift"])
_MEMORY_STATUS = frozenset(["none", "pending", "applied", "partial", "failed"])


class StaleTurnReviewError(RuntimeError):
    pass


class _AfterMemoryWriteError(RuntimeError):
    def __init__(self, original):
        RuntimeError.__init__(self, str(original))
        self.original = original


def assistant_sha256(text):
    return hashlib.sha256(str(text or "").encode("utf-8")).hexdigest()


def _utc_now():
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _stringify_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _normalize_review_json(value):
    if not value:
        return ""
    if isinstance(value, str):
        return value
    return _stringify_json(value)


def _parse_review_json(value):
    if not value:
        return {}
    try:
        return json.loads(value)
    except Exception:
        return {"raw": value}


def _public_memory_operation(operation):
    sanitized = dict(operation)
    sanitized.pop("evidence_quote", None)
    return sanitized


def _public_review_result(validated):
    sanitized = dict(validated)
    sanitized["memory_operations"] = [
        _public_memory_operation(item)
        for item in list(validated.get("memory_operations") or [])
    ]
    return sanitized


class TurnReviewStore(object):
    def __init__(self, profile_dir):
        self.profile_dir = Path(profile_dir)
        self.db_path = self.profile_dir / "companion_guard.db"
        self._ensure_db()

    def _ensure_db(self):
        self.profile_dir.mkdir(parents=True, exist_ok=True)
        with profile_lock(self.profile_dir, "companion_turn_guard_init"):
            conn = sqlite3.connect(str(self.db_path), timeout=30, isolation_level=None)
            try:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS turn_reviews (
                        turn_id TEXT PRIMARY KEY,
                        assistant_sha256 TEXT NOT NULL,
                        status TEXT NOT NULL,
                        style_reason TEXT NOT NULL DEFAULT '',
                        continuity_summary TEXT NOT NULL DEFAULT '',
                        review_json TEXT NOT NULL DEFAULT '',
                        memory_status TEXT NOT NULL DEFAULT '',
                        model TEXT NOT NULL DEFAULT '',
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS memory_operations (
                        turn_id TEXT NOT NULL,
                        assistant_sha256 TEXT NOT NULL,
                        operation_index INTEGER NOT NULL,
                        operation_json TEXT NOT NULL DEFAULT '',
                        status TEXT NOT NULL,
                        result_json TEXT NOT NULL DEFAULT '',
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        PRIMARY KEY (turn_id, assistant_sha256, operation_index)
                    )
                    """
                )
            finally:
                conn.close()

    def _connect(self):
        self._ensure_db()
        conn = sqlite3.connect(str(self.db_path), timeout=30, isolation_level=None)
        conn.row_factory = sqlite3.Row
        return conn

    def begin(self, turn_id, assistant_text):
        turn_id = str(turn_id or "").strip()
        if not turn_id:
            raise ValueError("turn_id is required")
        assistant_hash = assistant_sha256(assistant_text)
        now = _utc_now()
        with profile_lock(self.profile_dir, "companion_turn_guard_write"):
            conn = self._connect()
            try:
                conn.execute("BEGIN IMMEDIATE")
                row = conn.execute(
                    "SELECT * FROM turn_reviews WHERE turn_id = ?",
                    (turn_id,),
                ).fetchone()
                if row is None:
                    conn.execute(
                        """
                        INSERT INTO turn_reviews (
                            turn_id, assistant_sha256, status, style_reason,
                            continuity_summary, review_json, memory_status,
                            model, created_at, updated_at
                        ) VALUES (?, ?, 'pending', '', '', '', '', '', ?, ?)
                        """,
                        (turn_id, assistant_hash, now, now),
                    )
                elif row["assistant_sha256"] != assistant_hash:
                    conn.execute(
                        """
                        UPDATE turn_reviews
                           SET assistant_sha256 = ?,
                               status = 'pending',
                               style_reason = '',
                               continuity_summary = '',
                               review_json = '',
                               memory_status = '',
                               model = '',
                               updated_at = ?
                         WHERE turn_id = ?
                        """,
                        (assistant_hash, now, turn_id),
                    )
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
            finally:
                conn.close()
        return self.get(turn_id)

    def _commit_status(
        self,
        turn_id,
        assistant_hash,
        status,
        style_reason,
        continuity_summary,
        review_json,
        memory_status,
        model,
    ):
        if status not in _ALLOWED_STATUS:
            raise ValueError("invalid status")
        now = _utc_now()
        with profile_lock(self.profile_dir, "companion_turn_guard_write"):
            conn = self._connect()
            try:
                conn.execute("BEGIN IMMEDIATE")
                current = conn.execute(
                    "SELECT assistant_sha256 FROM turn_reviews WHERE turn_id = ?",
                    (turn_id,),
                ).fetchone()
                if current is None or current["assistant_sha256"] != assistant_hash:
                    raise StaleTurnReviewError("turn review is stale")
                conn.execute(
                    """
                    UPDATE turn_reviews
                       SET status = ?,
                           style_reason = ?,
                           continuity_summary = ?,
                           review_json = ?,
                           memory_status = ?,
                           model = ?,
                           updated_at = ?
                     WHERE turn_id = ? AND assistant_sha256 = ?
                    """,
                    (
                        status,
                        str(style_reason or ""),
                        str(continuity_summary or ""),
                        _normalize_review_json(review_json),
                        str(memory_status or ""),
                        str(model or ""),
                        now,
                        turn_id,
                        assistant_hash,
                    ),
                )
                if conn.total_changes <= 0:
                    raise StaleTurnReviewError("turn review is stale")
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
            finally:
                conn.close()
        return self.get(turn_id)

    def _set_memory_status(self, turn_id, assistant_hash, memory_status):
        if memory_status not in _MEMORY_STATUS:
            raise ValueError("invalid memory status")
        now = _utc_now()
        with profile_lock(self.profile_dir, "companion_turn_guard_write"):
            conn = self._connect()
            try:
                conn.execute("BEGIN IMMEDIATE")
                current = conn.execute(
                    "SELECT assistant_sha256 FROM turn_reviews WHERE turn_id = ?",
                    (turn_id,),
                ).fetchone()
                if current is None or current["assistant_sha256"] != assistant_hash:
                    raise StaleTurnReviewError("turn review is stale")
                conn.execute(
                    """
                    UPDATE turn_reviews
                       SET memory_status = ?, updated_at = ?
                     WHERE turn_id = ? AND assistant_sha256 = ?
                    """,
                    (memory_status, now, turn_id, assistant_hash),
                )
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
            finally:
                conn.close()
        return self.get(turn_id)

    def save_review_and_pending_ops(self, validated, model):
        turn_id = validated["turn_id"]
        assistant_hash = validated["assistant_sha256"]
        memory_operations = list(validated.get("memory_operations") or [])
        public_review = _public_review_result(validated)
        memory_status = "pending" if memory_operations else "none"
        now = _utc_now()
        with profile_lock(self.profile_dir, "companion_turn_guard_write"):
            conn = self._connect()
            try:
                conn.execute("BEGIN IMMEDIATE")
                current = conn.execute(
                    "SELECT assistant_sha256 FROM turn_reviews WHERE turn_id = ?",
                    (turn_id,),
                ).fetchone()
                if current is None or current["assistant_sha256"] != assistant_hash:
                    raise StaleTurnReviewError("turn review is stale")
                conn.execute(
                    """
                    UPDATE turn_reviews
                       SET status = ?,
                           style_reason = ?,
                           continuity_summary = ?,
                           review_json = ?,
                           memory_status = ?,
                           model = ?,
                           updated_at = ?
                     WHERE turn_id = ? AND assistant_sha256 = ?
                    """,
                    (
                        validated["style_decision"],
                        validated["style_reason"],
                        validated["continuity_summary"],
                        _normalize_review_json(public_review),
                        memory_status,
                        str(model or ""),
                        now,
                        turn_id,
                        assistant_hash,
                    ),
                )
                for index, operation in enumerate(memory_operations):
                    conn.execute(
                        """
                        INSERT OR IGNORE INTO memory_operations (
                            turn_id, assistant_sha256, operation_index,
                            operation_json, status, result_json, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, 'pending', '', ?, ?)
                        """,
                        (
                            turn_id,
                            assistant_hash,
                            index,
                            _normalize_review_json(operation),
                            now,
                            now,
                        ),
                    )
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
            finally:
                conn.close()
        return self.get(turn_id)

    def commit(self, result, model):
        validated = validate_review_result(
            result,
            user_message=str(result.get("_user_message") or ""),
            assistant_text_hash=str(result.get("assistant_sha256") or ""),
            allow_hash_only=True,
        )
        return self.save_review_and_pending_ops(validated, model)

    def mark_invalid(self, turn_id, assistant_hash, reason, review_json, model):
        try:
            return self._commit_status(
                turn_id,
                assistant_hash,
                "invalid",
                str(reason or ""),
                "",
                review_json,
                "none",
                model,
            )
        except StaleTurnReviewError:
            return None

    def get(self, turn_id):
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM turn_reviews WHERE turn_id = ?",
                (turn_id,),
            ).fetchone()
        finally:
            conn.close()
        if row is None:
            return None
        payload = dict(row)
        payload["review_json"] = _parse_review_json(payload.get("review_json"))
        return payload

    def get_memory_operations(self, turn_id, assistant_hash):
        conn = self._connect()
        try:
            rows = conn.execute(
                """
                SELECT operation_index, operation_json, status, result_json
                  FROM memory_operations
                 WHERE turn_id = ? AND assistant_sha256 = ?
                 ORDER BY operation_index ASC
                """,
                (turn_id, assistant_hash),
            ).fetchall()
        finally:
            conn.close()
        items = []
        for row in rows:
            items.append(
                {
                    "operation_index": row["operation_index"],
                    "operation_json": _parse_review_json(row["operation_json"]),
                    "status": row["status"],
                    "result_json": _parse_review_json(row["result_json"]),
                }
            )
        return items

    def update_memory_operation(self, turn_id, assistant_hash, operation_index, status, result_json):
        now = _utc_now()
        with profile_lock(self.profile_dir, "companion_turn_guard_write"):
            conn = self._connect()
            try:
                conn.execute("BEGIN IMMEDIATE")
                current = conn.execute(
                    "SELECT assistant_sha256 FROM turn_reviews WHERE turn_id = ?",
                    (turn_id,),
                ).fetchone()
                if current is None or current["assistant_sha256"] != assistant_hash:
                    raise StaleTurnReviewError("turn review is stale")
                conn.execute(
                    """
                    UPDATE memory_operations
                       SET status = ?, result_json = ?, updated_at = ?
                     WHERE turn_id = ? AND assistant_sha256 = ? AND operation_index = ?
                    """,
                    (
                        status,
                        _normalize_review_json(result_json),
                        now,
                        turn_id,
                        assistant_hash,
                        operation_index,
                    ),
                )
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
            finally:
                conn.close()

    def refresh_memory_status(self, turn_id, assistant_hash):
        operations = self.get_memory_operations(turn_id, assistant_hash)
        if not operations:
            memory_status = "none"
        else:
            statuses = [item["status"] for item in operations]
            if all(status == "applied" for status in statuses):
                memory_status = "applied"
            elif all(status == "failed" for status in statuses):
                memory_status = "failed"
            elif all(status == "pending" for status in statuses):
                memory_status = "pending"
            else:
                memory_status = "partial"
        self._set_memory_status(turn_id, assistant_hash, memory_status)
        return memory_status

    def list_unresolved(self):
        conn = self._connect()
        try:
            rows = conn.execute(
                """
                SELECT * FROM turn_reviews
                 WHERE status IN ('pending', 'invalid')
                 ORDER BY created_at ASC, turn_id ASC
                """
            ).fetchall()
        finally:
            conn.close()
        items = []
        for row in rows:
            payload = dict(row)
            payload["review_json"] = _parse_review_json(payload.get("review_json"))
            items.append(payload)
        return items


def _require_non_empty(value, field_name):
    text = str(value or "").strip()
    if not text:
        raise ValueError(field_name + " is required")
    return text


def _normalize_memory_operations(memory_operations, user_message):
    if not isinstance(memory_operations, list):
        raise ValueError("memory_operations must be a list")
    normalized = []
    for item in memory_operations:
        if not isinstance(item, dict):
            raise ValueError("memory_operations item must be an object")
        target = _require_non_empty(item.get("target"), "target")
        action = _require_non_empty(item.get("action"), "action")
        evidence_quote = _require_non_empty(item.get("evidence_quote"), "evidence_quote")
        if target != "user":
            raise ValueError("target must be user")
        if action not in ("add", "replace", "remove"):
            raise ValueError("action must be add|replace|remove")
        if evidence_quote not in user_message:
            raise ValueError("evidence_quote must be a substring of user_message")
        normalized_item = {
            "target": target,
            "action": action,
            "evidence_quote": evidence_quote,
        }
        if action == "add":
            normalized_item["content"] = _require_non_empty(item.get("content"), "content")
        elif action == "replace":
            normalized_item["old_text"] = _require_non_empty(item.get("old_text"), "old_text")
            normalized_item["content"] = _require_non_empty(item.get("content"), "content")
        else:
            normalized_item["old_text"] = _require_non_empty(item.get("old_text"), "old_text")
        normalized.append(normalized_item)
    return normalized


def validate_review_result(
    result,
    user_message,
    assistant_text=None,
    assistant_text_hash=None,
    allow_hash_only=False,
    expected_turn_id=None,
):
    if not isinstance(result, dict):
        raise ValueError("result must be an object")
    turn_id = _require_non_empty(result.get("turn_id"), "turn_id")
    if expected_turn_id is not None and turn_id != str(expected_turn_id):
        raise ValueError("turn_id does not match expected turn")
    expected_hash = _require_non_empty(result.get("assistant_sha256"), "assistant_sha256")
    if not re.match(r"^[0-9a-f]{64}$", expected_hash):
        raise ValueError("assistant_sha256 must be 64 lowercase hex chars")
    computed_hash = assistant_text_hash
    if computed_hash is None:
        if allow_hash_only:
            computed_hash = expected_hash
        else:
            computed_hash = assistant_sha256(assistant_text)
    if expected_hash != computed_hash:
        raise ValueError("assistant_sha256 does not match assistant_text")
    style_decision = _require_non_empty(result.get("style_decision"), "style_decision")
    if style_decision not in _ALLOWED_DECISION:
        raise ValueError("style_decision must be clean|drift")
    style_reason = _require_non_empty(result.get("style_reason"), "style_reason")
    if "continuity_summary" not in result:
        raise ValueError("continuity_summary is required")
    if not isinstance(result.get("continuity_summary"), str):
        raise ValueError("continuity_summary must be a string")
    continuity_summary = result.get("continuity_summary").strip()
    if style_decision == "drift" and not continuity_summary:
        raise ValueError("continuity_summary is required for drift")
    self_review = result.get("self_review")
    if not isinstance(self_review, dict):
        raise ValueError("self_review is required")
    normalized_review = {}
    rejected = False
    for key in _SELF_REVIEW_KEYS:
        verdict = _require_non_empty(self_review.get(key), "self_review." + key)
        if verdict not in ("pass", "reject"):
            raise ValueError("self_review.%s must be pass|reject" % key)
        normalized_review[key] = verdict
        if verdict == "reject":
            rejected = True
    verdict = _require_non_empty(result.get("verdict"), "verdict")
    if verdict not in ("pass", "reject"):
        raise ValueError("verdict must be pass|reject")
    if verdict == "reject" or rejected:
        raise ValueError("verdict/self_review reject cannot commit clean/drift")
    normalized = {
        "turn_id": turn_id,
        "assistant_sha256": expected_hash,
        "style_decision": style_decision,
        "style_reason": style_reason,
        "continuity_summary": continuity_summary,
        "memory_operations": _normalize_memory_operations(
            result["memory_operations"] if "memory_operations" in result else None,
            str(user_message or ""),
        ),
        "self_review": normalized_review,
        "verdict": verdict,
    }
    if "memory_status" in result:
        normalized["memory_status"] = str(result.get("memory_status") or "")
    return normalized


def _extract_review_payload(raw_output):
    raw_text = str(raw_output or "")
    matches = list(_RESULT_RE.finditer(raw_text))
    if len(matches) != 1:
        raise ValueError("missing structured result marker")
    match = matches[0]
    if raw_text.strip() != match.group(0):
        raise ValueError("structured result must be the only non-whitespace content")
    try:
        parsed = json.loads(match.group(1))
    except Exception as exc:
        raise ValueError("invalid structured result json: %s" % exc)
    if not isinstance(parsed, dict):
        raise ValueError("structured result must be an object")
    return parsed


def _build_review_request(turn_id, assistant_text, user_message):
    return (
        "请审查这一轮陪伴式回复是否保持角色与场景。\n"
        "判断原则：专业词本身不是 OOC；必须结合角色、场景、用户意图。"
        "非技术亲密场景里，如果把情感/控制持续翻译成系统运行语体且人格辨识度下降，才算 drift。"
        "角色回复不是用户记忆证据。\n"
        "输出必须且只能包含一个结构化 marker。\n"
        "turn_id: {turn_id}\n"
        "user_message:\n{user_message}\n"
        "assistant_text:\n{assistant_text}\n"
        "请填充字段：turn_id, assistant_sha256, style_decision, style_reason, continuity_summary, "
        "memory_operations, self_review, verdict。"
    ).format(
        turn_id=turn_id,
        user_message=str(user_message or ""),
        assistant_text=str(assistant_text or ""),
    )

def _supports_exact_lookup(memory_store):
    return bool(
        memory_store is not None
        and (hasattr(memory_store, "has_exact") or hasattr(memory_store, "_entries_for"))
    )


def _memory_store_has_exact(memory_store, target, text):
    if not _supports_exact_lookup(memory_store):
        return False
    if hasattr(memory_store, "has_exact"):
        return bool(memory_store.has_exact(target, text))
    if hasattr(memory_store, "_entries_for"):
        try:
            return text in list(memory_store._entries_for(target))
        except Exception:
            return False
    return False


def _operation_already_applied(memory_store, operation):
    if not _supports_exact_lookup(memory_store):
        return False
    target = operation["target"]
    action = operation["action"]
    if action == "add":
        return _memory_store_has_exact(memory_store, target, operation["content"])
    if action == "replace":
        has_old = _memory_store_has_exact(memory_store, target, operation["old_text"])
        has_new = _memory_store_has_exact(memory_store, target, operation["content"])
        return has_new and not has_old
    return not _memory_store_has_exact(memory_store, target, operation["old_text"])


def _memory_call_succeeded(result):
    return not isinstance(result, dict) or result.get("success", True) is not False


def _execute_memory_operation(memory_store, operation):
    action = operation["action"]
    if action == "add":
        return memory_store.add(operation["target"], operation["content"])
    if action == "replace":
        if _supports_exact_lookup(memory_store):
            has_old = _memory_store_has_exact(memory_store, operation["target"], operation["old_text"])
            has_new = _memory_store_has_exact(memory_store, operation["target"], operation["content"])
            if not has_old and has_new:
                return {"operation": "replace", "status": "already_applied", "content": operation["content"]}
            if not has_old and not has_new:
                raise RuntimeError("replace target missing")
        return memory_store.replace(operation["target"], operation["old_text"], operation["content"])
    return memory_store.remove(operation["target"], operation["old_text"])


def _resume_memory_operations(
    turn_store,
    record,
    memory_store,
    after_memory_write_hook=None,
):
    turn_id = record["turn_id"]
    assistant_hash = record["assistant_sha256"]
    ledger = turn_store.get_memory_operations(turn_id, assistant_hash)
    modifications = []
    for row in ledger:
        operation = row["operation_json"]
        if row["status"] == "applied":
            continue
        if _operation_already_applied(memory_store, operation):
            turn_store.update_memory_operation(
                turn_id,
                assistant_hash,
                row["operation_index"],
                "applied",
                {"status": "already_applied"},
            )
            continue
        try:
            result = _execute_memory_operation(memory_store, operation)
        except Exception as exc:
            turn_store.update_memory_operation(
                turn_id,
                assistant_hash,
                row["operation_index"],
                "failed",
                {"error": str(exc)},
            )
            continue
        if not _memory_call_succeeded(result):
            turn_store.update_memory_operation(
                turn_id,
                assistant_hash,
                row["operation_index"],
                "failed",
                result,
            )
            continue
        modifications.append(result)
        if after_memory_write_hook is not None:
            try:
                after_memory_write_hook(
                    turn_id=turn_id,
                    assistant_sha256=assistant_hash,
                    operation_index=row["operation_index"],
                    operation=operation,
                    result=result,
                )
            except Exception as exc:
                raise _AfterMemoryWriteError(exc)
        turn_store.update_memory_operation(
            turn_id,
            assistant_hash,
            row["operation_index"],
            "applied",
            result,
        )
    memory_status = turn_store.refresh_memory_status(turn_id, assistant_hash)
    return modifications, memory_status


def review_turn(
    profile_dir,
    turn_id,
    assistant_text,
    user_message,
    messages,
    provider,
    model,
    base_url=None,
    api_key=None,
    memory_store=None,
    store=None,
    call_llm_fn=call_llm,
    _after_memory_write_hook=None,
):
    if read_conversation_policy(profile_dir) != STYLE_GUARD_V1_POLICY:
        raise ValueError("profile is not style_guard_v1")
    turn_store = store or TurnReviewStore(profile_dir)
    current = turn_store.begin(turn_id, assistant_text)
    assistant_hash = current["assistant_sha256"]
    if current["status"] in _TERMINAL_STYLE_STATUS:
        try:
            modifications, memory_status = _resume_memory_operations(
                turn_store,
                current,
                memory_store,
                after_memory_write_hook=_after_memory_write_hook,
            )
        except StaleTurnReviewError:
            return {
                "turn_id": str(turn_id),
                "review_status": "stale",
                "memory_status": "pending",
                "memory_modifications": [],
            }
        existing = current["review_json"] if isinstance(current.get("review_json"), dict) else {}
        return {
            "turn_id": str(turn_id),
            "review_status": current["status"],
            "memory_status": memory_status,
            "memory_modifications": modifications,
            "review_result": existing,
        }
    review_messages = copy.deepcopy(list(messages or []))
    review_messages.append(
        {
            "role": "user",
            "content": _build_review_request(turn_id, assistant_text, user_message),
        }
    )
    try:
        raw_output = call_llm_fn(
            task="companion_turn_review",
            provider=provider,
            model=model,
            base_url=base_url,
            api_key=api_key,
            messages=review_messages,
            temperature=0.1,
            max_tokens=1200,
        )
        parsed = _extract_review_payload(raw_output)
        validated = validate_review_result(
            parsed,
            user_message=user_message,
            assistant_text=assistant_text,
            expected_turn_id=turn_id,
        )
        turn_store.save_review_and_pending_ops(validated, model)
        persisted = turn_store.get(validated["turn_id"])
        modifications, memory_status = _resume_memory_operations(
            turn_store,
            persisted,
            memory_store,
            after_memory_write_hook=_after_memory_write_hook,
        )
        public_review = _public_review_result(validated)
        public_review["memory_status"] = memory_status
        return {
            "turn_id": validated["turn_id"],
            "review_status": validated["style_decision"],
            "memory_status": memory_status,
            "memory_modifications": modifications,
            "review_result": public_review,
        }
    except StaleTurnReviewError:
        return {
            "turn_id": str(turn_id),
            "review_status": "stale",
            "memory_status": "pending",
            "memory_modifications": [],
        }
    except _AfterMemoryWriteError as exc:
        raise exc.original
    except Exception as exc:
        invalid_result = turn_store.mark_invalid(
            str(turn_id),
            assistant_hash,
            str(exc),
            {"error": str(exc)},
            model,
        )
        if invalid_result is None:
            return {
                "turn_id": str(turn_id),
                "review_status": "stale",
                "memory_status": "pending",
                "memory_modifications": [],
                "error": str(exc),
            }
        return {
            "turn_id": str(turn_id),
            "review_status": "invalid",
            "memory_status": "none",
            "memory_modifications": [],
            "error": str(exc),
        }


def build_guarded_history(messages, store):
    guarded = []
    previous_user_turn_id = None
    for message in list(messages or []):
        cloned = copy.deepcopy(message)
        role = cloned.get("role")
        if role == "user":
            previous_user_turn_id = cloned.get("message_id")
            guarded.append(cloned)
            continue
        if role != "assistant":
            guarded.append(cloned)
            continue
        turn_id = previous_user_turn_id
        if not turn_id:
            cloned["content"] = PLACEHOLDER_TEXT
            guarded.append(cloned)
            continue
        record = store.get(turn_id)
        if record is None or record.get("status") in ("pending", "invalid"):
            cloned["content"] = PLACEHOLDER_TEXT
            guarded.append(cloned)
            continue
        if record.get("assistant_sha256") != assistant_sha256(cloned.get("content")):
            cloned["content"] = PLACEHOLDER_TEXT
            guarded.append(cloned)
            continue
        if record.get("status") == "drift":
            cloned["content"] = "【上一轮事实摘要，仅用于承接事实】\n" + str(
                record.get("continuity_summary") or ""
            )
            guarded.append(cloned)
            continue
        guarded.append(cloned)
    return guarded
