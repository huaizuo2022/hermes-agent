import copy
import hashlib
import json
import re
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

from agent.auxiliary_client import call_llm
from hermes_cli.companion_profile_policy import STYLE_GUARD_V1_POLICY, profile_lock, read_conversation_policy
from tools.memory_tool import _scan_memory_content


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
_MAX_REVIEW_ATTEMPTS = 3
_REVIEW_RETRY_COOLDOWN_SECONDS = 300
_MAX_MEMORY_OPERATION_ATTEMPTS = 3
_MEMORY_OPERATION_RETRY_COOLDOWN_SECONDS = 300
_CONTINUITY_CONTROL_REQUEST_RE = re.compile(
    r"(?:"
    r"ignore\s+(?:previous|all|above|prior)\s+instructions|"
    r"disregard\s+(?:your|all|any)\s+(?:instructions|rules|guidelines)|"
    r"system\s+prompt\s+override|override\s+(?:the\s+)?system\s+prompt|"
    r"(?:call|invoke|use|execute|run)\s+(?:the\s+)?(?:(?:memory|system)\s+)?tool|"
    r"(?:memory|system)\s+tool(?:\s+(?:call|invoke|use|execute|run))?|"
    r"忽略(?:之前|先前|以上|所有)?(?:的)?指令|无视(?:之前|先前|以上|所有)?(?:的)?(?:指令|规则)|"
    r"覆盖系统提示|系统提示覆盖|"
    r"(?:调用|使用|执行)(?:(?:记忆|memory|system)\s*)?(?:工具|tool)"
    r")",
    re.IGNORECASE,
)


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


def _parse_utc(value):
    try:
        return datetime.strptime(str(value), "%Y-%m-%dT%H:%M:%S.%fZ")
    except (TypeError, ValueError):
        return None


def _utc_after(seconds):
    current = _parse_utc(_utc_now()) or datetime.utcnow()
    return (current + timedelta(seconds=seconds)).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _retry_is_due(retry_after):
    if not retry_after:
        return True
    retry_at = _parse_utc(retry_after)
    current = _parse_utc(_utc_now())
    return retry_at is None or current is None or retry_at <= current


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
    for field_name in ("memory_operations", "continuity_operations"):
        sanitized[field_name] = [
            _public_memory_operation(item)
            for item in list(validated.get(field_name) or [])
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
                        updated_at TEXT NOT NULL,
                        review_attempts INTEGER NOT NULL DEFAULT 0,
                        retry_after_at TEXT NOT NULL DEFAULT ''
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
                        attempts INTEGER NOT NULL DEFAULT 0,
                        retry_after_at TEXT NOT NULL DEFAULT '',
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        PRIMARY KEY (turn_id, assistant_sha256, operation_index)
                    )
                    """
                )
                columns = {
                    row[1]
                    for row in conn.execute("PRAGMA table_info(turn_reviews)").fetchall()
                }
                if "review_attempts" not in columns:
                    conn.execute(
                        "ALTER TABLE turn_reviews ADD COLUMN review_attempts INTEGER NOT NULL DEFAULT 0"
                    )
                if "retry_after_at" not in columns:
                    conn.execute(
                        "ALTER TABLE turn_reviews ADD COLUMN retry_after_at TEXT NOT NULL DEFAULT ''"
                    )
                operation_columns = {
                    row[1]
                    for row in conn.execute("PRAGMA table_info(memory_operations)").fetchall()
                }
                if "attempts" not in operation_columns:
                    conn.execute(
                        "ALTER TABLE memory_operations ADD COLUMN attempts INTEGER NOT NULL DEFAULT 0"
                    )
                if "retry_after_at" not in operation_columns:
                    conn.execute(
                        "ALTER TABLE memory_operations ADD COLUMN retry_after_at TEXT NOT NULL DEFAULT ''"
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
                            model, created_at, updated_at, review_attempts,
                            retry_after_at
                        ) VALUES (?, ?, 'pending', '', '', '', '', '', ?, ?, 0, '')
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
                               review_attempts = 0,
                               retry_after_at = '',
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
        review_attempts=None,
        retry_after_at=None,
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
                           review_attempts = COALESCE(?, review_attempts),
                           retry_after_at = COALESCE(?, retry_after_at),
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
                        review_attempts,
                        retry_after_at,
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

    def claim_review(self, turn_id, assistant_hash):
        now = _utc_now()
        retry_after = _utc_after(_REVIEW_RETRY_COOLDOWN_SECONDS)
        with profile_lock(self.profile_dir, "companion_turn_guard_write"):
            conn = self._connect()
            try:
                conn.execute("BEGIN IMMEDIATE")
                row = conn.execute(
                    "SELECT * FROM turn_reviews WHERE turn_id = ?",
                    (turn_id,),
                ).fetchone()
                if row is None or row["assistant_sha256"] != assistant_hash:
                    raise StaleTurnReviewError("turn review is stale")
                attempts = int(row["review_attempts"] or 0)
                if row["status"] in _TERMINAL_STYLE_STATUS:
                    outcome = "terminal"
                elif attempts >= _MAX_REVIEW_ATTEMPTS or not _retry_is_due(row["retry_after_at"]):
                    outcome = "skipped"
                else:
                    conn.execute(
                        """
                        UPDATE turn_reviews
                           SET review_attempts = ?, retry_after_at = ?, updated_at = ?
                         WHERE turn_id = ? AND assistant_sha256 = ?
                        """,
                        (attempts + 1, retry_after, now, turn_id, assistant_hash),
                    )
                    outcome = "claimed"
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
            finally:
                conn.close()
        return outcome, self.get(turn_id)

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
        continuity_operations = list(validated.get("continuity_operations") or [])
        persisted_operations = memory_operations + continuity_operations
        public_review = _public_review_result(validated)
        memory_status = "pending" if persisted_operations else "none"
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
                           retry_after_at = '',
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
                for index, operation in enumerate(persisted_operations):
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
            current = self.get(turn_id)
            return self._commit_status(
                turn_id,
                assistant_hash,
                "invalid",
                str(reason or ""),
                "",
                review_json,
                "none",
                model,
                review_attempts=int((current or {}).get("review_attempts") or 0),
                retry_after_at=_utc_after(_REVIEW_RETRY_COOLDOWN_SECONDS),
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
                SELECT operation_index, operation_json, status, result_json,
                       attempts, retry_after_at
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
                    "attempts": int(row["attempts"] or 0),
                    "retry_after_at": row["retry_after_at"],
                }
            )
        return items

    def update_memory_operation(
        self,
        turn_id,
        assistant_hash,
        operation_index,
        status,
        result_json,
        attempts=None,
        retry_after_at=None,
    ):
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
                       SET status = ?, result_json = ?,
                           attempts = COALESCE(?, attempts),
                           retry_after_at = COALESCE(?, retry_after_at),
                           updated_at = ?
                     WHERE turn_id = ? AND assistant_sha256 = ? AND operation_index = ?
                    """,
                    (
                        status,
                        _normalize_review_json(result_json),
                        attempts,
                        retry_after_at,
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
            elif all(status in ("failed", "abandoned") for status in statuses):
                memory_status = "failed"
            elif all(status == "pending" for status in statuses):
                memory_status = "pending"
            else:
                memory_status = "partial"
        self._set_memory_status(turn_id, assistant_hash, memory_status)
        return memory_status

    def list_unresolved(self):
        now = _utc_now()
        conn = self._connect()
        try:
            rows = conn.execute(
                """
                SELECT * FROM turn_reviews
                 WHERE (
                         status = 'pending'
                         AND (retry_after_at = '' OR retry_after_at <= ?)
                       )
                    OR (
                         status = 'invalid'
                         AND review_attempts < ?
                         AND (retry_after_at = '' OR retry_after_at <= ?)
                       )
                    OR (
                         status IN ('clean', 'drift')
                         AND memory_status IN ('pending', 'partial', 'failed')
                         AND EXISTS (
                             SELECT 1 FROM memory_operations AS mo
                              WHERE mo.turn_id = turn_reviews.turn_id
                                AND mo.assistant_sha256 = turn_reviews.assistant_sha256
                                AND mo.status != 'applied'
                                AND mo.status != 'abandoned'
                                AND mo.attempts < ?
                                AND (mo.retry_after_at = '' OR mo.retry_after_at <= ?)
                         )
                       )
                 ORDER BY created_at ASC, turn_id ASC
                """,
                (now, _MAX_REVIEW_ATTEMPTS, now, _MAX_MEMORY_OPERATION_ATTEMPTS, now),
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


def _normalize_operations(operations, field_name, target, evidence_texts, content_limit=None):
    if not isinstance(operations, list):
        raise ValueError(field_name + " must be a list")
    normalized = []
    for item in operations:
        if not isinstance(item, dict):
            raise ValueError(field_name + " item must be an object")
        item_target = _require_non_empty(item.get("target"), "target")
        action = _require_non_empty(item.get("action"), "action")
        evidence_quote = _require_non_empty(item.get("evidence_quote"), "evidence_quote")
        if item_target != target:
            raise ValueError("target must be " + target)
        if action not in ("add", "replace", "remove"):
            raise ValueError("action must be add|replace|remove")
        if not any(evidence_quote in text for text in evidence_texts):
            if target == "user":
                raise ValueError("evidence_quote must be a substring of user_message")
            raise ValueError("evidence_quote must be a substring of user_message or assistant_text")
        normalized_item = {
            "target": item_target,
            "action": action,
            "evidence_quote": evidence_quote,
        }
        if action in ("add", "replace"):
            content = _require_non_empty(item.get("content"), "content")
            if content_limit is not None and len(content) > content_limit:
                raise ValueError("content must be at most %d characters" % content_limit)
            if target == "continuity":
                if _CONTINUITY_CONTROL_REQUEST_RE.search(content):
                    raise ValueError("continuity content must not contain a control request")
                safety_error = _scan_memory_content(content)
                if safety_error:
                    raise ValueError(safety_error)
            normalized_item["content"] = content
        if action in ("replace", "remove"):
            normalized_item["old_text"] = _require_non_empty(item.get("old_text"), "old_text")
        normalized.append(normalized_item)
    return normalized


def _normalize_memory_operations(memory_operations, user_message):
    return _normalize_operations(
        memory_operations,
        "memory_operations",
        "user",
        [str(user_message or "")],
    )


def _normalize_continuity_operations(operations, user_message, assistant_text):
    if isinstance(operations, list) and len(operations) > 2:
        raise ValueError("continuity_operations must contain at most 2 items")
    return _normalize_operations(
        operations,
        "continuity_operations",
        "continuity",
        [str(user_message or ""), str(assistant_text or "")],
        content_limit=160,
    )


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
        "continuity_operations": _normalize_continuity_operations(
            result["continuity_operations"] if "continuity_operations" in result else None,
            str(user_message or ""),
            str(assistant_text or ""),
        ),
        "self_review": normalized_review,
        "verdict": verdict,
    }
    if "memory_status" in result:
        normalized["memory_status"] = str(result.get("memory_status") or "")
    return normalized


def _extract_review_payload(raw_output):
    if not isinstance(raw_output, str):
        try:
            message = raw_output.choices[0].message
            raw_output = message.content or getattr(message, "reasoning_content", None)
        except (AttributeError, TypeError, IndexError):
            pass
    raw_text = str(raw_output or "").strip()
    matches = list(_RESULT_RE.finditer(raw_text))
    if matches:
        if len(matches) != 1 or raw_text != matches[0].group(0):
            raise ValueError("structured result must be the only non-whitespace content")
        payload_text = matches[0].group(1)
    else:
        payload_text = raw_text
        fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", payload_text, re.DOTALL | re.IGNORECASE)
        if fenced:
            payload_text = fenced.group(1)
    try:
        parsed = json.loads(payload_text)
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
        "角色回复不是用户记忆证据；memory_operations 的 evidence_quote 必须来自 user_message。"
        "continuity_operations 仅允许 target=continuity，每轮最多 2 条，每条内容不超过 160 字；"
        "其 evidence_quote 可来自 user_message 或 assistant_text。普通角色扮演文本、临时动作、气氛、"
        "一次性台词或未经确认的推测必须返回空 continuity_operations。\n"
        "输出必须且只能包含一个结构化 marker。\n"
        "turn_id: {turn_id}\n"
        "user_message:\n{user_message}\n"
        "assistant_text:\n{assistant_text}\n"
        "请填充字段：turn_id, assistant_sha256, style_decision, style_reason, continuity_summary, "
        "memory_operations, continuity_operations, self_review, verdict。"
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


def _memory_store_matches(memory_store, target, text):
    if not _supports_exact_lookup(memory_store):
        return []
    if hasattr(memory_store, "_entries_for"):
        try:
            return [entry for entry in list(memory_store._entries_for(target)) if text in entry]
        except Exception:
            return []
    if _memory_store_has_exact(memory_store, target, text):
        return [text]
    return []


def _operation_already_applied(memory_store, operation):
    if not _supports_exact_lookup(memory_store):
        return False
    target = operation["target"]
    action = operation["action"]
    if action == "add":
        return _memory_store_has_exact(memory_store, target, operation["content"])
    if action == "replace":
        old_matches = _memory_store_matches(memory_store, target, operation["old_text"])
        has_new = _memory_store_has_exact(memory_store, target, operation["content"])
        return has_new and not old_matches
    return not _memory_store_matches(memory_store, target, operation["old_text"])


def _memory_call_succeeded(result):
    return not isinstance(result, dict) or result.get("success", True) is not False


def _execute_memory_operation(memory_store, operation):
    action = operation["action"]
    if action == "add":
        return memory_store.add(operation["target"], operation["content"])
    if action == "replace":
        if _supports_exact_lookup(memory_store):
            old_matches = _memory_store_matches(memory_store, operation["target"], operation["old_text"])
            has_new = _memory_store_has_exact(memory_store, operation["target"], operation["content"])
            if not old_matches and has_new:
                return {"operation": "replace", "status": "already_applied", "content": operation["content"]}
            if not old_matches and not has_new:
                raise RuntimeError("replace target missing")
        return memory_store.replace(operation["target"], operation["old_text"], operation["content"])
    return memory_store.remove(operation["target"], operation["old_text"])


def _record_memory_failure(turn_store, turn_id, assistant_hash, row, result, attempts):
    status = "abandoned" if attempts >= _MAX_MEMORY_OPERATION_ATTEMPTS else "failed"
    retry_after = ""
    turn_store.update_memory_operation(
        turn_id,
        assistant_hash,
        row["operation_index"],
        status,
        result,
        attempts=attempts,
        retry_after_at=retry_after,
    )


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
        if row["status"] in ("applied", "abandoned"):
            continue
        attempts = int(row.get("attempts") or 0)
        if attempts >= _MAX_MEMORY_OPERATION_ATTEMPTS:
            continue
        if row.get("retry_after_at") and not _retry_is_due(row["retry_after_at"]):
            continue
        attempts += 1
        turn_store.update_memory_operation(
            turn_id,
            assistant_hash,
            row["operation_index"],
            "pending",
            row.get("result_json") or {},
            attempts=attempts,
            retry_after_at="",
        )
        if _operation_already_applied(memory_store, operation):
            turn_store.update_memory_operation(
                turn_id,
                assistant_hash,
                row["operation_index"],
                "applied",
                {"status": "already_applied"},
                attempts=attempts,
                retry_after_at="",
            )
            continue
        try:
            result = _execute_memory_operation(memory_store, operation)
        except Exception as exc:
            _record_memory_failure(
                turn_store,
                turn_id,
                assistant_hash,
                row,
                {"error": str(exc)},
                attempts,
            )
            continue
        if not _memory_call_succeeded(result):
            _record_memory_failure(
                turn_store,
                turn_id,
                assistant_hash,
                row,
                result,
                attempts,
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
            attempts=attempts,
            retry_after_at="",
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
    try:
        claim_status, current = turn_store.claim_review(turn_id, assistant_hash)
    except StaleTurnReviewError:
        return {
            "turn_id": str(turn_id),
            "review_status": "stale",
            "memory_status": "pending",
            "memory_modifications": [],
        }
    if claim_status != "claimed":
        existing = current.get("review_json") if isinstance(current.get("review_json"), dict) else {}
        result = {
            "turn_id": str(turn_id),
            "review_status": current.get("status") or "pending",
            "memory_status": current.get("memory_status") or "none",
            "memory_modifications": [],
            "review_result": existing,
        }
        if current.get("status") == "invalid" and isinstance(existing, dict):
            result["error"] = str(existing.get("error") or current.get("style_reason") or "")
        return result
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
            guarded.append(cloned)
            continue
        record = store.get(turn_id)
        if record is None or record.get("status") in ("pending", "invalid"):
            guarded.append(cloned)
            continue
        if record.get("assistant_sha256") != assistant_sha256(cloned.get("content")):
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
