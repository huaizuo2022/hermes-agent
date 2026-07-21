import hashlib
import json
import os
import re
import uuid
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path

from hermes_cli.companion_profile_policy import (
    GUARDED_POLICY,
    GUARDED_V2_POLICY,
    profile_lock,
    read_evolution_policy,
)


RESULT_START = "<!-- GUARDED_EVOLUTION_RESULT "
RESULT_END = " GUARDED_EVOLUTION_RESULT -->"
REQUIRED_REVIEW_KEYS = (
    "necessary",
    "preserves_identity",
    "no_unfounded_jump",
    "no_error_solidification",
    "no_base_override",
)
_PROFILE_ID_RE = re.compile(r"^savana_[A-Za-z0-9_.-]+$")
_AUDIT_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_RESULT_RE = re.compile(
    re.escape(RESULT_START) + r"(.*?)" + re.escape(RESULT_END),
    re.DOTALL,
)
_SUPPORTED_POLICIES = frozenset([GUARDED_POLICY, GUARDED_V2_POLICY])
_RECOVERY_DIR_NAME = "savana_evolution_recovery"
_RECOVERY_LOCK_PURPOSE = "savana-evolution-recovery-transaction"


class InvalidEvolutionResult(ValueError):
    pass


class StaleEvolutionError(RuntimeError):
    pass


class EvolutionPolicyError(RuntimeError):
    pass


class BatchRecoveryError(RuntimeError):
    pass


def _section_bounds(soul_text):
    marker = re.search(r"^## Evolved Persona[^\n]*(?:\n|$)", soul_text, re.MULTILINE)
    if marker is None:
        raise InvalidEvolutionResult("SOUL.md is missing Evolved Persona")
    body_start = marker.end()
    next_header = re.search(r"^#", soul_text[body_start:], re.MULTILINE)
    body_end = body_start + next_header.start() if next_header else len(soul_text)
    return marker.start(), body_start, body_end


def extract_evolved_persona(soul_text):
    unused_marker_start, body_start, body_end = _section_bounds(soul_text)
    return soul_text[body_start:body_end].strip()


def strip_evolved_persona(soul_text):
    marker_start, unused_body_start, body_end = _section_bounds(soul_text)
    return soul_text[:marker_start] + soul_text[body_end:]


def replace_evolved_persona(soul_text, candidate):
    candidate = str(candidate or "").strip()
    if not candidate:
        raise InvalidEvolutionResult("candidate_evolved_persona is empty")
    if any(line.lstrip().startswith("#") for line in candidate.splitlines()):
        raise InvalidEvolutionResult("candidate_evolved_persona contains a heading")
    unused_marker_start, body_start, body_end = _section_bounds(soul_text)
    suffix = soul_text[body_end:]
    separator = "\n\n" if suffix else "\n"
    return soul_text[:body_start] + candidate + separator + suffix


def _sha256_text(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _atomic_write_text(path, content):
    path = Path(path)
    temp_path = path.with_name(".{0}.{1}.tmp".format(path.name, uuid.uuid4().hex))
    try:
        with open(str(temp_path), "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temp_path), str(path))
    finally:
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass


def _validate_result_shape(result):
    if not isinstance(result, dict):
        raise InvalidEvolutionResult("result must be an object")
    profile_id = str(result.get("profile_id") or "")
    if not _PROFILE_ID_RE.match(profile_id):
        raise InvalidEvolutionResult("invalid profile_id")
    expected_hash = str(result.get("expected_soul_sha256") or "")
    if not re.match(r"^[0-9a-f]{64}$", expected_hash):
        raise InvalidEvolutionResult("invalid expected_soul_sha256")
    if result.get("decision") not in ("no_change", "evolve"):
        raise InvalidEvolutionResult("invalid decision")
    if not str(result.get("reason") or "").strip():
        raise InvalidEvolutionResult("reason is required")
    review = result.get("self_review")
    if not isinstance(review, dict):
        raise InvalidEvolutionResult("self_review is required")
    for key in REQUIRED_REVIEW_KEYS:
        if review.get(key) not in ("pass", "reject"):
            raise InvalidEvolutionResult("invalid self_review.{0}".format(key))
    if result.get("verdict") not in ("pass", "reject"):
        raise InvalidEvolutionResult("invalid verdict")
    if result.get("decision") == "evolve":
        replace_evolved_persona(
            "## Evolved Persona\ncurrent\n",
            result.get("candidate_evolved_persona"),
        )


def _resolve_profile_dir(hermes_home, profile_id):
    if not _PROFILE_ID_RE.match(str(profile_id or "")):
        raise InvalidEvolutionResult("invalid profile_id")
    profiles_root = (Path(hermes_home) / "profiles").resolve()
    profile_dir = (profiles_root / profile_id).resolve()
    if profile_dir.parent != profiles_root:
        raise InvalidEvolutionResult("profile path escapes profiles root")
    return profile_dir


def _write_pending_audit(
    profile_dir,
    original,
    updated,
    result,
    model,
    policy,
    action="evolution",
    source_audit_id=None,
    batch_id=None,
):
    audit_dir = profile_dir / "evolution_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    audit_id = datetime.utcnow().strftime("%Y%m%dT%H%M%S%fZ-") + uuid.uuid4().hex
    audit = {
        "id": audit_id,
        "profile_id": profile_dir.name,
        "action": action,
        "status": "pending",
        "created_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "policy": policy,
        "model": str(model or ""),
        "reason": str(result.get("reason") or ""),
        "self_review": result.get("self_review") or {},
        "before": extract_evolved_persona(original),
        "after": extract_evolved_persona(updated),
    }
    if source_audit_id:
        audit["source_audit_id"] = source_audit_id
    if batch_id:
        audit["batch_id"] = batch_id
    audit_path = audit_dir / (audit_id + ".json")
    _atomic_write_text(
        audit_path,
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return audit_path


def _mark_audit_committed(audit_path):
    with open(str(audit_path), "r", encoding="utf-8") as handle:
        audit = json.load(handle)
    audit["status"] = "committed"
    audit["committed_at"] = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    _atomic_write_text(
        audit_path,
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _review_passes(result):
    review = result["self_review"]
    return (
        result["verdict"] == "pass"
        and all(review[key] == "pass" for key in REQUIRED_REVIEW_KEYS)
    )


def _recovery_dir(hermes_home):
    return Path(hermes_home) / _RECOVERY_DIR_NAME


def _recovery_transaction_lock(hermes_home):
    lock_target = Path(hermes_home).resolve() / ".savana-evolution-recovery"
    return profile_lock(lock_target, _RECOVERY_LOCK_PURPOSE)


def _pending_recovery_journals(hermes_home):
    recovery_dir = _recovery_dir(hermes_home)
    if not recovery_dir.exists():
        return []
    return sorted(path for path in recovery_dir.glob("*.json") if path.is_file())


def _require_no_pending_recovery(hermes_home):
    for journal_path in _pending_recovery_journals(hermes_home):
        try:
            journal = json.loads(journal_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise BatchRecoveryError(
                "unreadable Savana evolution recovery journal blocks writes: {0} ({1})".format(
                    journal_path,
                    exc,
                )
            )
        if journal.get("status") == "committed":
            try:
                journal_path.unlink()
            except Exception as exc:
                raise BatchRecoveryError(
                    "completed Savana evolution journal could not be cleaned: {0} ({1})".format(
                        journal_path,
                        exc,
                    )
                )
            continue
        raise BatchRecoveryError(
            "unresolved Savana evolution recovery journal blocks writes: {0}".format(
                journal_path
            )
        )


def _write_recovery_journal(journal_path, journal):
    _atomic_write_text(
        journal_path,
        json.dumps(journal, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _create_recovery_journal(hermes_home, prepared, model, policy, action="evolution"):
    recovery_dir = _recovery_dir(hermes_home)
    recovery_dir.mkdir(parents=True, exist_ok=True)
    batch_id = datetime.utcnow().strftime("%Y%m%dT%H%M%S%fZ-") + uuid.uuid4().hex
    journal_path = recovery_dir / (batch_id + ".json")
    journal = {
        "id": batch_id,
        "status": "committing",
        "created_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "policy": policy,
        "model": str(model or ""),
        "action": action,
        "entries": [],
    }
    for item in prepared:
        if item["status"] != "committable":
            continue
        journal["entries"].append({
            "profile_id": item["profile_id"],
            "profile_dir": str(item["profile_dir"]),
            "soul_path": str(item["soul_path"]),
            "original_sha256": _sha256_text(item["original"]),
            "candidate_sha256": _sha256_text(item["updated"]),
            "soul_state": "original",
            "audit_path": "",
            "audit_state": "not_created",
        })
    _write_recovery_journal(journal_path, journal)
    return journal_path, journal


def _journal_entry(journal, profile_id):
    for entry in journal["entries"]:
        if entry["profile_id"] == profile_id:
            return entry
    raise KeyError(profile_id)


def _record_recovery_journal(journal_path, journal, recovery_errors):
    try:
        _write_recovery_journal(journal_path, journal)
    except Exception as exc:
        recovery_errors.append("journal update failed: {0}".format(exc))


def _raise_recovery_error(original_error, recovery_errors, journal_path):
    message = (
        "Savana evolution batch recovery incomplete after {0}: {1}; journal: {2}"
        .format(
            original_error,
            "; ".join(recovery_errors),
            journal_path,
        )
    )
    raise BatchRecoveryError(message) from original_error


def _compensate_guarded_batch(
    journal_path,
    journal,
    prepared,
    written_profile_ids,
    original_error,
):
    recovery_errors = []
    journal["status"] = "recovering"
    journal["original_error"] = repr(original_error)
    _record_recovery_journal(journal_path, journal, recovery_errors)

    for item in reversed(prepared):
        if item["status"] != "committable":
            continue
        entry = _journal_entry(journal, item["profile_id"])
        if item["profile_id"] not in written_profile_ids:
            entry["soul_state"] = "original"
            continue
        try:
            _atomic_write_text(item["soul_path"], item["original"])
            entry["soul_state"] = "restored"
        except Exception as exc:
            entry["soul_state"] = "rollback_failed"
            entry["rollback_error"] = repr(exc)
            recovery_errors.append(
                "{0} SOUL rollback failed: {1}".format(item["profile_id"], exc)
            )
        _record_recovery_journal(journal_path, journal, recovery_errors)

    for entry in journal["entries"]:
        audit_path_value = entry.get("audit_path")
        if not audit_path_value:
            continue
        audit_path = Path(audit_path_value)
        if entry["soul_state"] == "rollback_failed":
            entry["audit_state"] = "retained_for_recovery"
            continue
        try:
            audit_path.unlink()
            entry["audit_state"] = "cleaned"
        except FileNotFoundError:
            entry["audit_state"] = "cleaned"
        except Exception as exc:
            entry["audit_state"] = "cleanup_failed"
            entry["audit_cleanup_error"] = repr(exc)
            recovery_errors.append(
                "{0} audit cleanup failed: {1}".format(entry["profile_id"], exc)
            )
        _record_recovery_journal(journal_path, journal, recovery_errors)

    if recovery_errors:
        journal["status"] = "recovery_required"
        journal["recovery_errors"] = list(recovery_errors)
        _record_recovery_journal(journal_path, journal, recovery_errors)
        _raise_recovery_error(original_error, recovery_errors, journal_path)

    journal["status"] = "recovered"
    _write_recovery_journal(journal_path, journal)
    try:
        journal_path.unlink()
    except Exception as exc:
        journal["status"] = "recovery_required"
        journal["recovery_errors"] = ["recovery journal cleanup failed: {0}".format(exc)]
        try:
            _write_recovery_journal(journal_path, journal)
        except Exception:
            pass
        _raise_recovery_error(original_error, journal["recovery_errors"], journal_path)


def apply_guarded_result(hermes_home, result, model, policy=GUARDED_POLICY):
    _validate_result_shape(result)
    if policy not in _SUPPORTED_POLICIES:
        raise EvolutionPolicyError("unsupported evolution policy")
    profile_id = result["profile_id"]
    output = "{0}{1}{2}".format(
        RESULT_START,
        json.dumps(result, ensure_ascii=False, separators=(",", ":")),
        RESULT_END,
    )
    responses = apply_guarded_results(
        hermes_home,
        output,
        model,
        policy=policy,
        expected_profile_ids=[profile_id],
    )
    if len(responses) != 1:
        raise InvalidEvolutionResult("single guarded result produced an invalid response set")
    response = responses[0]
    status = response.get("status")
    error = str(response.get("error") or "")
    if status == "stale":
        raise StaleEvolutionError(error or "SOUL.md changed after analysis")
    if status == "invalid":
        raise InvalidEvolutionResult(error or "invalid guarded evolution result")
    if status == "rejected" and error:
        raise EvolutionPolicyError(error)
    return response


def _prepare_guarded_result(profile_dir, result, policy):
    _validate_result_shape(result)
    if policy not in _SUPPORTED_POLICIES:
        raise EvolutionPolicyError("unsupported evolution policy")
    profile_id = result["profile_id"]
    if read_evolution_policy(profile_dir) != policy:
        raise EvolutionPolicyError("profile is not {0}".format(policy))
    soul_path = profile_dir / "SOUL.md"
    original = soul_path.read_text(encoding="utf-8")
    if _sha256_text(original) != result["expected_soul_sha256"]:
        raise StaleEvolutionError("SOUL.md changed after analysis")
    if result["decision"] == "no_change":
        return {"profile_id": profile_id, "status": "no_change"}
    if not _review_passes(result):
        return {"profile_id": profile_id, "status": "rejected"}
    updated = replace_evolved_persona(
        original,
        result["candidate_evolved_persona"],
    )
    if strip_evolved_persona(updated) != strip_evolved_persona(original):
        raise InvalidEvolutionResult("candidate changed base persona")
    return {
        "profile_id": profile_id,
        "status": "committable",
        "profile_dir": profile_dir,
        "soul_path": soul_path,
        "original": original,
        "updated": updated,
        "result": result,
    }


def _commit_prepared_batch(
    hermes_home,
    prepared,
    model,
    policy,
    action="evolution",
    source_audit_ids=None,
):
    committable = [item for item in prepared if item["status"] == "committable"]
    if not committable:
        return [
            {"profile_id": item["profile_id"], "status": item["status"]}
            for item in prepared
        ]
    journal_path, journal = _create_recovery_journal(
        hermes_home,
        prepared,
        model,
        policy,
        action=action,
    )
    responses = []
    written_profile_ids = set()
    source_audit_ids = source_audit_ids or {}
    try:
        for item in prepared:
            if item["status"] != "committable":
                responses.append({"profile_id": item["profile_id"], "status": item["status"]})
                continue
            current_policy = read_evolution_policy(item["profile_dir"])
            if current_policy != policy:
                raise EvolutionPolicyError("profile is not {0}".format(policy))
            current = item["soul_path"].read_text(encoding="utf-8")
            if _sha256_text(current) != item["result"]["expected_soul_sha256"]:
                raise StaleEvolutionError("SOUL.md changed after analysis")
            audit_path = _write_pending_audit(
                item["profile_dir"],
                item["original"],
                item["updated"],
                item["result"],
                model,
                policy,
                action=action,
                source_audit_id=source_audit_ids.get(item["profile_id"]),
                batch_id=journal["id"],
            )
            entry = _journal_entry(journal, item["profile_id"])
            entry["audit_path"] = str(audit_path)
            entry["audit_state"] = "pending"
            _write_recovery_journal(journal_path, journal)
            _atomic_write_text(item["soul_path"], item["updated"])
            written_profile_ids.add(item["profile_id"])
            entry["soul_state"] = "candidate"
            _write_recovery_journal(journal_path, journal)
            _mark_audit_committed(audit_path)
            entry["audit_state"] = "committed"
            _write_recovery_journal(journal_path, journal)
            responses.append({
                "profile_id": item["profile_id"],
                "status": "committed",
                "audit_id": audit_path.stem,
            })
        journal["status"] = "committed"
        _write_recovery_journal(journal_path, journal)
        try:
            journal_path.unlink()
        except Exception as exc:
            raise BatchRecoveryError(
                "committed Savana evolution journal cleanup failed: {0} ({1})".format(
                    journal_path,
                    exc,
                )
            )
        return responses
    except BatchRecoveryError:
        raise
    except Exception as exc:
        _compensate_guarded_batch(
            journal_path,
            journal,
            prepared,
            written_profile_ids,
            exc,
        )
        raise


def _apply_guarded_results_locked(
    hermes_home,
    output,
    model,
    policy=GUARDED_POLICY,
    expected_profile_ids=None,
):
    matches = list(_RESULT_RE.finditer(str(output or "")))
    parsed = []
    responses = []
    for match in matches:
        try:
            result = json.loads(match.group(1))
            _validate_result_shape(result)
            parsed.append(result)
        except Exception as exc:
            responses.append({"profile_id": "", "status": "invalid", "error": str(exc)})

    counts = {}
    for result in parsed:
        profile_id = result["profile_id"]
        counts[profile_id] = counts.get(profile_id, 0) + 1

    expected = None
    if expected_profile_ids is not None:
        expected = [str(item).strip() for item in list(expected_profile_ids)]
        if not expected:
            return [{
                "profile_id": "",
                "status": "invalid",
                "error": "expected_profile_ids must not be empty",
            }]
        expected_seen = set()
        for profile_id in expected:
            if not _PROFILE_ID_RE.match(profile_id):
                return [{
                    "profile_id": profile_id,
                    "status": "invalid",
                    "error": "invalid expected profile id",
                }]
            if profile_id in expected_seen:
                return [{
                    "profile_id": profile_id,
                    "status": "invalid",
                    "error": "duplicate expected profile id",
                }]
            expected_seen.add(profile_id)
    for result in parsed:
        profile_id = result["profile_id"]
        if counts[profile_id] > 1:
            if not any(item.get("profile_id") == profile_id for item in responses):
                responses.append({
                    "profile_id": profile_id,
                    "status": "invalid",
                    "error": "duplicate profile result",
                })
            continue
        if expected is not None and profile_id not in expected:
            responses.append({
                "profile_id": profile_id,
                "status": "invalid",
                "error": "unexpected profile result",
            })
    observed_profile_ids = set(item.get("profile_id") for item in responses if item.get("profile_id"))
    if expected is not None:
        for profile_id in expected:
            if profile_id not in counts:
                responses.append({
                    "profile_id": profile_id,
                    "status": "invalid",
                    "error": "missing structured result",
                })
    if responses:
        return responses

    parsed_by_id = {}
    for result in parsed:
        parsed_by_id[result["profile_id"]] = result
    profile_dirs = {}
    canonical_owners = {}
    for profile_id in parsed_by_id:
        try:
            profile_dir = _resolve_profile_dir(hermes_home, profile_id)
        except InvalidEvolutionResult as exc:
            return [{"profile_id": profile_id, "status": "invalid", "error": str(exc)}]
        canonical_path = str(profile_dir)
        if canonical_path in canonical_owners:
            return [{
                "profile_id": profile_id,
                "status": "invalid",
                "error": "profile ids resolve to the same canonical directory",
            }]
        canonical_owners[canonical_path] = profile_id
        profile_dirs[profile_id] = profile_dir
    ordered_profile_ids = sorted(parsed_by_id, key=lambda item: str(profile_dirs[item]))

    with ExitStack() as stack:
        for profile_id in ordered_profile_ids:
            stack.enter_context(profile_lock(profile_dirs[profile_id], "soul"))

        prepared = []
        preflight = []
        for profile_id in ordered_profile_ids:
            result = parsed_by_id[profile_id]
            try:
                profile_dir = profile_dirs[profile_id]
                current_policy = read_evolution_policy(profile_dir)
                if current_policy != policy:
                    raise EvolutionPolicyError("profile is not {0}".format(policy))
                prepared_result = _prepare_guarded_result(profile_dir, result, policy)
                prepared.append(prepared_result)
                preflight.append({"profile_id": profile_id, "status": prepared_result["status"]})
            except StaleEvolutionError as exc:
                preflight.append({"profile_id": profile_id, "status": "stale", "error": str(exc)})
            except EvolutionPolicyError as exc:
                preflight.append({"profile_id": profile_id, "status": "rejected", "error": str(exc)})
            except InvalidEvolutionResult as exc:
                preflight.append({"profile_id": profile_id, "status": "invalid", "error": str(exc)})

        if any(item.get("status") not in ("no_change", "committable") for item in preflight):
            return preflight

        return _commit_prepared_batch(
            hermes_home,
            prepared,
            model,
            policy,
        )


def apply_guarded_results(hermes_home, output, model, policy=GUARDED_POLICY, expected_profile_ids=None):
    with _recovery_transaction_lock(hermes_home):
        _require_no_pending_recovery(hermes_home)
        return _apply_guarded_results_locked(
            hermes_home,
            output,
            model,
            policy=policy,
            expected_profile_ids=expected_profile_ids,
        )


def rollback_guarded_evolution(hermes_home, profile_id, audit_id):
    profile_dir = _resolve_profile_dir(hermes_home, profile_id)
    if not _AUDIT_ID_RE.match(str(audit_id or "")):
        raise InvalidEvolutionResult("invalid audit_id")
    audit_path = profile_dir / "evolution_audit" / (audit_id + ".json")
    with _recovery_transaction_lock(hermes_home):
        with profile_lock(profile_dir, "soul"):
            policy = read_evolution_policy(profile_dir)
            if policy not in _SUPPORTED_POLICIES:
                raise EvolutionPolicyError("profile is not guarded")
            with open(str(audit_path), "r", encoding="utf-8") as handle:
                source_audit = json.load(handle)
            if source_audit.get("status") != "committed":
                raise InvalidEvolutionResult("audit is not committed")
            source_policy = source_audit.get("policy")
            if source_policy is None and policy == GUARDED_POLICY:
                source_policy = GUARDED_POLICY
            if source_policy != policy:
                raise EvolutionPolicyError("audit policy does not match current profile policy")
            soul_path = profile_dir / "SOUL.md"
            original = soul_path.read_text(encoding="utf-8")
            if extract_evolved_persona(original) != source_audit.get("after"):
                raise StaleEvolutionError("current persona differs from audit version")
            updated = replace_evolved_persona(original, source_audit.get("before"))
            rollback_result = {
                "expected_soul_sha256": _sha256_text(original),
                "reason": "operator rollback to audit {0}".format(audit_id),
                "self_review": {},
            }
            prepared = [{
                "profile_id": profile_id,
                "status": "committable",
                "profile_dir": profile_dir,
                "soul_path": soul_path,
                "original": original,
                "updated": updated,
                "result": rollback_result,
            }]
            responses = _commit_prepared_batch(
                hermes_home,
                prepared,
                "operator",
                policy,
                action="rollback",
                source_audit_ids={profile_id: audit_id},
            )
            return responses[0]
