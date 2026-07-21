import hashlib
import json
import os
import re
import uuid
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


class InvalidEvolutionResult(ValueError):
    pass


class StaleEvolutionError(RuntimeError):
    pass


class EvolutionPolicyError(RuntimeError):
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


def _write_pending_audit(profile_dir, original, updated, result, model, policy, action="evolution", source_audit_id=None):
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


def apply_guarded_result(hermes_home, result, model, policy=GUARDED_POLICY):
    _validate_result_shape(result)
    if policy not in _SUPPORTED_POLICIES:
        raise EvolutionPolicyError("unsupported evolution policy")
    profile_id = result["profile_id"]
    profile_dir = _resolve_profile_dir(hermes_home, profile_id)
    if read_evolution_policy(profile_dir) != policy:
        raise EvolutionPolicyError("profile is not {0}".format(policy))
    with profile_lock(profile_dir, "soul"):
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
        audit_path = _write_pending_audit(profile_dir, original, updated, result, model, policy)
        try:
            _atomic_write_text(soul_path, updated)
            _mark_audit_committed(audit_path)
        except Exception:
            _atomic_write_text(soul_path, original)
            raise
        return {
            "profile_id": profile_id,
            "status": "committed",
            "audit_id": audit_path.stem,
        }


def apply_guarded_results(hermes_home, output, model, policy=GUARDED_POLICY):
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
        try:
            responses.append(apply_guarded_result(hermes_home, result, model, policy=policy))
        except StaleEvolutionError as exc:
            responses.append({"profile_id": profile_id, "status": "stale", "error": str(exc)})
        except EvolutionPolicyError as exc:
            responses.append({"profile_id": profile_id, "status": "rejected", "error": str(exc)})
        except InvalidEvolutionResult as exc:
            responses.append({"profile_id": profile_id, "status": "invalid", "error": str(exc)})
    return responses


def rollback_guarded_evolution(hermes_home, profile_id, audit_id):
    profile_dir = _resolve_profile_dir(hermes_home, profile_id)
    if read_evolution_policy(profile_dir) != GUARDED_POLICY:
        raise EvolutionPolicyError("profile is not guarded_v1")
    if not _AUDIT_ID_RE.match(str(audit_id or "")):
        raise InvalidEvolutionResult("invalid audit_id")
    audit_path = profile_dir / "evolution_audit" / (audit_id + ".json")
    with profile_lock(profile_dir, "soul"):
        with open(str(audit_path), "r", encoding="utf-8") as handle:
            source_audit = json.load(handle)
        if source_audit.get("status") != "committed":
            raise InvalidEvolutionResult("audit is not committed")
        soul_path = profile_dir / "SOUL.md"
        original = soul_path.read_text(encoding="utf-8")
        if extract_evolved_persona(original) != source_audit.get("after"):
            raise StaleEvolutionError("current persona differs from audit version")
        updated = replace_evolved_persona(original, source_audit.get("before"))
        rollback_result = {
            "reason": "operator rollback to audit {0}".format(audit_id),
            "self_review": {},
        }
        rollback_audit_path = _write_pending_audit(
            profile_dir,
            original,
            updated,
            rollback_result,
            "operator",
            read_evolution_policy(profile_dir),
            action="rollback",
            source_audit_id=audit_id,
        )
        try:
            _atomic_write_text(soul_path, updated)
            _mark_audit_committed(rollback_audit_path)
        except Exception:
            _atomic_write_text(soul_path, original)
            raise
        return {
            "profile_id": profile_id,
            "status": "committed",
            "audit_id": rollback_audit_path.stem,
        }
