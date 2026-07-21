#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Python 3.6 compatible Savana evolution batch extractor.

Business rule:
- Only profiles that were actually chatted on the previous natural day
  in Asia/Shanghai are eligible for daily self-evolution.
- Eligible profiles are processed in batches to avoid one giant prompt.
"""

import hashlib
import json
import os
import re
import sqlite3
import sys
import time
from difflib import SequenceMatcher
from datetime import datetime

from hermes_cli.companion_profile_policy import (
    GUARDED_POLICY,
    GUARDED_V2_POLICY,
    LEGACY_POLICY,
    read_evolution_policy,
)
from hermes_cli.companion_turn_guard import assistant_sha256

DEFAULT_BATCH_SIZE = 10
DEFAULT_MESSAGE_LIMIT = 30
DEFAULT_BATCH_CURSOR = 0
STATE_DIR_NAME = os.path.join("cron", "state", "savana-self-evolution")
RUN_DATE_FORMAT = "%Y-%m-%d"
_SAVANA_BATCH_PROFILES_JSON_PREFIX = "- Evolution Batch Profiles JSON: "


def parse_non_negative_int_env(name, default):
    raw_value = os.environ.get(name)
    if raw_value is None or raw_value == "":
        return default
    try:
        value = int(raw_value)
    except ValueError:
        return default
    if value < 0:
        return default
    return value


def parse_positive_int_env(name, default):
    value = parse_non_negative_int_env(name, default)
    if value <= 0:
        return default
    return value


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def shanghai_now():
    old_tz = os.environ.get("TZ")
    try:
        os.environ["TZ"] = "Asia/Shanghai"
        if hasattr(time, "tzset"):
            time.tzset()
        return time.time(), datetime.fromtimestamp(time.time())
    finally:
        if old_tz is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = old_tz
        if hasattr(time, "tzset"):
            time.tzset()


def current_business_date(now_dt):
    override = os.environ.get("SAVANA_EVOLUTION_TARGET_DATE", "").strip()
    if override:
        return override
    return now_dt.strftime(RUN_DATE_FORMAT)


def previous_business_date(now_dt):
    override = os.environ.get("SAVANA_EVOLUTION_SOURCE_DATE", "").strip()
    if override:
        return override
    return datetime.fromtimestamp(time.mktime(now_dt.timetuple()) - 24 * 3600).strftime(RUN_DATE_FORMAT)


def business_day_bounds(date_str):
    start = datetime.strptime(date_str + " 00:00:00", "%Y-%m-%d %H:%M:%S")
    end = datetime.strptime(date_str + " 23:59:59", "%Y-%m-%d %H:%M:%S")
    return time.mktime(start.timetuple()), time.mktime(end.timetuple())


def parse_character_name(soul_path):
    if not os.path.exists(soul_path):
        return "Unknown"
    try:
        with open(soul_path, "r", encoding="utf-8") as f:
            first_line = f.readline().strip()
            if first_line.startswith("#"):
                return first_line.lstrip("#").strip()
    except Exception:
        pass
    return "Unknown"


def parse_intimacy_score(soul_path):
    if not os.path.exists(soul_path):
        return 0
    try:
        with open(soul_path, "r", encoding="utf-8") as f:
            for line in f:
                if "Intimacy level:" in line:
                    parts = line.split("Intimacy level:")
                    if len(parts) > 1:
                        return int(parts[1].strip().split("/")[0])
    except Exception:
        pass
    return 0


def extract_evolved_persona(soul_path):
    if not os.path.exists(soul_path):
        return ""
    try:
        with open(soul_path, "r", encoding="utf-8") as f:
            content = f.read()
        marker = "## Evolved Persona"
        idx = content.find(marker)
        if idx == -1:
            return ""
        tail = content[idx + len(marker):].strip()
        next_header = tail.find("\n## ")
        if next_header != -1:
            tail = tail[:next_header].strip()
        return tail
    except Exception:
        return ""


def extract_base_soul(soul_path):
    if not os.path.exists(soul_path):
        return ""
    try:
        with open(soul_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception:
        return ""
    marker = "## Evolved Persona"
    marker_index = content.find(marker)
    if marker_index == -1:
        return content.strip()
    body_start = content.find("\n", marker_index)
    if body_start == -1:
        return content[:marker_index].rstrip()
    next_header = content.find("\n## ", body_start + 1)
    if next_header == -1:
        return content[:marker_index].rstrip()
    return (content[:marker_index].rstrip() + "\n\n" + content[next_header + 1:].lstrip()).strip()


def soul_sha256(soul_path):
    digest = hashlib.sha256()
    with open(soul_path, "rb") as f:
        digest.update(f.read())
    return digest.hexdigest()


def strip_invisible_chars(text):
    if not text:
        return text
    # Strip invisible/zero-width chars that might trigger cron threat scanners
    for char in ['\u200b', '\u200c', '\u200d', '\u2060', '\ufeff', '\u202a', '\u202b', '\u202c', '\u202d', '\u202e']:
        text = text.replace(char, '')
    return text


def dedupe_messages(rows):
    messages = []
    last_msg = None
    for platform_message_id, role, content, timestamp in rows:
        cleaned_content = strip_invisible_chars(content)
        if (
            last_msg
            and last_msg["role"] == role
            and last_msg["content"] == cleaned_content
            and abs(timestamp - last_msg["timestamp"]) < 10
        ):
            continue
        msg_obj = {
            "message_id": "" if platform_message_id is None else str(platform_message_id),
            "role": role,
            "content": cleaned_content,
            "timestamp": timestamp,
        }
        messages.append(msg_obj)
        last_msg = msg_obj
    return messages


def get_messages_for_window(db_path, window_start, window_end, limit):
    if not os.path.exists(db_path):
        return []
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
                SELECT platform_message_id, role, content, timestamp
                FROM messages
                WHERE role IN ('user', 'assistant')
                  AND timestamp >= ?
                  AND timestamp <= ?
                ORDER BY timestamp DESC
                LIMIT ?
            """,
            (window_start, window_end, limit),
        )
        rows = cursor.fetchall()
        rows.reverse()
        return dedupe_messages(rows)
    except Exception as exc:
        sys.stderr.write("Error reading DB {}: {}\n".format(db_path, str(exc)))
        return []
    finally:
        if conn:
            conn.close()


def manifest_dir(hermes_home):
    return os.path.join(hermes_home, STATE_DIR_NAME)


def manifest_path(hermes_home, source_date):
    return os.path.join(manifest_dir(hermes_home), source_date + ".json")


def load_manifest(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_manifest(path, manifest):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def profile_sort_key(item):
    return (-item["last_active"], item["profile_id"])


def discover_profiles(hermes_home, source_date, batch_size, message_limit):
    profiles_dir = os.path.join(hermes_home, "profiles")
    if not os.path.exists(profiles_dir):
        return []

    start_ts, end_ts = business_day_bounds(source_date)
    discovered = []
    for name in os.listdir(profiles_dir):
        profile_path = os.path.join(profiles_dir, name)
        if not os.path.isdir(profile_path) or not name.startswith("savana_"):
            continue

        db_path = os.path.join(profile_path, "state.db")
        soul_path = os.path.join(profile_path, "SOUL.md")
        if not os.path.exists(db_path) or not os.path.exists(soul_path):
            continue

        messages = get_messages_for_window(db_path, start_ts, end_ts, message_limit)
        if not messages:
            continue

        last_active = messages[-1]["timestamp"]
        discovered.append({
            "profile_id": name,
            "profile_path": profile_path,
            "soul_path": soul_path,
            "character_name": parse_character_name(soul_path),
            "intimacy": parse_intimacy_score(soul_path),
            "evolved_persona": extract_evolved_persona(soul_path),
            "evolution_policy": read_evolution_policy(profile_path),
            "messages": messages,
            "last_active": last_active,
        })

    discovered.sort(key=profile_sort_key)
    return discovered


def chunk_profiles(profiles, batch_size):
    batches = []
    current = []
    for item in profiles:
        current.append(item["profile_id"])
        if len(current) >= batch_size:
            batches.append(current)
            current = []
    if current:
        batches.append(current)
    return batches


def chunk_profiles_by_policy(profiles, batch_size):
    batches = []
    batch_policies = []
    for policy in (LEGACY_POLICY, GUARDED_POLICY, GUARDED_V2_POLICY):
        policy_profiles = [
            item for item in profiles
            if item.get("evolution_policy", LEGACY_POLICY) == policy
        ]
        for batch in chunk_profiles(policy_profiles, batch_size):
            batches.append(batch)
            batch_policies.append(policy)
    return batches, batch_policies


def build_manifest(hermes_home, source_date, batch_size, message_limit):
    profiles = discover_profiles(hermes_home, source_date, batch_size, message_limit)
    batches, batch_policies = chunk_profiles_by_policy(profiles, batch_size)
    return {
        "source_date": source_date,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "batch_size": batch_size,
        "message_limit": message_limit,
        "cursor": DEFAULT_BATCH_CURSOR,
        "total_profiles": len(profiles),
        "profile_ids": [item["profile_id"] for item in profiles],
        "batches": batches,
        "batch_policies": batch_policies,
    }


def manifest_complete(manifest):
    return manifest.get("cursor", 0) >= len(manifest.get("batches", []))


def load_or_create_manifest(hermes_home, source_date, batch_size, message_limit):
    path = manifest_path(hermes_home, source_date)
    manifest = load_manifest(path)
    if manifest:
        return path, manifest
    manifest = build_manifest(hermes_home, source_date, batch_size, message_limit)
    save_manifest(path, manifest)
    return path, manifest


def advance_manifest_cursor(hermes_home, source_date):
    path = manifest_path(hermes_home, source_date)
    manifest = load_manifest(path)
    if not manifest:
        return None
    manifest["cursor"] = min(
        manifest.get("cursor", 0) + 1,
        len(manifest.get("batches", [])),
    )
    manifest["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    save_manifest(path, manifest)
    return manifest


def eligible_profiles_for_batch(hermes_home, manifest):
    profiles_dir = os.path.join(hermes_home, "profiles")
    profile_map = {}
    for profile_id in manifest.get("profile_ids", []):
        profile_path = os.path.join(profiles_dir, profile_id)
        db_path = os.path.join(profile_path, "state.db")
        soul_path = os.path.join(profile_path, "SOUL.md")
        if not os.path.exists(db_path) or not os.path.exists(soul_path):
            continue
        profile_map[profile_id] = (profile_path, db_path, soul_path)
    return profile_map


def current_batch_policy(manifest):
    cursor = manifest.get("cursor", 0)
    batch_policies = manifest.get("batch_policies", [])
    if cursor < 0 or cursor >= len(batch_policies):
        return LEGACY_POLICY
    return batch_policies[cursor]


def batch_profiles(hermes_home, manifest):
    if manifest_complete(manifest):
        return []
    batch_ids = manifest["batches"][manifest["cursor"]]
    expected_policy = current_batch_policy(manifest)
    start_ts, end_ts = business_day_bounds(manifest["source_date"])
    profile_map = eligible_profiles_for_batch(hermes_home, manifest)
    result = []
    for profile_id in batch_ids:
        info = profile_map.get(profile_id)
        if not info:
            continue
        profile_path, db_path, soul_path = info
        messages = get_messages_for_window(
            db_path,
            start_ts,
            end_ts,
            manifest.get("message_limit", DEFAULT_MESSAGE_LIMIT),
        )
        if not messages:
            continue
        last_active = messages[-1]["timestamp"]
        current_policy = read_evolution_policy(profile_path)
        if current_policy != expected_policy:
            raise RuntimeError(
                "batch policy drift for {0}: expected {1}, got {2}".format(
                    profile_id,
                    expected_policy,
                    current_policy,
                )
            )
        result.append({
            "profile_id": profile_id,
            "profile_path": profile_path,
            "character_name": parse_character_name(soul_path),
            "soul_path": soul_path,
            "intimacy": parse_intimacy_score(soul_path),
            "evolved_persona": extract_evolved_persona(soul_path),
            "evolution_policy": current_policy,
            "base_soul": extract_base_soul(soul_path),
            "soul_sha256": soul_sha256(soul_path),
            "last_active": last_active,
            "messages": messages,
        })
    result.sort(key=profile_sort_key)
    if expected_policy in (GUARDED_POLICY, GUARDED_V2_POLICY) and len(result) != len(batch_ids):
        live_ids = set(item["profile_id"] for item in result)
        missing_ids = [profile_id for profile_id in batch_ids if profile_id not in live_ids]
        raise RuntimeError(
            "live profiles missing from guarded batch: " + ", ".join(missing_ids)
        )
    return result


def display_time(ts):
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def format_message_content(content):
    if "请以星野式沉浸表达回复用户这句话" in content:
        parts = content.split("这句话：")
        if len(parts) > 1:
            raw_user_msg = parts[1].split("\n")[0]
            return "[Injected Template Input] -> " + raw_user_msg.strip()
    return content


_QUALITY_CORRECTION_PATTERNS = (
    re.compile(r"你.{0,6}说话.{0,4}(太)?机械"),
    re.compile(r"你.{0,6}像程序"),
    re.compile(r"(?:\bOOC\b|ooc).{0,4}出戏|出戏.{0,4}(?:\bOOC\b|ooc)"),
    re.compile(r"你.{0,6}(?:出戏|ooc)"),
    re.compile(r"不许把我叫成访客"),
)


def _is_quality_correction_message(content):
    text = str(content or "")
    for pattern in _QUALITY_CORRECTION_PATTERNS:
        if pattern.search(text):
            return True
    return False


def _load_turn_reviews(profile_path, turn_ids):
    db_path = os.path.join(profile_path, "companion_guard.db")
    filtered_turn_ids = [str(turn_id).strip() for turn_id in list(turn_ids or []) if str(turn_id).strip()]
    if not os.path.exists(db_path) or not filtered_turn_ids:
        return None
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        placeholders = ",".join("?" for unused_item in filtered_turn_ids)
        cursor.execute(
            """
                SELECT turn_id, assistant_sha256, status, continuity_summary
                FROM turn_reviews
                WHERE turn_id IN ({0})
            """.format(placeholders),
            filtered_turn_ids,
        )
        reviews = {}
        for turn_id, assistant_hash, status, continuity_summary in cursor.fetchall():
            reviews[str(turn_id or "")] = {
                "assistant_sha256": str(assistant_hash or ""),
                "status": str(status or ""),
                "continuity_summary": str(continuity_summary or ""),
            }
        return reviews
    except Exception:
        return None
    finally:
        if conn:
            conn.close()


def _normalize_summary_text(value):
    return " ".join(str(value or "").split())


def _summary_leaks_raw(summary, raw_text):
    normalized_summary = _normalize_summary_text(summary)
    normalized_raw = _normalize_summary_text(raw_text)
    if normalized_raw in normalized_summary or normalized_summary in normalized_raw:
        return True
    if len(normalized_summary) < 8 or len(normalized_raw) < 8:
        return False
    match = SequenceMatcher(None, normalized_summary, normalized_raw).find_longest_match(
        0,
        len(normalized_summary),
        0,
        len(normalized_raw),
    )
    leaked = normalized_summary[match.a:match.a + match.size].strip()
    return len(leaked) >= 8


def _print_guarded_v2_dialogue(chat):
    print("\n### Dialogue History\n")
    turn_ids = [
        message.get("message_id")
        for message in chat["messages"]
        if message.get("role") == "user"
    ]
    reviews = _load_turn_reviews(chat["profile_path"], turn_ids)
    previous_user = None
    for message in chat["messages"]:
        if message["role"] == "user":
            previous_user = message
            suffix = " [quality_correction_only]" if _is_quality_correction_message(message.get("content")) else ""
            print("[evolution_evidence] USER: {0}{1}".format(
                format_message_content(message.get("content") or ""),
                suffix,
            ))
            print("")
            continue
        if message["role"] != "assistant":
            continue
        turn_id = str((previous_user or {}).get("message_id") or "")
        record = reviews.get(turn_id) if reviews and turn_id else None
        if (
            not record
            or record.get("status") in ("pending", "invalid")
            or record.get("assistant_sha256") != assistant_sha256(message.get("content") or "")
        ):
            print("[context_only] ASSISTANT: [review unavailable]")
            print("")
            continue
        if record.get("status") == "drift":
            if _summary_leaks_raw(record.get("continuity_summary"), message.get("content")):
                print("[context_only] ASSISTANT: [review unavailable]")
                print("")
                continue
            print("[context_only] ASSISTANT SUMMARY: {0}".format(
                record.get("continuity_summary") or "[review unavailable]"
            ))
            print("")
            continue
        print("[context_only] ASSISTANT: {0}".format(
            format_message_content(message.get("content") or "")
        ))
        print("")


def print_manifest_header(now_dt, manifest, current_batch_count, batch_policy):
    print("# Savana Characters Evolution Report")
    print("Generated at: {}\n".format(now_dt.strftime("%Y-%m-%d %H:%M:%S")))
    print("- Review Date: {}".format(manifest["source_date"]))
    print("- Batch Cursor: {}/{}".format(
        manifest.get("cursor", 0) + 1 if manifest.get("batches") else 0,
        len(manifest.get("batches", []))
    ))
    print("- Batch Size: {}".format(manifest.get("batch_size", DEFAULT_BATCH_SIZE)))
    print("- Profiles Included In Batch: {}".format(current_batch_count))
    if batch_policy in (GUARDED_POLICY, GUARDED_V2_POLICY):
        print("- Evolution Batch Policy: {}".format(batch_policy))
        print(
            _SAVANA_BATCH_PROFILES_JSON_PREFIX
            + json.dumps(
                list(manifest["batches"][manifest["cursor"]]),
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )
    print("- Total Eligible Profiles Yesterday: {}".format(manifest.get("total_profiles", 0)))
    print("- Per-Profile Message Limit: {}\n".format(
        manifest.get("message_limit", DEFAULT_MESSAGE_LIMIT)
    ))


def print_profile_section(chat, now_ts):
    last_active_str = display_time(chat["last_active"])
    elapsed = max(0, now_ts - chat["last_active"])
    days = int(elapsed // (24 * 3600))
    hours = int((elapsed % (24 * 3600)) // 3600)
    elapsed_str = "{}天 {}小时".format(days, hours)
    print("## Character: {} (Profile ID: {})".format(
        chat["character_name"],
        chat["profile_id"]
    ))
    print("- SOUL.md File Path: {}".format(chat["soul_path"]))
    print("- Last Active Time: {}".format(last_active_str))
    print("- Time Elapsed Since Last Message: {}".format(elapsed_str))
    print("- Current Intimacy Level: {}/10".format(chat["intimacy"]))
    print("- Current Evolved Persona Section:")
    print("  ```markdown")
    print("  " + (chat["evolved_persona"] or "(暂无自主进化，人设遵循基础设定)"))
    print("  ```")
    if chat.get("evolution_policy") in (GUARDED_POLICY, GUARDED_V2_POLICY):
        print("- SOUL.md SHA-256: {}".format(chat["soul_sha256"]))
        print("\n### Base Persona Snapshot\n")
        print("```markdown")
        print(chat.get("base_soul") or "")
        print("```")
    if chat.get("evolution_policy") == GUARDED_V2_POLICY:
        _print_guarded_v2_dialogue(chat)
    else:
        print("\n### Dialogue History\n")
        for msg in chat["messages"]:
            role_display = "User" if msg["role"] == "user" else chat["character_name"]
            print("[{} ({})]: {}".format(
                role_display,
                display_time(msg["timestamp"]),
                format_message_content(msg["content"] or "")
            ))
            print("")
    print("---\n")


def main():
    hermes_home = os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes"))
    now_ts, now_dt = shanghai_now()
    batch_size = parse_positive_int_env(
        "SAVANA_EVOLUTION_BATCH_SIZE",
        DEFAULT_BATCH_SIZE
    )
    message_limit = parse_positive_int_env(
        "SAVANA_EVOLUTION_MESSAGE_LIMIT",
        DEFAULT_MESSAGE_LIMIT
    )
    source_date = previous_business_date(now_dt)
    manifest_path_value, manifest = load_or_create_manifest(
        hermes_home,
        source_date,
        batch_size,
        message_limit
    )

    if manifest_complete(manifest):
        print("# Savana Characters Evolution Report")
        print("Generated at: {}\n".format(now_dt.strftime("%Y-%m-%d %H:%M:%S")))
        print("- Review Date: {}".format(manifest["source_date"]))
        print("- Batch Cursor: {}/{}".format(
            manifest.get("cursor", 0),
            len(manifest.get("batches", []))
        ))
        print("- Manifest Path: {}".format(manifest_path_value))
        print("\nNo pending Savana evolution batches remain for the review date.")
        return

    try:
        profiles = batch_profiles(hermes_home, manifest)
    except RuntimeError as exc:
        sys.stderr.write(str(exc) + "\n")
        raise
    if not profiles:
        print("# Savana Characters Evolution Report")
        print("Generated at: {}\n".format(now_dt.strftime("%Y-%m-%d %H:%M:%S")))
        print("- Review Date: {}".format(manifest["source_date"]))
        print("- Manifest Path: {}".format(manifest_path_value))
        print("\nCurrent batch resolved to zero live profiles. Advance cursor manually if needed.")
        return

    batch_policy = current_batch_policy(manifest)
    print_manifest_header(now_dt, manifest, len(profiles), batch_policy)
    for chat in profiles:
        print_profile_section(chat, now_ts)


if __name__ == "__main__":
    main()
