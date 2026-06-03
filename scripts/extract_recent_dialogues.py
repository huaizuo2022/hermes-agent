#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Python 3.6 compatible script to extract recent chat dialogues and cold sessions from Hermes profiles
for daily review and self-evolution.
"""

import os
import sys
import time
import sqlite3
from datetime import datetime

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
                        val = parts[1].strip().split("/")[0]
                        return int(val)
    except Exception:
        pass
    return 0

def extract_evolved_persona(soul_path):
    if not os.path.exists(soul_path):
        return ""
    try:
        with open(soul_path, "r", encoding="utf-8") as f:
            content = f.read()
        header = "## Evolved Persona"
        idx = content.find(header)
        if idx == -1:
            return ""
        section_content = content[idx + len(header):].strip()
        next_header_idx = section_content.find("\n## ")
        if next_header_idx != -1:
            section_content = section_content[:next_header_idx].strip()
        return section_content
    except Exception:
        return ""

def get_recent_messages(db_path, since_timestamp=0):
    if not os.path.exists(db_path):
        return []
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        # If since_timestamp is 0, query last 30 messages overall to identify state
        if since_timestamp == 0:
            query = """
                SELECT role, content, timestamp 
                FROM messages 
                WHERE role IN ('user', 'assistant')
                ORDER BY timestamp DESC
                LIMIT 30
            """
            cursor.execute(query)
            rows = cursor.fetchall()
            rows.reverse()  # Restore chronological order
        else:
            query = """
                SELECT role, content, timestamp 
                FROM messages 
                WHERE timestamp >= ? AND role IN ('user', 'assistant')
                ORDER BY timestamp ASC
            """
            cursor.execute(query, (since_timestamp,))
            rows = cursor.fetchall()
            
        messages = []
        last_msg = None
        for r in rows:
            role, content, timestamp = r[0], r[1], r[2]
            if last_msg and last_msg["role"] == role and last_msg["content"] == content and abs(timestamp - last_msg["timestamp"]) < 10:
                continue
            msg_obj = {
                "role": role,
                "content": content,
                "timestamp": timestamp
            }
            messages.append(msg_obj)
            last_msg = msg_obj
        return messages
    except Exception as e:
        sys.stderr.write("Error reading DB {}: {}\n".format(db_path, str(e)))
        return []
    finally:
        if conn:
            conn.close()

def main():
    hermes_home = os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes"))
    profiles_dir = os.path.join(hermes_home, "profiles")
    
    if not os.path.exists(profiles_dir):
        print("No profiles directory found at: {}".format(profiles_dir))
        return

    now = time.time()
    all_profile_chats = []
    
    for name in os.listdir(profiles_dir):
        profile_path = os.path.join(profiles_dir, name)
        if not os.path.isdir(profile_path) or not name.startswith("savana_"):
            continue
            
        db_path = os.path.join(profile_path, "state.db")
        soul_path = os.path.join(profile_path, "SOUL.md")
        
        if not os.path.exists(db_path) or not os.path.exists(soul_path):
            continue
            
        char_name = parse_character_name(soul_path)
        intimacy = parse_intimacy_score(soul_path)
        
        # Fetch recent messages (chronological order)
        messages = get_recent_messages(db_path, since_timestamp=0)
        if not messages:
            continue
            
        last_active = messages[-1]["timestamp"]
        elapsed = now - last_active
        
        # Criteria:
        # 1. Active in last 24h OR
        # 2. Cold chat: inactive between 72h and 96h (3-4 days), with high intimacy (>= 7)
        is_active_24h = elapsed < 24 * 3600
        is_cold_3d = (72 * 3600 <= elapsed <= 120 * 3600) and intimacy >= 7
        
        if is_active_24h or is_cold_3d:
            # For active chats, limit message history to last 24h only
            if is_active_24h:
                messages = [m for m in messages if m["timestamp"] >= (now - 24 * 3600)]
                
            all_profile_chats.append({
                "profile_id": name,
                "character_name": char_name,
                "soul_path": soul_path,
                "messages": messages,
                "last_active": last_active,
                "elapsed": elapsed,
                "intimacy": intimacy,
                "evolved_persona": extract_evolved_persona(soul_path)
            })

    # Sort: Prioritize cold chats (3-5 days) first to make sure they evolve, then sort by elapsed time ascending
    all_profile_chats.sort(key=lambda x: (not (72 * 3600 <= x["elapsed"] <= 120 * 3600), x["elapsed"]))

    # Limit to top 5 profiles to prevent large context sizes and DeepSeek API timeouts
    all_profile_chats = all_profile_chats[:5]

    if not all_profile_chats:
        print("# Active & Cold Characters Evolution Report\n\nNo active or cold characters met the evaluation criteria.")
        return

    print("# Active & Cold Characters Evolution Report")
    print("Generated at: {}\n".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    
    for chat in all_profile_chats:
        last_active_str = datetime.fromtimestamp(chat["last_active"]).strftime("%Y-%m-%d %H:%M:%S")
        
        # Calculate elapsed time string
        days = int(chat["elapsed"] // (24 * 3600))
        hours = int((chat["elapsed"] % (24 * 3600)) // 3600)
        elapsed_str = "{}天 {}小时".format(days, hours)
        status_tag = "活跃状态" if chat["elapsed"] < 24 * 3600 else "断联冷落状态"
        
        print("## Character: {} (Profile ID: {})".format(chat["character_name"], chat["profile_id"]))
        print("- SOUL.md File Path: {}".format(chat["soul_path"]))
        print("- Last Active Time: {}".format(last_active_str))
        print("- Time Elapsed: {} ({})".format(elapsed_str, status_tag))
        print("- Current Intimacy Level: {}/10".format(chat["intimacy"]))
        print("- Current Evolved Persona Section:")
        print("  ```markdown")
        print("  " + (chat["evolved_persona"] or "(暂无自主进化，人设遵循基础设定)"))
        print("  ```")
        print("\n### Dialogue History\n")
        
        for msg in chat["messages"]:
            dt_str = datetime.fromtimestamp(msg["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
            role_display = "User" if msg["role"] == "user" else chat["character_name"]
            content = msg["content"] or ""
            
            if "请以星野式沉浸表达回复用户这句话" in content:
                parts = content.split("这句话：")
                if len(parts) > 1:
                    raw_user_msg = parts[1].split("\n")[0]
                    content = "[Injected Template Input] -> " + raw_user_msg.strip()
                    
            print("[{} ({})]: {}".format(role_display, dt_str, content))
            print("")
        print("---\n")

if __name__ == "__main__":
    main()
