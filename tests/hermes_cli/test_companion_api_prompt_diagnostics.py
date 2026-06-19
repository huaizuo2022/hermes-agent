import logging
import json

from hermes_cli import companion_api


def test_log_companion_prompt_diagnostics_emits_breakdown(caplog):
    caplog.set_level(logging.INFO, logger="hermes_cli.companion_api")

    companion_api._log_companion_prompt_diagnostics(
        session_id="savana_u_c",
        provider="deepseek",
        model="deepseek-chat",
        user_message="你好，记住我喜欢攀岩",
        conversation_history=[
            {"role": "user", "content": "上一轮用户消息"},
            {"role": "assistant", "content": "上一轮助手回复"},
        ],
        character_profile={
            "name": "测试角色",
            "personality": "温柔",
            "background": "背景设定",
            "sample_dialogues": [{"user": "u", "character": "a"}],
            "relationship": {"relationship_stage": "ambiguous"},
        },
        directives="额外指令",
        soul_text="# role\nhello",
        memory_text="memory block",
        user_profile_text="user block",
        session_stats={
            "message_count": 12,
            "input_tokens": 3456,
            "output_tokens": 789,
            "cache_read_tokens": 1200,
            "cache_write_tokens": 300,
        },
    )

    matched = [
        record
        for record in caplog.records
        if record.getMessage().startswith("savana_companion.prompt_diagnostics ")
    ]
    assert matched, "expected savana companion diagnostics log"

    record = matched[-1]
    payload = json.loads(record.getMessage().split(" ", 1)[1])
    assert payload["session_id"] == "savana_u_c"
    assert payload["provider"] == "deepseek"
    assert payload["model"] == "deepseek-chat"
    assert payload["history_messages"] == 2
    assert payload["history_user_messages"] == 1
    assert payload["history_assistant_messages"] == 1
    assert payload["user_message_chars"] > 0
    assert payload["directives_chars"] > 0
    assert payload["soul_chars"] > 0
    assert payload["memory_chars"] > 0
    assert payload["user_profile_chars"] > 0
    assert payload["profile_sample_dialogues"] == 1
    assert payload["profile_has_relationship"] is True
    assert payload["session_message_count"] == 12
    assert payload["session_input_tokens"] == 3456
    assert payload["session_output_tokens"] == 789
