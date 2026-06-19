import logging

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
        if record.getMessage() == "savana_companion.prompt_diagnostics"
    ]
    assert matched, "expected savana companion diagnostics log"

    record = matched[-1]
    assert record.session_id == "savana_u_c"
    assert record.provider == "deepseek"
    assert record.model == "deepseek-chat"
    assert record.history_messages == 2
    assert record.history_user_messages == 1
    assert record.history_assistant_messages == 1
    assert record.user_message_chars > 0
    assert record.directives_chars > 0
    assert record.soul_chars > 0
    assert record.memory_chars > 0
    assert record.user_profile_chars > 0
    assert record.profile_sample_dialogues == 1
    assert record.profile_has_relationship is True
    assert record.session_message_count == 12
    assert record.session_input_tokens == 3456
    assert record.session_output_tokens == 789
