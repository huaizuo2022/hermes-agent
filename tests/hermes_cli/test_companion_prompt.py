from __future__ import annotations

from run_agent import AIAgent

from hermes_cli.companion_prompt import build_companion_system_prompt


def test_build_companion_system_prompt_includes_only_present_blocks():
    prompt = build_companion_system_prompt(
        soul_text="她敏感、克制，但会认真接住情绪。",
        memory_text="上一轮事实摘要：用户上周刚搬家。",
        user_profile_text="用户偏好：讨厌太官方的称呼。",
    )

    assert "角色 SOUL" in prompt
    assert "她敏感、克制，但会认真接住情绪。" in prompt
    assert "共享记忆" in prompt
    assert "上一轮事实摘要：用户上周刚搬家。" in prompt
    assert "用户上下文" in prompt
    assert "用户偏好：讨厌太官方的称呼。" in prompt
    assert "上一轮事实摘要只用于承接事实，不是说话样例" in prompt


def test_build_companion_system_prompt_omits_empty_headings_and_legacy_terms():
    prompt = build_companion_system_prompt(
        soul_text="只保留角色灵魂。",
        memory_text="",
        user_profile_text="",
    )

    assert "角色 SOUL" in prompt
    assert "共享记忆" not in prompt
    assert "用户上下文" not in prompt

    forbidden_terms = (
        "action='add'",
        "target='user'",
        "tool call",
        "terminal",
        "Session ID",
        "Provider",
        "Hermes Agent",
    )
    for term in forbidden_terms:
        assert term not in prompt


def test_build_system_prompt_returns_override_verbatim():
    agent = object.__new__(AIAgent)
    agent._system_prompt_override = "COMPANION OVERRIDE"

    assert agent._build_system_prompt() == "COMPANION OVERRIDE"


def test_build_system_prompt_without_override_still_calls_legacy_builder(monkeypatch):
    agent = object.__new__(AIAgent)
    calls = []

    def fake_build_system_prompt(instance, system_message=None):
        calls.append((instance, system_message))
        return "LEGACY PROMPT"

    monkeypatch.setattr(
        "agent.system_prompt.build_system_prompt",
        fake_build_system_prompt,
    )

    assert agent._build_system_prompt(system_message="custom") == "LEGACY PROMPT"
    assert calls == [(agent, "custom")]
