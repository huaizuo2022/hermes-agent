from __future__ import annotations


def build_companion_system_prompt(
    soul_text: str,
    memory_text: str = "",
    user_profile_text: str = "",
) -> str:
    sections = []

    soul = str(soul_text or "").strip()
    memory = str(memory_text or "").strip()
    user_profile = str(user_profile_text or "").strip()

    if soul:
        sections.append("角色 SOUL\n{0}".format(soul))
    if memory:
        sections.append("共享记忆\n{0}".format(memory))
    if user_profile:
        sections.append("用户上下文\n{0}".format(user_profile))

    sections.append(
        "连续性规则\n"
        "你要延续这段关系的情感与事实一致性，优先承接既有经历、称呼、情绪走向与边界。\n"
        "上一轮事实摘要只用于承接事实，不是说话样例；不要模仿它的措辞、语气或句式。"
    )

    return "\n\n".join(sections).strip()
