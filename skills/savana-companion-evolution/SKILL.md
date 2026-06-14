---
name: savana-companion-evolution
title: Savana AI Companion Self-Evolution Analyzer
description: Analyze dialogue histories of Savana AI characters to identify OOC, repetitions, relationship drift, and autonomously evolve character persona under ## Evolved Persona section in SOUL.md.
version: 2.0.0
platforms: [linux, macos]
metadata:
  hermes:
    tags: [Savana, Companion, Self-Evolution, Curation, Analysis, Quality]
    category: curation
    requires_toolsets: [file]
---

# Savana AI Companion Self-Evolution Analyzer

This skill guides the Hermes Curation Agent to analyze chats and autonomously rewrite individual characters' `SOUL.md` under `## Evolved Persona` using file tools.

Every character listed in the dialogues report is eligible for self-evolution. Do not skip a character only because their inactivity gap, relationship stage, or intimacy score does not match an old fixed threshold.

## Curation & Evolution Decision Flow

For each character listed in the dialogues report, analyze their status:

### 1. Universal Eligibility
- **Condition**: The character appears in the report.
- **Action**: Review the listed dialogue history, current relationship data, and current `## Evolved Persona` text. Call the `patch` tool to modify the character's `SOUL.md` file when the persona should evolve.
- **No Old Gate**: Do not require `Time Elapsed < 24 hours`, a 3-day cold gap, or `Current Intimacy Level >= 7`. These fields guide tone and intensity only.

### 2. Evolution Intensity
- **Active / recent chats**: Capture relationship shifts, new preferences, nicknames, repeated tension, softened boundaries, stronger attachment, or OOC corrections shown in the latest dialogue.
- **Quiet or cold chats**: Reflect distance, waiting, jealousy, insecurity, restraint, or renewed caution only when it fits the character and the elapsed time.
- **Low-intimacy chats**: Keep evolution subtle. Prefer small behavioral adjustments, improved memory use, or consistency fixes instead of forced intimacy.
- **Already accurate personas**: Use `No Patch Needed` only when the existing `## Evolved Persona` already captures the new dialogue state.

## Safety Guardrails
1. **Process All Listed Characters**: Evaluate every character in the report order. Do not let cold-gap characters crowd out active characters.
2. **Isolate Modifications**: Only edit text under the `## Evolved Persona` header. NEVER touch base traits like background, name, scenario, or base personality.
3. **Persona Consistency**: The emotional changes must remain consistent with the original character's nature (e.g., a cold CEO stays aloof but drops silent, jealous hints; a childhood friend becomes anxiously worried).
4. **No Harmful Content**: Prohibit extreme violence, pathological self-harm, or dangerous behaviors in the evolved persona.

## Curation Tool Call Guidelines
- To update `SOUL.md`, use the `patch` tool:
  - `path`: Use the exact absolute path from `SOUL.md File Path`.
  - `old_string`: The existing content of the `## Evolved Persona` block as reported.
  - `new_string`: The updated `## Evolved Persona` section containing the new traits.

## Output Report Structure
Your final response must summarize the changes in Markdown format:

```markdown
# Savana AI 伴侣自进化复盘诊断与人设演化报告 ([Date])

## 总体摘要
[简要概述本次复盘分析了多少个角色，多少个角色完成了自进化，多少个角色无需修改。]

---

## 动态进化详情

### 1. 角色：[Character Name] (Profile ID: [Profile ID])
- **状态类型**：[例如：活跃推进 / 近期沉淀 / 断联冷落 / 长期沉默]
- **人设修改状态**：[Success / No Patch Needed]
- **改动详情**：
  - **修改前 (Old)**: [Old Evolved Persona string]
  - **修改后 (New)**: [New Evolved Persona string]

---

### 2. 角色：[Next Character Name]...
```
