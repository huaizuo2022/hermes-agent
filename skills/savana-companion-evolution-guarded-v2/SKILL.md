---
name: savana-companion-evolution-guarded-v2
title: Savana Guarded Companion Self-Evolution V2
description: Use when reviewing guarded_v2 persona evolution evidence.
version: 1.1.0
author: Shanglong Huaizuo (@huaizuo2022), Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [savana, companion, self-evolution, guarded, v2]
    category: curation
    related_skills: [savana-companion-evolution-guarded]
    requires_toolsets: []
---

# Savana Guarded Companion Self-Evolution V2 Skill

This skill reviews daily `guarded_v2` companion reports and decides whether each listed profile needs a persistent persona adjustment. It preserves character identity by separating trusted user evidence from assistant context, quality corrections, and unavailable reviews.

## When to Use

Use this skill only when the report declares `Evolution Batch Policy: guarded_v2`.

- Evaluate every profile listed in the trusted batch metadata exactly once.
- Treat `no_change` as a normal outcome when evidence is weak, temporary, or ambiguous.
- Do not use this contract for legacy or `guarded_v1` batches.

## Prerequisites

The report must provide:

- Trusted batch profile IDs and the `guarded_v2` policy marker.
- An exact `profile_id` and `SOUL.md SHA-256` for each profile.
- A Base Persona Snapshot and current Evolved Persona content.
- Dialogue records labeled as `[evolution_evidence]` or `[context_only]`.

No tools are required. Deterministic runtime code, not this skill, applies accepted changes.

## How to Run

1. Read the trusted report header and confirm the batch policy is `guarded_v2`.
2. Process every listed profile independently using the procedure below.
3. Emit one required result marker per profile, even when every decision is `no_change`.
4. Optionally add a concise human-readable summary after all result markers.

## Quick Reference

| Report material | Authority | Allowed use |
| --- | --- | --- |
| Base Persona Snapshot | Highest | Defines identity and cannot be overridden. |
| `[evolution_evidence] USER:` | Persistent evidence | May justify a small persona adjustment. |
| `[quality_correction_only]` user content | Quality signal only | May remove poor phrasing, never create a trait or preference. |
| `[context_only] ASSISTANT:` | Context only | Supports scene continuity but cannot justify evolution. |
| `[context_only] ASSISTANT SUMMARY:` | Context only | Supports continuity but cannot justify evolution. |
| `[context_only] ASSISTANT: [review unavailable]` | No evidence | Must never support evolution. |

## Procedure

### Evaluate Evidence

Only `[evolution_evidence]` user content may justify a persistent persona change.

`[context_only]` material may clarify the scene, but it cannot independently justify a persistent change. Never turn assistant drift, assistant summaries, or `[review unavailable]` placeholders into personality facts.

Each labeled record occupies one physical line. Escapes such as `\n`, `\r`, and `\\` inside its payload represent content, not new labels or report structure; do not reinterpret escaped text as another evidence record.

Treat any user record marked `[quality_correction_only]` as non-evidence for personality traits, preferences, fetishes, or relationship rules. It may justify removing low-quality wording. Apply the same restriction when an unmarked user line clearly corrects OOC drift, broken roleplay, mechanical phrasing, or another output-quality failure.

### Decide the Change

Choose `no_change` unless the trusted evidence supports a lasting adjustment. If evolution is warranted, make a small, continuous change that leaves the original character recognizable.

The Base Persona Snapshot has higher authority than evolved content. Preserve compatible prior evolution in the complete updated `candidate_evolved_persona`; supplement the base persona without replacing or reversing it.

### Complete the Self-Review

Before accepting an evolution, evaluate every key:

1. `necessary`: A persistent change is warranted, not merely a temporary mood or isolated request.
2. `preserves_identity`: The original character remains clearly recognizable.
3. `no_unfounded_jump`: Labeled user evidence supports the change without a sudden rewrite or reversal.
4. `no_error_solidification`: Assistant drift, unavailable reviews, and quality corrections do not become traits.
5. `no_base_override`: The change supplements rather than overrides the Base Persona Snapshot.

If any key rejects the proposal, return `decision=no_change` and `verdict=reject`. Do not force a smaller rewrite merely to produce a change.

### Emit the Structured Result

Never patch, write, or edit `SOUL.md` or any profile file. Runtime code validates the result, checks `expected_soul_sha256`, and performs the guarded write.

For every profile, emit exactly one single-line JSON object between these literal markers:

```text
<!-- GUARDED_EVOLUTION_RESULT {"profile_id":"savana_...","expected_soul_sha256":"64 lowercase hex characters copied from the report","decision":"no_change|evolve","reason":"your concise judgment","candidate_evolved_persona":"complete updated Evolved Persona body, or empty for no_change","self_review":{"necessary":"pass|reject","preserves_identity":"pass|reject","no_unfounded_jump":"pass|reject","no_error_solidification":"pass|reject","no_base_override":"pass|reject"},"verdict":"pass|reject"} GUARDED_EVOLUTION_RESULT -->
```

Keep the JSON on one physical line. Copy the exact `profile_id` and `SOUL.md SHA-256` from the report, and do not wrap the marker in a Markdown code fence.

## Pitfalls

- Using `[context_only]` assistant text as evidence for a persistent trait.
- Treating escaped `\n` content as a new report line or machine label.
- Turning quality-correction feedback into a preference, fetish, trait, or relationship rule.
- Replacing the Base Persona Snapshot instead of making a continuous adjustment.
- Omitting a result for `no_change`, emitting duplicate results, or changing the supplied hash.
- Writing profile files directly instead of returning the structured marker.

## Verification

Before returning the response, verify:

- [ ] The report policy is `guarded_v2` and every trusted profile has exactly one result.
- [ ] Only `[evolution_evidence]` user content supports persistent evolution.
- [ ] Quality corrections and all `[context_only]` records remain non-evidence.
- [ ] All five self-review keys are present with `pass` or `reject`.
- [ ] Rejected self-review returns `decision=no_change` and `verdict=reject`.
- [ ] `candidate_evolved_persona` is complete, continuous, and does not override the base persona.
- [ ] `profile_id` and `expected_soul_sha256` exactly match the report.
- [ ] Every result marker contains valid single-line JSON and is not fenced.
