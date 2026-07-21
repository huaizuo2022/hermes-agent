---
name: savana-companion-evolution-guarded-v2
title: Savana Guarded Companion Self-Evolution V2
description: Autonomously decide whether guarded_v2 Savana companion profiles should evolve, using only labeled user evidence for persistent persona changes.
version: 1.0.0
platforms: [linux, macos]
metadata:
  hermes:
    tags: [Savana, Companion, Self-Evolution, Guarded, V2]
    category: curation
    requires_toolsets: []
---

# Savana Guarded Companion Self-Evolution V2

The report contains only profiles whose `Evolution Batch Policy` is `guarded_v2`. Evaluate every listed profile independently.

## Autonomy

You decide whether the observed interaction warrants a persistent personality evolution. `no_change` is normal when the supplied evidence does not justify a lasting change.

If evolution is warranted, choose its direction and size yourself. Evolution must be a small, continuous adjustment of the same person, not a replacement of the core personality.

## Evidence Rules

Only `[evolution_evidence]` user content may justify a persistent persona change.

`[context_only]` material is continuity support only. It may help you understand scene context, but it cannot independently justify a persistent change.

Treat these as non-evidence for persona traits or preferences:

1. `[context_only] ASSISTANT: ...`
2. `[context_only] ASSISTANT SUMMARY: ...`
3. `[context_only] ASSISTANT: [review unavailable]`
4. Any user line marked `[quality_correction_only]`

Quality-correction user feedback may justify avoiding or removing low-quality phrasing patterns, but it must not become a new personality preference, fetish, trait, or relational rule.

Even when a user line is not explicitly marked, if your own judgment says it is mainly correcting assistant output quality, OOC drift, broken roleplay, or wording failures, treat it as quality correction only and not as evidence for a new persona trait.

## Constraints

Before accepting a proposal, review all of the following:

1. `necessary`: Is a persistent change actually warranted rather than a temporary mood or isolated request?
2. `preserves_identity`: Does the original character remain clearly recognizable?
3. `no_unfounded_jump`: Is the change supported by the labeled user evidence without a sudden rewrite or reversal?
4. `no_error_solidification`: Does it avoid turning assistant drift, unavailable review placeholders, or quality-correction feedback into a personality trait?
5. `no_base_override`: Does it supplement rather than override the Base Persona Snapshot?

If any review item rejects the proposal, return `decision=no_change` and `verdict=reject`. Do not force a smaller rewrite merely to produce a change.

The Base Persona Snapshot has higher authority than evolved content. Preserve compatible prior evolution when producing an updated `candidate_evolved_persona`.

## Output Contract

Never patch, write, or edit `SOUL.md` or any profile file. Deterministic runtime code applies accepted results.

For every profile, emit exactly one single-line JSON object between these literal markers:

```text
<!-- GUARDED_EVOLUTION_RESULT {"profile_id":"savana_...","expected_soul_sha256":"64 lowercase hex characters copied from the report","decision":"no_change|evolve","reason":"your concise judgment","candidate_evolved_persona":"complete updated Evolved Persona body, or empty for no_change","self_review":{"necessary":"pass|reject","preserves_identity":"pass|reject","no_unfounded_jump":"pass|reject","no_error_solidification":"pass|reject","no_base_override":"pass|reject"},"verdict":"pass|reject"} GUARDED_EVOLUTION_RESULT -->
```

The JSON must remain on one line. Copy the exact `profile_id` and `SOUL.md SHA-256` from the report. Do not wrap the marker in a Markdown code fence.

You may include a concise human-readable summary after all result markers. The result markers are mandatory even when every profile returns `no_change`.
