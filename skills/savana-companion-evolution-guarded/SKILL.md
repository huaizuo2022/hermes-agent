---
name: savana-companion-evolution-guarded
title: Savana Guarded Companion Self-Evolution
description: Autonomously decide whether newly created Savana companion profiles should evolve, then self-review any proposed persona change before returning a structured result.
version: 1.0.0
platforms: [linux, macos]
metadata:
  hermes:
    tags: [Savana, Companion, Self-Evolution, Guarded]
    category: curation
    requires_toolsets: []
---

# Savana Guarded Companion Self-Evolution

The report contains only profiles whose `Evolution Batch Policy` is `guarded_v1`. Evaluate every listed profile independently.

## Autonomy

You decide whether the observed interaction warrants a persistent personality evolution. `no_change` is a normal and desirable result when the dialogue does not justify lasting change. There are no day, turn, score, evidence-count, or staged-progression thresholds.

If evolution is warranted, choose its direction and size yourself. It must feel like continuous growth of the same person rather than replacement by a new personality.

## Constraints

Before accepting a proposal, review all of the following:

1. `necessary`: Is a persistent change actually warranted rather than a temporary mood?
2. `preserves_identity`: Does the original character remain clearly recognizable?
3. `no_unfounded_jump`: Is the change supported by the supplied interaction without a sudden rewrite or reversal?
4. `no_error_solidification`: Does it avoid turning the assistant's own OOC wording, mechanical-language failure, or the user's correction of that failure into a personality trait?
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
