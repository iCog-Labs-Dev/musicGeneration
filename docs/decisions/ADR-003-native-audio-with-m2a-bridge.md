# ADR-003: Phased hybrid — native aimusic.audio + optional m2a bridge

## Status
Accepted

## Date
2026-08-20

## Context
M1–M3 already landed in this repo. Upstream needs reviewable PRs without
discarding that work or living forever on a submodule.

## Decision
1. Harden pipeline here (`m2a`).
2. Port into `aimusic.audio` for product PRs.
3. Optional bridge (`AIMUSIC_AUDIO_BACKEND=m2a` / `[audio-bridge]`) during port.
4. Native path is product source of truth after parity.
5. Prefer developing product on a **fork** of musicGeneration; keep this repo as R&D/docs/tags.

## Alternatives considered
- Git submodule — rejected
- Fork-only rewrite discarding m2a — rejected
- Permanent polyrepo with no upstream audio — rejected as end state

## Consequences
Tag releases (`v0.1.0`); avoid dual-maintaining identical modules; document
handoff in HANDOFF_TO_FORK.md.
