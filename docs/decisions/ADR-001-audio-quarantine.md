# ADR-001: Quarantine audio under optional product extra / standalone package

## Status
Accepted

## Date
2026-08-20

## Context
`musicGeneration` requires a light, pure symbolic core. Production audio needs
HTTP, optional heavy DSP, and nondeterministic endpoints.

## Decision
- Product: all audio under `aimusic.audio` + `[audio]` extra.
- This repo: standalone package `m2a` / `midi2audio-generative`.
- Core symbolic install must not pull torch/librosa/madmom/demucs.

## Alternatives considered
- Vendoring m2a into aimusic root — rejected (pollutes imports)
- Forever-only external repo with no product path — rejected as end state

## Consequences
Import quarantine tests in product; optional extras; CLI lazy-imports audio.
