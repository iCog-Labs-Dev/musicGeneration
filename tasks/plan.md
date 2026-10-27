# Implementation Plan: M1 PR3 — CI-safe spine + m2a bridge

## Overview

Native groove + simple stem render + thin orchestrator under `aimusic.audio`,
plus optional `AIMUSIC_AUDIO_BACKEND=m2a` bridge. Fork PR into
`arsenylosev/musicGeneration:main`. Fluidsynth / reconcile deferred.

## Task List

- [x] Branch `feature/m1-simple-spine`; expand `[audio]`; pin `[audio-bridge]` @46b3dfe
- [x] Port groove / analysis / simple render / orchestrator
- [x] Wire `render-audio --no-validate-only`
- [x] Bridge + skipUnless tests
- [x] Docs / DECISIONS / tasks
- [x] Push + `gh pr create` → fork `main` (https://github.com/arsenylosev/musicGeneration/pull/4)

## Not doing

- Fluidsynth, reconcile CI, restyle/scoring, org PR to iCog
