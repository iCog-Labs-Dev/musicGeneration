# MIDI → Audio Pipeline Architecture

**Status:** M0 landed (RenderPackage contract); audio DSP stages follow the roadmap.  
**Canonical design:** sibling repo `midi2audio_generative/AUDIO_PIPELINE_ARCHITECTURE.pdf`
(*From BeatState to Master*, v0.1). This document is the in-repo product digest.

## Executive summary

This repository is a **symbolic composer**: configs → GTTM energies → sparse graph →
Schrödinger Bridge → `BeatState` trajectory → multi-track `Score` → MIDI.

The optional audio producer turns that MIDI into production audio:
analysis → symbolic expressivization → deterministic stems → generative restyle →
automated scoring → mix/master.

The systems meet at exactly one artifact: a **RenderPackage** directory
(`run_<hash>/`). Integration is a **file contract**, not shared in-memory objects.

## Design bets

| ID | Bet | Why |
|----|-----|-----|
| **B1** | Structural bypass | Emit `structure.json` from planner `BeatState` fields; MIDI inference is fallback + reconcile cross-check |
| **B2** | Effects quarantine | Audio lives under `aimusic.audio` + optional `[audio]` extra; core stays `numpy` + `mido` |
| **B3** | Microtonality first-class | Route/render/restyle by `(role, edo, pitch content)`; gate with tuning-preservation floors |
| **B4** | Host-conditioned planning | Inject host onset lattice into planning (next quarter); see [audio-b4-host-planning.md](audio-b4-host-planning.md) |

## System context

```text
aimusic (symbolic)  →  RenderPackage  →  aimusic.audio / m2a bridge  →  stems / master
```

The only backward arrow (Phase 4 / B4) is a **data file** (`host_lattice.json`),
never a reverse import from audio into planning.

## Package layout (target)

```text
aimusic/
├── core/render_package.py     # contract producer (landed in M0)
├── render/package.py          # thin re-export
├── theory/edo.py              # + scala/MTS helpers (M1.5)
├── app/cli.py                 # + render-audio / expressivize / audit (later)
└── audio/                     # optional quarantine (stub in M0)
    ├── analysis/ groove/ render/ restyle/ prompts/
    ├── scoring/ search/ mixmaster/
    └── orchestrator.py
config/audio.default.yaml
tests/test_architecture.py
tests/test_render_package.py
```

## Import rules (enforced)

1. Nothing outside `aimusic.audio` may import `aimusic.audio`.
2. `aimusic.audio` may import only `aimusic.core.*` and `aimusic.theory.*`.
3. Importing `aimusic.core` must not pull `torch`, `librosa`, `madmom`, or `demucs`.

See [test_architecture.py](../tests/test_architecture.py) and [DECISIONS.md](../DECISIONS.md).

## Determinism, caching, provenance

- Cache keys: content hashes of inputs + canonical config subset + code version.
- Nondeterministic steps (LLM groove authoring, paid endpoints) sit behind artifact
  boundaries; cache hits are bit-identical.
- `RunManifest` is extended with audio stage records (prompts, seeds, costs, hashes).
- Budget: `max_endpoint_calls` / `max_usd_estimate` consulted before every paid call.
- MIDI-GPT (CC-BY-NC) stays behind a config flag defaulting **off**.

## Current capability note

Symbolic Method A, multi-track decode, and 12/19-EDO MIDI (including MPE/MTS paths)
are already on fork `main`. Audio work still develops against a **fixture corpus**
so DSP never blocks on symbolic changes. The contract is the decoupler.

## Related docs

- [render-package-contract.md](render-package-contract.md)
- [audio-roadmap.md](audio-roadmap.md) — M0–M6 quarter plan
- [audio-stages.md](audio-stages.md) — STAGE 1–6 design
- [audio-next-steps.md](audio-next-steps.md) — agent backlog
- [docs/decisions/](decisions/) — ADRs
