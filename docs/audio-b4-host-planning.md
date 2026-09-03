# Host-conditioned planning (B4) — deferred

Digest of `AUDIO_PIPELINE_ARCHITECTURE.pdf` §5. **Do not implement** unless
explicitly tasked. Seams are cut so next quarter can land without redesign.

## Idea

The reference paper's single-reference mode re-quantizes onsets onto a host
lattice **after** composition (repair). Because we search a graph with an
explicit energy function, we can instead constrain the **search**:

1. `audio.analysis.host` writes `host_lattice.json` (beats, downbeats, onset
   probability / accent profiles).
2. `planning.candidates` optionally rejects BeatStates whose groove positions
   are forbidden (or, in complement mode, occupied) by the host.
3. `planning.graph` adds energy `E_host(state)` = −log host onset probability at
   that metrical position, weighted by `w_host` in ScoringConfig.
4. Schrödinger Bridge then composes into the host rhythmic vocabulary.
   Bounded re-quant (`groove/lattice.py`) remains as fallback; on our own
   generated material it should be near a **no-op** (acceptance test).

## Import rule

Feedback travels as **files**. `audio.analysis.host` writes; planning reads via
a NumPy-only loader. Never import `aimusic.audio` from planning.

## ADR

See [ADR-004](decisions/ADR-004-defer-host-conditioned-planning.md).
