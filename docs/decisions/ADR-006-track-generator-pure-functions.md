# ADR-006: Pure state/window/RNG track-generator functions (REQ-11)

## Status
Accepted

## Date
2026-09-07

## Context
The design spec calls for pure `gen_drums`, `gen_bass`, `gen_comping`, and
`gen_lead` functions taking beat state, a local window, and an RNG key. The
actual implementations (`generate_bass_events`, `generate_comping_events`,
`generate_lead_events`, `generate_drum_events` in `aimusic/decode.py`) instead
looped over an entire path internally, and accepted a `key: RNGKey` parameter
that was never read or advanced -- every one of the four functions returned
the same `key` object it was given, unchanged. Per-track "memory" (previous
bass/lead pitch, the previous comping voicing) lived in a plain local
variable mutated across loop iterations -- invisible to callers, impossible
to inspect, override, or replay independently of running the whole path.

## Decision
- Introduced a shared per-beat contract: `BeatWindow` (prev/next structural
  state, beat index/total, first/last/phrase-edge flags), a `TrackStepResult`
  (events + next decoder state + next key), and a `TrackGenerator` Protocol
  that all four generators implement identically.
- Added explicit decoder-state types -- `BassDecoderState`,
  `CompingDecoderState`, `LeadDecoderState`, `DrumDecoderState` -- that carry
  exactly the voice-leading/voicing memory each track needs. Nothing is
  hidden in a closure anymore; state is a value passed in and a value
  returned.
- `gen_bass`, `gen_comping`, `gen_lead`, `gen_drums` are now pure per-beat
  functions: `(state, window, *, decoder_state, key, decode_config,
  vocabularies, edo, ticks_per_beat) -> TrackStepResult`. Each is
  independently callable on one beat, with no path, no loop, and no shared
  state with any other track.
- `_run_track` is the single place that loops over a path, builds windows,
  and threads decoder state + key from one beat to the next. The original
  `generate_*_events` functions are now thin, signature-compatible wrappers
  around `_run_track` + the matching `gen_*` function, so every existing
  caller (including `decode_path_to_score`) is unaffected.
- Per-beat RNG is now real: `_activation_unit` either reproduces the
  original beat-index-hash activation exactly (`DecodeConfig
  .deterministic_activation=True`, the default -- a *fixed policy*) or draws
  a genuine random unit from the per-beat key (`False`). Either way the key
  is advanced once per beat, so `next_key != key` is now actually true, and
  a probabilistic activation policy can be swapped in without touching any
  `gen_*` call site.
- Track order independence was already implicit (`allocate_named_keys`
  derives each track's key by name, independent of call order, and no track
  reads another track's state) and remains true after the refactor; it is
  now covered by an explicit test rather than being an accidental property.

## Alternatives considered
- Keep the whole-path functions and just start consuming `key` inside their
  loops -- rejected; it would satisfy "explicit RNG output" but not
  "independently callable and testable" or "no hidden mutable voice state,"
  which are separate, explicit acceptance criteria.
- A single generic decoder-state dict shared across all four tracks --
  rejected in favor of one small frozen dataclass per track, so each track's
  state shape is self-documenting and type-checked rather than a loosely
  keyed blob.
- Drop `DecodeConfig.deterministic_activation` and always draw from the key
  -- rejected because it would change existing note-event output (and
  therefore every register/density/anchoring test) for no requirement gain;
  REQ-11 explicitly asks to keep a deterministic policy available.

## Consequences
- `aimusic/decode.py` gains `BeatWindow`, `TrackStepResult`,
  `TrackGenerator`, `BassDecoderState`, `CompingDecoderState`,
  `LeadDecoderState`, `DrumDecoderState`, `gen_bass`, `gen_comping`,
  `gen_lead`, `gen_drums`, `_build_windows`, `_run_track`, and
  `_activation_unit`. `_should_emit` now takes a per-beat activation unit
  instead of a raw beat index (pure function, no behavior change).
- `aimusic/core/config.py`: `DecodeConfig` gains
  `deterministic_activation: bool = True`.
- `tests/test_decode.py` is unchanged and still passes bit-for-bit --
  confirming register, density, anchoring, and overlap guarantees are fully
  preserved by the refactor.
- `tests/test_track_generators.py` (new) covers the acceptance criteria:
  independent callability, replay determinism, explicit decoder state for
  voice-leading, track-order independence, and the fixed-vs-keyed activation
  policy.
- Backward compatible: `generate_bass_events` / `generate_comping_events` /
  `generate_lead_events` / `generate_drum_events` keep their existing
  signatures and return types; `decode_path_to_score` required no changes.
