# Tension Diagnostics — Requirements & Design Spec

> **Status:** Draft, pre-implementation. This document is the reference for
> the fix and its acceptance criteria. Update it if the design changes during
> implementation — it should stay accurate, not just aspirational.

---

## 1. Problem statement

Tension in this codebase is currently computed **four separate times**, in
four separate places, with four separate formulas — none of which use tonal
distance:

| # | Location | Inputs used | Status |
|---|----------|-------------|--------|
| 1 | `aimusic/core/diagnostics.py:99` `compute_tension_curve()` | role label only, against a **fictional** role vocabulary (`"Tonic"`, `"Subdominant"`, `"Dominant"`) that doesn't match the real one (`"hold"`, `"prep"`, `"change"`, `"cad"`) | Dead code — never called from the real pipeline, only from its own test |
| 2 | `aimusic/app/cli.py:27-32,73-83` `ROLE_TENSION` dict | role label + boundary level | **Actually used** — feeds `StructuralDiagnostics.tension_curve` in the manifest today |
| 3 | `aimusic/decode.py:72-81` `_tension_level()` | role label + boundary level | Used for note velocity/expression during rendering (different purpose — audio dynamics, not diagnostics) |
| 4 | `aimusic/planning/plans.py` `PlanningSection.target_tension_arc` | authored per-section target | Captured into `MethodAPlanDiagnostics.target_tension_arcs`, but **never sampled into a curve and never compared against anything realized** |

None of the diagnostics-facing formulas (#1, #2) reference `key_id` or
`chord_id`, so key motion and chord motion — the actual harmonic distance
being traveled — have no effect on tension. The circle-of-fifths / TPS math
already exists and is tested (`aimusic/theory/tonal.py:199-275`,
`tonal_distance()` and `basic_space_distance()`, EDO-generic), it's just
never wired into tension.

There is also no mechanism anywhere to compare a section's authored
`target_tension_arc` against what the selected path actually produced.

---

## 2. Goal

Replace the ad hoc, duplicated tension logic with **one documented, versioned,
pure tension function** that combines role, boundary, key motion, and chord
motion (with an optional head/groove contribution), and use it to:

1. Produce a **realized tension curve** from a selected `BeatState` path.
2. Produce a **target tension curve** from the section plan (`PlanningSection.target_tension_arc`, sampled across each section's time range).
3. **Compare** the two and report absolute error, section-level error, peak-timing offset, and a curve-shape summary.
4. Surface all of the above in the run manifest.

`decode.py`'s `_tension_level()` is a separate concern (drives MIDI velocity/expression, not diagnostics) and is **out of scope** — it will not be touched or unified in this pass. Called out explicitly so it isn't silently forgotten or conflated later.

---

## 3. Scope — files involved

### New

- **`aimusic/scoring/tension.py`** — the new module. Everything below lives here.
- **`tests/test_tension.py`** — new test module.

### Modified

- **`aimusic/core/diagnostics.py`**
  - Remove `compute_tension_curve()` (dead code, wrong vocabulary).
  - `StructuralDiagnostics` gains `target_tension_curve: List[Tuple[float, float]]` and `tension_deviation: Dict[str, Any]` fields, included in `to_dict()`.
  - `RunManifest` gains `tension_model_version: str` (mirrors `TENSION_MODEL_VERSION` from the new module), included in `to_dict()`.
- **`aimusic/app/cli.py`**
  - `_build_structural_diagnostics` stops using the local `ROLE_TENSION` dict / inline formula; calls `aimusic.scoring.tension.realized_tension_curve(...)` instead.
  - New step: build the target curve from `plan_result.endpoints.sections` via `aimusic.scoring.tension.target_tension_curve(...)`, and the deviation report via `compare_tension_curves(...)`.
  - `ROLE_TENSION` dict removed from `cli.py` (superseded by `TensionWeights` in the new module).
- **`tests/test_diagnostics.py`**
  - Remove/replace `test_compute_tension_curve` (currently asserts against the fictional vocabulary — this test is *validating the bug*). Replaced with an assertion that `StructuralDiagnostics`/`RunManifest` correctly carry the new curve/deviation fields end-to-end.

### Untouched (explicitly)

- `aimusic/theory/tonal.py` — `tonal_distance()` / `basic_space_distance()` are consumed as-is, not modified.
- `aimusic/decode.py` — `_tension_level()` stays as the rendering-dynamics function it is.
- `aimusic/planning/plans.py` — `PlanningSection.target_tension_arc` schema is consumed as-is; no change to how sections are authored.
- `aimusic/scoring/gttm_features.py` — has prior art (`harmonic_key_proximity_feature`, `harmonic_chord_proximity_feature`, `cadential_harmonic_motion_feature`), built on the same `tonal_distance`/`basic_space_distance` primitives over the same `(prev_state, next_state)` convention, and — like the old `compute_tension_curve` — currently dead outside its own test module. **Not reused directly**: those are unbounded, unclamped proximity scores (`1/(1+distance)`, with negative sentinel values for unresolved tokens) meant for GTTM-style prior scoring, not a clamped `[0,1]` diagnostics value. The new tension module uses the same decay family (`distance/(1+distance)`, its functional inverse) for consistency rather than inventing a third unrelated scheme. See the module docstring in `aimusic/scoring/tension.py` for the explicit rationale.

---

## 4. Design

### 4.1 Versioning

```python
TENSION_MODEL_VERSION = "1.0.0"
```

Bumped whenever the formula or weight semantics change. Stored in the
manifest so old manifests can be told apart from new ones. Documented in a
module docstring: what each version means, changelog as a comment block at
the top of the file.

### 4.2 `TensionWeights` (pure config, not I/O)

```python
@dataclass(frozen=True)
class TensionWeights:
    role: float = ...
    boundary: float = ...
    key_motion: float = ...
    chord_motion: float = ...
    head_groove: float = 0.0   # optional, defaults off
```

Explicit, named, defaults documented with rationale (e.g. why role dominates
boundary). No magic numbers buried in formula bodies.

### 4.3 `beat_tension(prev_state, state, vocabularies, edo, weights=DEFAULT_WEIGHTS) -> float`

Pure function. No I/O, no globals, no mutation. Given two consecutive
`BeatState`s plus the `Vocabularies`/`edo` needed to resolve their labels and
roots:

- **role**: resolved via `vocabularies.roles.token_for_id(state.role_id).label`, mapped through a documented table (correct labels this time: `hold`/`prep`/`change`/`cad`).
- **boundary**: `state.boundary_lvl`, normalized against the boundary vocabulary's max level (not a hardcoded constant).
- **key motion component**: `tonal_distance(prev_key.root_pc, key.root_pc, edo)` between the two states' resolved `KeyToken.root_pc`, normalized via `distance / (1 + distance)` — an unbounded decay (same functional family as `gttm_features`'s `1/(1+distance)` proximity score, inverted), not a hand-picked cap, so it can't saturate early or need EDO-specific tuning.
- **chord motion component**: `basic_space_distance(prev_chord.root_pc, prev_chord.quality, chord.root_pc, chord.quality, edo)` between resolved `ChordToken`s, normalized the same way (`distance / (1 + distance)`). This matters because `basic_space_distance`'s `k` term (symmetric difference across three basic-space levels) isn't bounded by `edo` the way the root-distance term `j` is — a fixed divisor risks saturating to 1.0 too early for wide chord vocabularies (7-tone extended chords), losing discriminative power. The decay formula avoids that by construction.
- **head/groove (optional)**: small contribution from head label change / groove subdivision density change, gated by `weights.head_groove` (0 by default — must be explicitly opted into).

Combined via weighted sum, then clamped to `[0, 1]`. Exact combination formula
documented in the function docstring with a worked numeric example.

For the first beat in a path (no `prev_state`), key/chord motion terms are 0
by convention (documented) — tension there falls back to role + boundary
only.

### 4.4 `realized_tension_curve(path, vocabularies, edo, weights=DEFAULT_WEIGHTS) -> List[Tuple[float, float]]`

Walks consecutive pairs in `path`, calling `beat_tension` for each beat index,
returning `(time, tension)` pairs. This is what replaces `cli.py`'s inline
`ROLE_TENSION` logic and the dead `compute_tension_curve`.

### 4.5 `target_tension_curve(sections: Sequence[PlanningSection]) -> List[Tuple[float, float]]`

For each section, linearly interpolates its `target_tension_arc` (which may
have more than 2 control points — see `PlanningSection.__post_init__`,
minimum length 2) across `[start_time, end_time)`, producing one `(time,
value)` sample per beat, matching the time indexing used by
`realized_tension_curve` so the two curves are directly comparable
point-for-point.

### 4.6 `TensionDeviationReport` + `compare_tension_curves(target, realized, sections) -> TensionDeviationReport`

```python
@dataclass(frozen=True)
class TensionDeviationReport:
    mean_absolute_error: float
    max_absolute_error: float
    section_errors: Dict[str, float]          # section name -> MAE within that section's time range
    target_peak_time: float
    realized_peak_time: float
    peak_timing_offset: float                 # realized - target, signed
    shape_correlation: float                  # Pearson correlation between the two curves, [-1, 1]
```

- **Absolute error**: point-wise `|target - realized|` over the overlapping
  time range; `mean_absolute_error` and `max_absolute_error` summarize it.
- **Section-level error**: same MAE, computed per section by slicing both
  curves to `[section.start_time, section.end_time)`.
- **Peak timing**: time index of the max value in each curve, and their
  signed difference.
- **Curve-shape summary**: Pearson correlation coefficient between target and
  realized series as a cheap, well-understood shape-similarity metric (pure
  Python/`statistics` or `numpy`, no new dependency).

`compare_tension_curves` is pure — takes two curves + sections, returns a
report, no side effects.

### 4.7 Wiring into the manifest

`cli.py`'s `_build_structural_diagnostics` becomes responsible for calling
all three functions and packing the results into `StructuralDiagnostics`
(realized + target curves) and a separate `tension_deviation` dict
(`dataclasses.asdict(report)`) attached either on `StructuralDiagnostics` or
directly on `RunManifest.to_dict()` — final placement decided during
implementation, documented here once settled.

---

## 5. Acceptance criteria (verification checklist)

These map directly to the task's stated acceptance criteria, made concrete
and checkable:

1. **Tonally distant transitions produce greater tension than equivalent
   local transitions, under controlled fixtures.**
   - Fixture: two consecutive `BeatState` pairs with identical role, boundary,
     and head/groove, differing only in chord/key root.
   - Case A: root moves by a single fifth (e.g. C → G in 12-EDO).
   - Case B: root moves by a tritone / maximally distant interval (e.g. C →
     F♯ in 12-EDO).
   - Assert `beat_tension(A) < beat_tension(B)`.
   - Repeated for 19-EDO with EDO-appropriate "near" vs "far" root pairs
     (using `nearest_roots()` / `tonal_distance()` directly to pick them, not
     hardcoded 12-EDO intervals).

2. **Changing a role or boundary changes tension in the documented
   direction.**
   - Fixture: fixed key/chord motion (e.g. both 0, same root).
   - Assert `beat_tension(role=hold) < beat_tension(role=prep) < beat_tension(role=change) < beat_tension(role=cad)`, all else equal.
   - Assert increasing `boundary_lvl` (holding role fixed) strictly increases
     tension, up to the vocabulary's max boundary level.
   - Both directions documented in the `beat_tension` docstring, and the test
     asserts against that documented ordering rather than just "some
     inequality holds."

3. **Manifests contain target and realized curves plus deviation metrics.**
   - End-to-end test: run a small `MethodARunConfig` through `run_method_a`
     (or the fixture graph pattern already used in
     `tests/test_diagnostics.py::test_e2e_produce_stable_short_passage`),
     build the manifest, serialize to JSON, reload, and assert:
     - `structure.tension_curve` (realized) is present, non-empty, length
       matches path length.
     - `structure.target_tension_curve` is present, non-empty, same length/time-indexing as realized.
     - `structure.tension_deviation` (or wherever it lands) is present and
       contains `mean_absolute_error`, `section_errors`, `peak_timing_offset`,
       `shape_correlation` keys with numeric values.
     - `tension_model_version` is present at the top level and equals
       `TENSION_MODEL_VERSION`.

4. **Tests cover 12-EDO and 19-EDO tonal-distance behavior.**
   - Every fixture in criterion #1 and the chord-motion equivalent is
     parameterized (or duplicated) across `edo=12` and `edo=19`, following the
     existing pattern in `tests/test_tonal.py` (`TestIntervalHelpers`, which
     already tests both EDOs side by side).
   - At least one test exercises `basic_space_distance` differences (chord
     quality change, not just root) at both EDOs, since chord motion is a
     distinct weighted term from key motion.

### Additional non-functional checks

- **Purity**: a test calls `beat_tension` twice with identical inputs
  (including fresh, separately-constructed `BeatState`/`Vocabularies`
  objects) and asserts identical output — guards against hidden global state
  or ordering dependence.
- **No regressions in dependents**: `aimusic/decode.py` is untouched;
  existing `tests/test_decode.py` (if present) should pass unmodified — this
  confirms the rendering-dynamics tension path wasn't accidentally coupled to
  the new module.
- **Dead code removed cleanly**: `compute_tension_curve` and its import in
  `tests/test_diagnostics.py` are gone; `grep -rn compute_tension_curve` over
  the repo returns nothing.
- **`ROLE_TENSION` dict removed from `cli.py`**; `grep -rn ROLE_TENSION`
  returns nothing (weights now live only in `TensionWeights`).

---

## 6. Test plan (`tests/test_tension.py`)

| Test | Covers acceptance criterion |
|---|---|
| `test_key_motion_distant_exceeds_local_12edo` | #1 |
| `test_key_motion_distant_exceeds_local_19edo` | #1, #4 |
| `test_chord_motion_quality_change_increases_tension_12edo` | #1, #4 |
| `test_chord_motion_quality_change_increases_tension_19edo` | #1, #4 |
| `test_role_ordering_documented_direction` | #2 |
| `test_boundary_level_monotonic_increase` | #2 |
| `test_first_beat_no_prev_state_falls_back_to_role_boundary` | design §4.3 edge case |
| `test_beat_tension_is_pure_and_deterministic` | non-functional purity |
| `test_target_tension_curve_interpolates_section_arc` | §4.5 |
| `test_target_tension_curve_multi_control_point_arc` | §4.5 (arcs with >2 points) |
| `test_realized_tension_curve_matches_path_length` | §4.4 |
| `test_compare_tension_curves_mean_absolute_error` | §4.6 |
| `test_compare_tension_curves_section_level_error` | §4.6 |
| `test_compare_tension_curves_peak_timing_offset` | §4.6 |
| `test_compare_tension_curves_shape_correlation` | §4.6 |
| `test_manifest_contains_target_realized_and_deviation` (in `test_diagnostics.py`, e2e) | #3 |
| `test_tension_model_version_present_in_manifest` | #3, versioning |

Each fixture-based test constructs `BeatState`/`Vocabularies` directly (no
mocks needed — these are plain dataclasses), following the existing style in
`tests/test_tonal.py` and `tests/test_diagnostics.py`.

---

## 7. Explicitly out of scope for this pass

- Unifying `decode.py`'s velocity/expression tension with diagnostics
  tension — different consumer, different constraints, flagged for a
  possible future pass rather than folded in silently.
- Changing how `PlanningSection.target_tension_arc` is authored/generated
  upstream (`build_section_plan` in `plans.py`) — we consume it as-is.
- Any change to `tonal_distance` / `basic_space_distance` themselves — they're
  correct and tested; we're wiring them in, not modifying them.
