# Pipeline Evaluation & Diagnostics Guide

This document defines the evaluation criteria for the generated symbolic music and catalogs known failure modes within the GTTM + Schrödinger Bridge pipeline.

## 1. Evaluation Criteria

When inspecting a generated run (via the `inspect` CLI command or listening to the `.mid` output), use the following criteria to evaluate the quality of the generation:

### A. Structural Coherence
* **Metric:** The `tension_curve` captured in the run manifest.
* **Pass Criteria:** The tension curve should demonstrate deliberate arcs (e.g., resting at 0.1, climbing to 0.9, and resolving). Flatlines at 0.5 indicate a failure of the symbolic planner to assign meaningful prolongational roles.
* **Goal:** The piece must reflect human-like phrasing (e.g., A-B-A form) rather than random Markov-chain wandering.

### B. Cadence Plausibility
* **Metric:** The `chord_timeline` and `role_timeline` boundaries.
* **Pass Criteria:** Phrases must end with culturally appropriate harmonic resolutions relative to the assigned N-EDO tuning. In standard configurations, this means observing a high-tension role (Dominant) stepping gracefully into a low-tension role (Tonic) at section boundaries.

### C. Decode Validity
* **Metric:** Successful MIDI rendering and `SBConvergenceTrace`.
* **Pass Criteria:** The `map_bridge_path` must extract a strict 1:1 timeline of notes from the `SparseGraph`. The path score must yield a finite log-probability, proving that the generated notes are mathematically valid within the transition rules.

---

## 2. Failure Modes & Fallback Behaviors

The pipeline is designed to "fail fast" and deterministically. If a generation run crashes, it is typically due to one of the following architectural failure modes.

### A. Empty Layers
* **Cause:** The graph builder (pre-solver) applied constraints that were too strict, resulting in a specific beat (`time_index`) having zero valid `BeatState` nodes.
* **Detection:** Handled during problem construction.
* **Fallback Behavior:** The pipeline immediately raises an `SBContractError` ("SparseGraph must contain at least two layers" or similar reachability errors). The run is aborted before wasting compute on matrix math.

### B. Solver Instability
* **Cause:** The iterative proportional fitting (Sinkhorn) within `sb.py` failed to converge. This occurs if the input distributions (`pi0`, `piT`) are heavily disjointed from the edge weights, causing extreme floating-point underflow.
* **Detection:** Tracked via `final_max_delta` against the configured `tolerance`.
* **Fallback Behavior:** If `iterations` hit the `max_iterations` ceiling without dipping below the tolerance threshold, the system raises an `SBSolverError` detailing the non-convergence to prevent generating mathematically unconfident music.

### C. Decode Dead Ends
* **Cause:** The solver converged, but during Maximum A Posteriori (MAP) path extraction, the trace entered a state with zero valid outgoing edges (an absolute probability of 0.0).
* **Detection:** Handled during the backpointer reconstruction phase in `map_bridge_path`.
* **Fallback Behavior:** The system traps the dead end and raises a `ValueError` ("Bridge backpointer reconstruction failed" or "No outgoing bridge transitions").