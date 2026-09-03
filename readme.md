# GTTM + Schrödinger Bridge: A Symbolic Music Generator

This repository implements the design described in the **Software Design Specification: GTTM + Schrödinger Bridge** document. It provides a modular architecture for generating long-form, EDO-generic symbolic music using functional programming principles.

The system is designed for MIDI-oriented symbolic music generation, with future support for richer rendering and optional neural priors.

---

## Abstract

This project defines a modular software architecture for generating long-form symbolic music using a hybrid of:

1. **GTTM-inspired structural energies**
2. **Corpus-based statistical priors**
3. **Optional neural predictive models**
4. **Schrödinger Bridge inference**

The goal is to produce coherent musical trajectories between endpoint passages while maintaining long-range structure, novelty, and musical consistency.

The design targets progressive rock and jazz fusion compositions of approximately **5–15 minutes**. It supports multiple equal divisions of the octave, including **12-EDO** and **19-EDO**, using a single configurable parameter.

The immediate output format is **MIDI**. Audio rendering is a downstream task.
Optional MIDI→audio integration (RenderPackage contract, quarantined
`aimusic.audio`) is documented under [`docs/audio-pipeline.md`](docs/audio-pipeline.md)
and tracked in [`docs/audio-roadmap.md`](docs/audio-roadmap.md).

---

## 1. Scope and Objectives

### 1.1 Primary Objectives

The main goals of this project are to:

- Generate 5–15 minute pieces with multi-level musical structure, including:
  - meter
  - grouping
  - harmonic motion
  - tension arcs
- Ensure that generated music remains novel rather than simply derivative of a training corpus.
- Support two generation plans:
  - **Method A:** Generate start and end passages, then compute a Schrödinger Bridge trajectory between them under a combined prior.
  - **Method B:** Generate start and middle passages, compute a bridge from start to middle, then a second bridge from middle back to start.
- Support both:
  - **Algorithmic mode:** GTTM + Schrödinger Bridge only
  - **Hybrid mode:** GTTM + Schrödinger Bridge with an optional neural predictive model
- Support EDO as a configurable parameter `N`, rather than hard-coding the system to 12-EDO.
- Treat both **12-EDO** and **19-EDO** as first-class use cases.

### 1.2 Non-Goals for the Initial Implementation

The initial implementation does not focus on:

- Direct audio generation
- Full timbral modeling
- Advanced production-level mixing or mastering

---

## 2. Design Principles

### 2.1 Functional Style and Modularity

The implementation should follow a functional-programming-inspired style where practical in Python.

The codebase should:

- Prefer immutable configuration and data objects using `dataclasses` with `frozen=True`.
- Prefer pure functions with minimal hidden side effects.
- Avoid hidden global state.
- Thread randomness explicitly, similar to the style used in JAX.
- Define clear protocol-style interfaces for pluggable components.
- Keep modules separated by responsibility to avoid circular dependencies.

### 2.2 Backend-Agnostic Numerics

Core algorithms should be written against a small backend interface.

The initial backend is **NumPy**, but the design should preserve the possibility of using **JAX** later.

---

## 3. Setup and Installation

This project uses a standard Python virtual environment to manage dependencies such as MIDI generation tools and testing frameworks.

### 3.1 Clone the Repository

First, clone the repository and navigate into the project directory:

```bash
git clone https://github.com/iCog-Labs-Dev/musicGeneration.git
cd musicGeneration
```

### 3.2 Create a Virtual Environment

Create a new Python virtual environment:

```bash
python -m venv venv
```

### 3.3 Activate the Virtual Environment

On **Linux/macOS**:

```bash
source venv/bin/activate
```

On **Windows Command Prompt**:

```cmd
venv\Scripts\activate
```

On **Windows PowerShell**:

```powershell
venv\Scripts\Activate.ps1
```

### 3.4 Install Required Dependencies

This project tracks its dependencies in a `requirements.txt` file.

After activating the virtual environment, install all required libraries by running:

```bash
pip install -r requirements.txt
```

This installs the required packages, including dependencies such as `mido` and `pytest`.

---

## 4. Running Tests

The project uses `pytest` for the test suite.

To run all tests, execute:

```bash
pytest
```

This verifies that:

- Python module paths are configured correctly.
- Functional math utilities behave as expected.
- Planning and routing logic works correctly.
- MIDI exports remain deterministic.

---

## 5. CLI Workflow

The current CLI lives at `aimusic.app.cli` and supports three artifact-oriented commands:

- `generate`: run the current Method A pipeline, decode a `BeatState` path into a multi-track `Score`, export multitrack MIDI, and write a run manifest.
- `export`: render an existing serialized `Score` JSON file to multitrack MIDI.
- `inspect`: print a compact report from a saved run manifest.

### 5.1 Generate a New Score

This command runs the current implementation end to end and writes three files into the output directory:

- `*_score.json`
- `*.mid`
- `*_manifest.json`

Basic example:

```bash
python3 -m aimusic.app.cli generate \
  --seed 11 \
  --beats 8 \
  --meter 4/4 \
  --groove-family straight \
  --tempo-bpm 120 \
  --out ./outputs
```

Example with instrument overrides:

```bash
python3 -m aimusic.app.cli generate \
  --beats 8 \
  --meter 4/4 \
  --groove-family straight \
  --track-program bass=34 \
  --track-program comping=5 \
  --track-program lead=88 \
  --drum-track drums \
  --out ./outputs
```

Useful flags:

- `--sample-path` switches from MAP extraction to sampled bridge-path extraction.
- `--drum-density`, `--bass-density`, `--comping-density`, and `--lead-density` control decode density.
- `--edo`, `--pitch-bend-range`, and `--rendering-method` control MIDI rendering behavior.
- MPE export uses per-note pitch bends; MTS export uses a byte-correct bulk tuning dump.
- Built-in audio previews reproduce MPE pitch bends. MTS previews are disabled with an
  actionable message because preview converters cannot guarantee MTS support; audition
  downloaded MTS files with an MTS-compatible synthesizer.
- `--track-program track=program` overrides the General MIDI program for a symbolic track.
- `--drum-track track` forces a symbolic track onto the percussion channel.

Default symbolic-track mappings:

- `bass -> 33`
- `comping -> 4`
- `lead -> 81`
- `drums -> percussion channel`

### 5.2 Export a Saved Score to MIDI

If you already have a serialized `Score` JSON artifact, you can render it directly:

```bash
python3 -m aimusic.app.cli export ./outputs/example_score.json --out ./outputs/example.mid
```

Export with instrument overrides:

```bash
python3 -m aimusic.app.cli export ./outputs/example_score.json \
  --out ./outputs/example.mid \
  --track-program lead=81 \
  --track-program bass=38 \
  --drum-track drums
```

For score-based export, the CLI defaults to `--base-tuning 0` so decoded score pitch heights map correctly into MIDI note space.

### 5.3 Inspect a Run Manifest

```bash
python3 -m aimusic.app.cli inspect ./outputs/example_manifest.json
```

---

## 6. Conceptual Pipeline

The system is organized as a layered generation pipeline.

A typical run follows this sequence:

```text
Configs + vocabularies + priors
-> endpoint plan, Method A or Method B
-> build sparse layered graph of BeatState candidates
-> solve Schrödinger Bridge on that graph
-> sample or MAP a BeatState trajectory
-> decode BeatState trajectory to multi-track symbolic Score
-> render Score to MIDI using aimusic.render
```

---

## 7. Core Representations

### 7.1 EDO Configuration

Pitch classes are represented in `Z_N`, where `N` is the number of equal divisions of the octave.

Pitch heights are represented as integers measured in EDO steps.

Examples:

- `N = 12` for 12-EDO
- `N = 19` for 19-EDO

### 7.2 Beat-Level Structural State

The beat-level structural state is represented using `BeatState`.

A `BeatState` is a compact token-based representation of the musical state at a beat.

It includes:

```text
meter_id
beat_in_bar
boundary_lvl
key_id
chord_id
role_id
head_id
groove_id
```

### 7.3 Score-Level Representation

The score-level representation uses `NoteEvent`.

A `NoteEvent` represents a single note using:

```text
ton
toff
h
v
e
track
```

Where:

- `ton` is the note onset time.
- `toff` is the note offset time.
- `h` is the pitch height.
- `v` is the velocity.
- `e` is the EDO or tuning-related field.
- `track` identifies the musical track or instrument layer.

---
## 8. Repository Organization

The project is organized around the main `aimusic/` package, with tests kept separately under `tests/`. The codebase is grouped by responsibility so that core data structures, theory utilities, scoring logic, planning logic, rendering, and application entrypoints remain independent and easier to maintain.

```text
musicGeneration/
├── aimusic/
│   ├── __init__.py
│   ├── app/
│   │   ├── __init__.py
│   │   └── main.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── core_types.py
│   │   ├── rng.py
│   │   └── vocab.py
│   ├── planning/
│   │   ├── __init__.py
│   │   ├── candidates.py
│   │   ├── graph.py
│   │   ├── plans.py
│   │   └── sb.py
│   ├── scoring/
│   │   ├── __init__.py
│   │   ├── gttm_features.py
│   │   ├── priors.py
│   │   └── rhythm_features.py
│   ├── theory/
│   │   ├── __init__.py
│   │   ├── edo.py
│   │   └── tonal.py
│   └── render/
│       ├── __init__.py
│       └── midi_render.py
├── tests/
├── requirements.txt
└── README.md
```

| Package / Module  | Responsibility |
| --- | --- |
| `aimusic/` | Main Python package for the symbolic music generation system. |
| `aimusic.app.cli` | CLI entrypoint with `generate`, `export`, and `inspect` subcommands. |
| `aimusic.app.main` | Application and demo entrypoint for running generation workflows. |
| `aimusic.core.config` | Immutable configuration dataclasses for generation, EDO settings, scoring, and planning. |
| `aimusic.core.core_types` | Canonical shared data types such as `BeatState`, `Layer`, `Edge`, `Score`, and related structures. |
| `aimusic.core.vocab` | Token vocabularies for meters, grooves, chords, keys, roles, and other symbolic categories. |
| `aimusic.core.diagnostics` | Run manifests, SB diagnostics, tension curves, and structural diagnostics. |
| `aimusic.core.rng` | Deterministic random-number-generation helpers used by sampling and planning code. |
| `aimusic.theory.edo` | EDO pitch math, pitch-class operations, and microtonal helper functions. |
| `aimusic.theory.tonal` | Tonal-system definitions, chord templates, tonal distances, and harmonic utilities. |
| `aimusic.scoring.gttm_features` | GTTM-inspired feature functions and weighted structural-energy computation. |
| `aimusic.scoring.priors` | Prior model interfaces, including `NullPrior`, placeholder `NeuralPrior`, manifests, and prior scoring utilities. |
| `aimusic.scoring.rhythm_features` | Rhythm-related scoring helpers and compatibility layer over beat-state scoring. |
| `aimusic.planning.candidates` | Candidate proposal and hard-gating logic for valid beat-level states. |
| `aimusic.planning.graph` | Sparse layered-graph construction, layer expansion, edge building, and pruning. |
| `aimusic.planning.plans` | Endpoint generation, section metadata, and orchestration for Method A and future Method B workflows. |
| `aimusic.planning.sb` | Schrödinger Bridge solver, bridge extraction, trajectory sampling, and MAP path extraction. |
| `aimusic.decode` | BeatState path to multi-track symbolic Score decoding (drums, bass, comping, lead). |
| `aimusic.render` | Rendering package for converting symbolic scores into output formats such as MIDI. |
| `aimusic.render.midi_render` | MIDI generation and deterministic MIDI file export, kept separate from planning logic. |
| `tests/` | Test suite for validating core math, planning, scoring, routing, and deterministic MIDI behavior. |
| `requirements.txt` | Project dependency list used to install required Python packages. |
| `README.md` | Project overview, setup instructions, architecture notes, and development checklist. |

---

## 9. Implementation Checklist

The current implementation progress is tracked below.

- [x] Define configs and token vocabularies, starting with 12-EDO and then extending to 19-EDO.
- [x] Implement GTTM feature energies and a simple rule-based tonal distance metric.
- [x] Implement candidate generation and sparse graph building with pruning.
- [x] Implement the Schrödinger Bridge solver on sparse edges using the NumPy backend.
- [ ] Implement Method A plan, then Method B.
- [ ] Implement decoder components for drums, bass, comping, and lead.
- [x] Implement MIDI rendering.
  - [x] 12-EDO direct mapping implemented.
  - [x] 19-EDO MPE rendering and cent-accuracy validation implemented.
  - [x] MTS bulk tuning dump rendering and frequency validation implemented.
- [ ] Add the placeholder `NeuralPrior` seam and artifact contract.
- [ ] Integrate the external neural prior implementation when available.
- [ ] Add section-wise Schrödinger Bridge generation and richer diagnostics.

---

## 10. Notes for Development

When developing this project, remember to:

- Activate the virtual environment before running commands.
- Install dependencies from `requirements.txt`.
- Run tests with `pytest` after making changes.
- Keep new modules responsibility-focused.
- Avoid introducing circular imports.
- Preserve deterministic behavior where possible, especially for sampling and MIDI export.
- Keep EDO logic generic and avoid hard-coding assumptions specific to 12-EDO.
