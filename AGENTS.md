# AGENTS.md

Guidance for humans and coding agents working in this repository.

## Project

Symbolic music generation (`aimusic`): GTTM-inspired energies + Schrödinger Bridge
→ multi-track Score → MIDI. Optional MIDI→audio pipeline is quarantined under
`aimusic.audio` (see [docs/audio-pipeline.md](docs/audio-pipeline.md)).

## Commands

- Install: `pip install -e ".[dev]"`
- Tests (core): `python -m unittest discover -s tests -p 'test_*.py'`
- Lint: `flake8 aimusic tests ui.py`
- Typecheck: `mypy aimusic tests ui.py`
- Generate: `python -m aimusic.app.cli generate --seed 11 --beats 4 --out ./outputs`

## Code conventions

- Frozen dataclasses; pure functions; explicit RNG threading
- EDO-generic code; 12-EDO and 19-EDO are first-class
- Named modules by responsibility; avoid circular imports

## Audio pipeline (optional `[audio]`)

- Symbolic core stays pure (`numpy` + `mido`). Production audio is quarantined
  under `aimusic.audio` + extra `[audio]`.
- Systems meet only at **RenderPackage** (`run_<hash>/`). See
  [docs/render-package-contract.md](docs/render-package-contract.md).
- CLI audio subcommands must **lazy-import** (`importlib`); never static-import
  `aimusic.audio` from core modules.
- Default backend (once present): native orchestrator; override with
  `AIMUSIC_AUDIO_BACKEND=m2a|native`.
- Paid restyle off by default; tests use mocks / vcrpy. MIDI-GPT default **off**
  (CC-BY-NC).
- Do not implement B4 (`E_host` in planning) unless explicitly tasked.
- Quality claims must cite scorer metrics (rhythm/notes/tuning/CLAP/FAD floors).

## Read first (audio sessions)

1. [docs/audio-pipeline.md](docs/audio-pipeline.md)
2. [DECISIONS.md](DECISIONS.md), [docs/decisions/](docs/decisions/)
3. One P0/P1 item from [docs/audio-next-steps.md](docs/audio-next-steps.md)

## Verify

```bash
# core / architecture (no heavy audio)
python -m pytest tests/test_architecture.py tests/test_render_package.py -q
python -m unittest discover -s tests -p 'test_*.py'
```

## Boundaries

- Never commit `.env`, secrets, or large model weights
- Do not open org PRs to `iCog-Labs-Dev/musicGeneration` unless the user asks
- Do not add git submodules of `midi2audio_generative`
