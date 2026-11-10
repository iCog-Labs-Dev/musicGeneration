# Audio configuration and setup

Install the lightweight Python stack with `pip install -e ".[audio,dev]"`.
No native FluidSynth, SoundFont, service keys or models are needed for config
loading, preview tests or the wheel smoke test.

## Packaged defaults and profiles

Runtime defaults live in
[`aimusic/audio/resources/audio.default.yaml`](../aimusic/audio/resources/audio.default.yaml),
loaded via `importlib.resources`, not a checkout-relative path.
`preview.yaml` and `listening.yaml` in that directory overlay the defaults:

| Profile | Instrument renderer | Simple fallback |
| --- | --- | --- |
| Unspecified / `preview` | `simple` | permitted for compatible input |
| `listening` | `fluidsynth` | forbidden |

Both keep paid restyle and MIDI-GPT **off**. Renderer selection is separate
from `AIMUSIC_AUDIO_BACKEND=native|m2a`, which selects the orchestrator.
These profiles define effective settings; the native adapter, registry
capability enforcement and CLI SoundFont flag wiring are later implementation
slices. Loading `listening` is not evidence of an instrument-quality render.

```python
from pathlib import Path
from aimusic.audio.config import load_audio_config

preview = load_audio_config()                  # same settings as "preview"
listening = load_audio_config("listening")
custom = load_audio_config(Path("my-audio.yaml"))
settings = custom.to_dict()                    # detached effective settings
```

An explicit YAML file overlays shipped defaults. Existing file-path profiles,
including `config/audio.default.yaml`, remain accepted. Only bare string names
`preview` and `listening` select resources; use `Path("preview")` or
`"./preview"` for a file with that name. Missing files and non-mapping YAML fail.
`AudioConfig.raw` is recursively read-only; conversions retain all settings
and never reopen the original YAML. `source_path` is the resolved explicit
file path, or `None` for packaged profiles. Values must be JSON-like: null,
strings, booleans, finite numbers, lists and string-keyed mappings. YAML sets,
ordered maps/pairs, dates, binary values and recursive aliases are rejected
with their field path rather than retained as shared mutable objects.

## Canonical routing and compatibility

```yaml
render:
  backend: simple
  routing:
    comping: {default: simple, microtonal: fluidsynth}
```

`render.role_map: {comp: fluidsynth}` remains accepted and expands to
`render.routing.comping: {default: fluidsynth, microtonal: fluidsynth}`.
The role alias `comp` normalizes to `comping`. Equivalent or complementary
aliases merge; contradictory assignments for the same role and tuning class
fail with both definitions in the diagnostic. The existing registry consumes
canonical `default` routes and accepts legacy direct dictionaries; tuning-aware
capability dispatch remains a later slice. A profile can override shipped
defaults; conflicting aliases *within that profile* cannot mask one another.
Renderer capability validation still decides whether a requested route can
preserve the input's tuning; a route is not a tuning certificate.

## SoundFont precedence

The audio-owned resolver is ready for later CLI wiring:

```python
from aimusic.audio.config import resolve_soundfont
asset = resolve_soundfont(listening, explicit=Path("/local/approved-bank.sf2"))
```

Precedence is **explicit CLI argument > selected config/profile
`paths.soundfont` > `AIMUSIC_SOUNDFONT`**. A specified invalid path is an error,
never permission to try a lower-precedence asset. The resolver checks that the
selected path is a readable regular file without importing a native library;
actual SoundFont format acceptance belongs to the renderer.

Relative `paths.soundfont` resolves against the explicit YAML file's directory
at load time, surviving later working-directory changes or deletion of the
YAML. Relative CLI/environment paths use the working directory at resolution;
`~` expands to the user's home. Cache/output directories retain their existing
working-directory-relative behavior. With no asset specified, the resolver
returns `None`; the renderer must require an asset for listening.

Python bindings, the native FluidSynth library, and the user-supplied SoundFont
are separate prerequisites for real rendering. Do not download an arbitrary
asset or commit SoundFonts/WAVs. Record the chosen asset's origin, license and
SHA-256 before native acceptance. See [verification lanes](audio-testing.md).

Other groups retain their defaults: `edo`, `groove`, `restyle`, `scoring`,
`search`, `budget`, `mixmaster`, `stages`, and `paths`. Scoring floors are
unchanged; restyle/MIDI-GPT remain opt-in rather than enabled by a profile.

## Offline wheel smoke test

```bash
python -m pytest tests/test_audio_config.py tests/test_audio_analysis.py tests/test_audio_wheel.py -q
```

The wheel test uses the dev build tools (`pip`, `setuptools>=68`, `wheel`),
builds with `--no-build-isolation --no-index`, and installs with
`--no-deps --no-index` into a temporary target. It supplies only the existing
PyYAML dependency offline, removes the copied build source, and loads defaults
and both profiles in a fresh `python -I -S` process outside the checkout.
`PYTHONPATH`, host site-packages and editable-install hooks are excluded.
Missing packaged YAML makes the test fail; no native assets or network are used.
