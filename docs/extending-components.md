# Replace a music-generation component

Choose a tonal metric, candidate generator, whole-score decoder, or renderer through the shared application factory. CLI generation/export and UI generation use the same resolver. Missing selections keep the existing implementations. Method A and section-wise planning receive the selected metric and generator.

## Select components from a JSON file

```json
{
  "tonal_metric": {
    "factory": "my_music.components:make_metric",
    "options": {"scale": 0.75},
    "public_options": ["scale"]
  },
  "candidate_generator": {"factory": "my_music.components:make_candidates"},
  "decoder": {"factory": "my_music.components:make_decoder"},
  "renderer": {"factory": "my_music.components:make_renderer"}
}
```

The named module must be installed or on Python's import path. Each reference is `module:export`, where the export is a constructor function or class. The server operator controls the file: importing a factory executes trusted Python code. Do not accept arbitrary factory files or names from browser clients. There is no plugin download service, package discovery or mutable global registry.

```powershell
python -m aimusic.app.cli generate --seed 11 --beats 4 --components components.json --out outputs
python -m aimusic.app.cli export score.json --components components.json --out score.mid
python ui.py --components components.json
```

The UI reads the operator's configuration at startup and binds it to its callback. Each generation resolves fresh objects; browser fields continue controlling the ordinary music settings. The CLI remains single-pass by default; the UI retains the upstream section-wise default. Set `section_wise=False` when calling the UI helper for a single-pass experiment.

Unknown roles or fields, malformed factory references, non-JSON/non-finite options, missing constructors and incompatible call shapes fail before planning. Configuration accepts the four keys shown above. Options are JSON keyword arguments. The resolver copies nested values before construction and keeps a separate provenance snapshot, so a factory cannot accidentally consume a saved list and change the next UI run.

Manifests and RenderPackages are shareable output. Constructor options stay private by default. `public_options` explicitly names the option values safe to copy into those artifacts; omit it to publish none. Only list shareable settings such as `scale`. Never list credentials, access tokens, signed URLs or an object containing secrets. Providers can also read credentials from their environment. Provenance records the number of unpublished options without their values, so omitted settings are not mistaken for a fully reproducible configuration.

## Follow constructor conventions

| Role | Constructor inputs | Required component |
|---|---|---|
| `tonal_metric` | Configured keyword options | `TonalMetric` |
| `candidate_generator` | Configured options plus reserved `tonal_metric` | `CandidateGenerator` |
| `decoder` | Configured keyword options | `ScoreDecoder` |
| `renderer` | Configured options plus reserved `midi_settings` | `ScoreRenderer` |

A candidate constructor receives the already selected metric. A renderer constructor receives the effective `MidiRenderSettings` for the current CLI/UI MIDI workflow, including tuning and instruments. Bind those settings in the implementation, not in the general renderer protocol. Options cannot replace reserved injected dependencies. Constructors own their validation and can reject incompatible settings. Construction failures are reported with the role and original cause.

A supplied or configured built-in `MidiRenderer` must match the run's tuning, rendering method and instruments. The factory checks this even through its own artifact wrappers, before planning. `MidiRenderSettings` copies its instrument mapping into a read-only mapping; constructors cannot change the run by editing it in place. Use `dataclasses.replace` to create deliberately different settings, or `to_dict()` for public metadata. Custom opaque renderers remain responsible for honoring their supplied backend settings.

## Run a complete example

Save the following as `extension_example.py`. The example selects all four constructors, writes real 19-EDO MIDI and checks the returned file details. Default decoder and MIDI algorithms remain unchanged; the metric changes their structural context.

```python
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from aimusic.app.components import ComponentConfig, ComponentSpec
from aimusic.app.demos import render_exact_bridge_demo
from aimusic.core.protocols import ScoreDecoder, ScoreRenderer, TonalMetric
from aimusic.decode import decode_path_to_score
from aimusic.planning.candidates import MetricBoundCandidates
from aimusic.planning.protocols import CandidateGenerator
from aimusic.render.midi_adapter import MidiRenderer, MidiRenderSettings
from aimusic.theory.tonal import basic_space_distance, tonal_distance


@dataclass(frozen=True)
class ScaledMetric:
    scale: float

    def tonal_distance(self, a_root: int, b_root: int, edo: int = 12) -> float:
        return self.scale * tonal_distance(a_root, b_root, edo)

    def basic_space_distance(
        self, a_root: int, a_quality: str,
        b_root: int, b_quality: str, edo: int = 12,
    ) -> float:
        return self.scale * basic_space_distance(a_root, a_quality, b_root, b_quality, edo)


def make_metric(*, scale: float) -> TonalMetric:
    if not 0 < scale <= 1:
        raise ValueError("scale must be greater than zero and at most one")
    return ScaledMetric(scale)


def make_candidates(*, tonal_metric: TonalMetric) -> CandidateGenerator:
    return MetricBoundCandidates(tonal_metric)


def make_decoder() -> ScoreDecoder:
    return decode_path_to_score


def make_renderer(*, midi_settings: MidiRenderSettings) -> ScoreRenderer:
    return MidiRenderer(midi_settings)


def main() -> None:
    config = ComponentConfig(
        tonal_metric=ComponentSpec("extension_example:make_metric", {"scale": 0.75}, public_options=("scale",)),
        candidate_generator=ComponentSpec("extension_example:make_candidates"),
        decoder=ComponentSpec("extension_example:make_decoder"),
        renderer=ComponentSpec("extension_example:make_renderer"),
    )
    with TemporaryDirectory() as folder:
        destination = Path(folder) / "example.mid"
        result = render_exact_bridge_demo(
            start_chord="C:maj", end_chord="G:maj", total_beats=4,
            output_path=str(destination), seed=11, edo=19, component_config=config,
        )
        assert destination.read_bytes().startswith(b"MThd")
        assert result.render_result is not None
        assert result.render_result.output_path == str(destination)
        assert result.render_result.size_bytes == destination.stat().st_size
        assert result.render_result.format == "midi"


if __name__ == "__main__":
    main()
```

The demo retains the legacy start/end chord, meter and groove arguments, but Method A selects its own endpoints as before. Its application implementation lives in `aimusic.app.demos`. The old `aimusic.planning.plans.render_exact_bridge_demo` delegates there for compatibility and does not construct decoder or renderer defaults.

## Understand the contracts

The protocol specifies callable inputs and outputs. The contract also defines the behavior required by the caller.

| Component | Input/output and required behavior |
|---|---|
| `TonalMetric` | Two root/chord distance methods with integer EDO. Return finite, nonnegative real distances. Selected distances reach proposals, pruning, harmonic scoring, cadence and tension reports. |
| `CandidateGenerator` | Existing source/time/configuration/vocabulary/prior arguments plus explicit random key. Return `(CandidateGenerationResult, RNGKey)` with legal successors and consistent diagnostics. |
| `ScoreDecoder` | Existing whole-path callable and keyword settings. Return `(Score, RNGKey)`, preserving timing and terminal-state conventions. |
| `ScoreRenderer` | `(score: Score, request: RenderRequest) -> RenderResult`. This is the only renderer protocol. No MIDI objects, tuning or instrument arguments appear here. |

`RenderRequest` contains a nonblank string `output_path` and `format`. `RenderResult` contains their corresponding values and a nonnegative integer `size_bytes`. It has no EDO field. A valid empty artifact can have size zero.

`RenderResultAdapter(renderer)` is format-neutral. It supplies a fresh temporary path in the destination parent with the same filename, requires a regular file at that path, checks the returned type/path/format/actual size, then atomically replaces the final destination and returns its path. The parent directory must exist. Errors remove staged output and preserve any existing final artifact. Directories and symbolic links are not accepted as completed files. This is file publication, not an output-quality or sandbox guarantee.

`MidiRenderer(MidiRenderSettings(...))` owns MIDI tuning and instrument data and accepts the `midi` format. It delegates to the unchanged `render_midi` function. Existing direct `render_midi(Score | Sequence[SymbolicNote], EDO, path, ...) -> None` calls still work. The earlier PR-only `ReportingScoreRenderer` and renderer union have been removed. Update experimental renderer objects to the single neutral signature.

A custom text/JSON/music-notation renderer can implement the same neutral protocol and use the general adapter directly. Current CLI packages and UI preview still require valid MIDI. They do not automatically turn arbitrary formats into playable MIDI or audio.

## Keep candidate diagnostics consistent

Shared classes validate their fields when constructed. `CandidateRejection` requires domain states, nonnegative integer time and a nonblank reason. A candidate result requires tuple fields, unique accepted states, distinct accepted/rejected outcomes, matching rejection source/time, and these counters:

- `proposed_count >= unique_count`.
- `unique_count == legal_count + rejected_count`.
- Scores are empty or align with accepted states; every supplied score is finite and nonboolean.

The graph revalidates external results and checks the requested source/time, returned random key, known vocabulary IDs and transition legality. Empty results stay empty; there is no fallback generator. The internal endpoint check now records how many endpoint states it evaluated while retaining the existing zero raw-generator-proposal diagnostic at the final step.

`proposal_budget` bounds raw proposals. `d_max` bounds retained scored edges. If a graph call supplies both `d_max` and an SB configuration, they must agree. Without an SB configuration, the supplied retained limit is used to construct one.

`SymbolicNote` permits signed integer pitch heights, finite nonnegative starts and later ends, and MIDI-range integer controls. `TrackInstrumentConfig` validates optional programs and boolean drum flags. Existing imports from the candidate and MIDI modules remain aliases of the shared definitions.

## Record what actually ran

CLI and UI manifests add `config.components`, containing each role's selection source, factory reference, actual implementation identity, explicitly public option values and an unpublished-option count. Renderer provenance includes the supplied MIDI construction settings and atomic wrapper identity. Existing top-level manifest fields and score/MIDI formats remain unchanged.

Programmatic callers can pass objects to the existing injection arguments or supply parsed `ComponentConfig`. Selecting a factory and an object for the same role is an error. Runtime-object provenance says `runtime_override`, records its actual identity, and marks unknown constructor options as null. Opaque objects are not fully reproducible merely because their class name is recorded. Keep their setup/version information separately.

Create expensive resources in the provider's constructor and manage cleanup in the embedding application. No process-wide active component or startup/shutdown plugin framework was added. Keep random state explicit through `RNGKey`; do not introduce hidden global randomness. Custom metric distances bypass the default shared caches. Cadence retains its existing comparison with root distance exactly one, so rescaling can change cadence decisions.

Component exceptions propagate. A renderer failure can leave the already-written score JSON, but the CLI does not emit a new success manifest or package afterward. Other output artifacts are not rolled back. Python annotations and runtime structural/signature checks do not prove behavior or return annotations. Positive/negative mypy fixtures check tested implementations, and behavior/integration tests verify real outputs and errors.

For the verification record and review coverage, see [REQ-10 implementation](req-10-implementation.md). Existing browser/playback acceptance is recorded in [manual checks](req-10-manual-checks.md).
