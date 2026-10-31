# Replace a music-generation component

Use the existing pipeline with a different distance formula, candidate generator, whole-score decoder, or renderer. Supply the replacement as a Python argument. The normal command-line options and UI controls continue to use the existing implementations.

For the change history and verification, see [REQ-10 implementation](req-10-implementation.md).

## Run a complete example

Run this example with the repository installed, including its development dependencies. It changes the distance formula and supplies decoder and renderer wrappers. The wrappers delegate to the existing algorithms, so you can add your own behavior inside the same signatures.

```python
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Mapping, Sequence

from aimusic.core.component_types import RenderResult, SymbolicNote, TrackInstrumentConfig
from aimusic.core.config import DecodeConfig
from aimusic.core.core_types import BeatState, Score
from aimusic.core.protocols import ReportingScoreRenderer, ScoreDecoder, TonalMetric
from aimusic.render.adapters import RenderResultAdapter
from aimusic.core.rng import RNGKey
from aimusic.core.vocab import Vocabularies
from aimusic.decode import decode_path_to_score
from aimusic.planning.plans import render_exact_bridge_demo
from aimusic.render.midi_render import render_midi
from aimusic.theory.edo import EDO
from aimusic.theory.tonal import basic_space_distance, tonal_distance


@dataclass(frozen=True)
class ScaledMetric:

	factor: float = 0.5

	def tonal_distance(self, a_root: int, b_root: int, edo: int = 12) -> float:
		return self.factor * tonal_distance(a_root, b_root, edo)

	def basic_space_distance(
		self, a_root: int, a_quality: str,
		b_root: int, b_quality: str, edo: int = 12,
	) -> float:
		return self.factor * basic_space_distance(
			a_root, a_quality, b_root, b_quality, edo,
		)


def custom_decoder(
	path: Sequence[BeatState], *, key: RNGKey,
	decode_config: DecodeConfig | None = None,
	vocabularies: Vocabularies, edo: int = 12,
	ticks_per_beat: int = 480, tempo_bpm: float = 120.0,
	include_terminal_state: bool = False,
) -> tuple[Score, RNGKey]:
	return decode_path_to_score(
		path, key=key, decode_config=decode_config,
		vocabularies=vocabularies, edo=edo,
		ticks_per_beat=ticks_per_beat, tempo_bpm=tempo_bpm,
		include_terminal_state=include_terminal_state,
	)


def custom_renderer(
	notes: Sequence[SymbolicNote] | Score, edo: EDO,
	output_path: str, ticks_per_beat: int = 480, *,
	track_instruments: Mapping[str, TrackInstrumentConfig] | None = None,
) -> RenderResult:
	return RenderResultAdapter(render_midi)(
		notes, edo, output_path, ticks_per_beat,
		track_instruments=track_instruments,
	)


metric: TonalMetric = ScaledMetric()
decoder: ScoreDecoder = custom_decoder
renderer: ReportingScoreRenderer = custom_renderer

with TemporaryDirectory() as temporary:
	destination = Path(temporary) / "example.mid"
	result = render_exact_bridge_demo(
		start_chord="C:maj", end_chord="G:maj", output_path=str(destination),
		total_beats=4, seed=11, edo=19,
		tonal_metric=metric, decoder=decoder, renderer=renderer,
	)
	assert result.score.note_events
	assert destination.read_bytes().startswith(b"MThd")
	assert result.render_result is not None
	assert result.render_result.output_path == str(destination)
	assert result.render_result.size_bytes == destination.stat().st_size
```

The temporary directory removes the example output when the block ends. For a file you can keep, pass your own destination to `render_exact_bridge_demo`.

The demo retains the legacy `start_chord` and `end_chord` arguments for existing scripts. Method A currently selects endpoints itself, as it did before this change.

## Select components in other workflows

| Entry point | Accepted replacements |
|---|---|
| `build_sparse_graph` | `tonal_metric`, `candidate_generator` |
| `run_method_a` | `tonal_metric`, `candidate_generator` |
| `render_exact_bridge_demo` | All four |
| `aimusic.app.cli.handle_generate` | All four |
| `ui._generate_artifacts` | All four, through the existing internal UI helper |
| `aimusic.app.cli.handle_export` | `renderer` |

Pass `None`, or omit a replacement, to select the existing implementation. The default candidate generator uses the supplied metric automatically. A custom candidate generator owns its own proposal logic. If it needs the same metric, give it that object when constructing it.

For a candidate generator, copy the full signature from `aimusic.planning.protocols.CandidateGenerator`. Return `(CandidateGenerationResult, RNGKey)`. For a typed minimal double, see `EmptyCandidates` in `tests/typecheck_protocols.py`. For actual graph substitution, see `ControlledCandidates` in `tests/test_planning_injection.py`.

An empty candidate result means that the generator found no allowed successors. The graph does not fall back to another generator. The graph still scores edges, enforces retained-edge limits, and restricts the final step to the requested endpoints.

## Follow the contracts

The protocol defines the callable shape. The contract also defines the behavior the caller relies on.

| Component | Required behavior |
|---|---|
| `TonalMetric` | Return a finite, nonnegative real distance from both methods. Honor the supplied EDO. Distances affect candidates, pruning, harmonic scoring, cadence checks, and tension reports. |
| `CandidateGenerator` | Return deduplicated legal `BeatState` successors for the supplied source and time. Preserve tuple fields, source/time diagnostics, count types, and score alignment. Return the next explicit random key. |
| `ScoreDecoder` | Accept a sequence of states and all declared keyword arguments. Return the generated `Score` and next `RNGKey`. Preserve timing and terminal-state conventions. |
| `ScoreRenderer` | Legacy signature: accept both `Score` and symbolic-note sequences, write to the supplied path, honor tuning/instrument settings, close resources, and return `None`. |
| `ReportingScoreRenderer` | The same inputs, returning `RenderResult(output_path, format, size_bytes, edo)` after writing the file. Current workflows require MIDI. |

All renderer entry points accept either signature (`Renderer` is their union). They use `RenderResultAdapter` to obtain a completed result. The adapter calls the supplied implementation once with a fresh temporary destination in the final destination's parent directory. The filename stays the same. It checks that this new file exists, reads its actual byte size, and returns a frozen `RenderResult` with the requested path, `format="midi"`, size, and EDO. If a reporting implementation supplies inconsistent details, the adapter raises `ValueError`. After checking the report, it atomically replaces the final destination and reports that final path. A no-op cannot claim an existing file as a new result. File and renderer errors propagate; temporary output is removed and any existing final file is preserved. The demo and UI artifact result expose this as `.render_result`; existing constructors retain a default of `None`. CLI handlers keep their existing return types and use the reported path for output handling. Manifests do not gain new fields.

Use `RenderResultAdapter()` directly to render with defaults and receive these details. Wrap a legacy renderer with `RenderResultAdapter(legacy_renderer)` to keep its inputs and `None` return unchanged. A missing output raises `FileNotFoundError`; no success result is returned.

The renderer uses the existing project-owned `EDO` helper. It does not take a `mido.MidiFile`. The CLI packages MIDI output, and the UI helper writes score/MIDI/manifest files for preview and download. Both need a valid MIDI file at the supplied path. A double that only records a call cannot complete those workflows.

Custom metric distances bypass the default shared caches. The component can own a cache if needed. Avoid mutable behavior during a run. Cadence scoring retains its existing comparison with a root distance of exactly `1`, so changing the scale can change cadence decisions. This is part of the experiment, not an automatic rescaling step.

Keep settings such as tempo, densities, proposal budgets, and instrument programs in existing configuration values. Supply executable component objects separately. The pipeline does not serialize those objects into a manifest or store a process-wide active component. Record a custom experiment's implementation version and settings separately if you need to reproduce it later.

Create components before calling the workflow. The caller owns expensive setup and cleanup. Default components need no startup or shutdown hooks. Randomness remains explicit through `RNGKey`; do not introduce a hidden global random generator.

Component exceptions propagate. The pipeline does not retry with a default component. Invalid custom distances or candidate results raise `TypeError` or `ValueError`. A renderer failure can leave a score JSON file. The adapter removes staged output and preserves any existing final MIDI file; the CLI does not write a success manifest or package afterward. Other output artifacts are not rolled back.

## Check a replacement

Assign the implementation to its protocol type, as the example does, then run mypy on that file. Check behavior with fixed inputs and known outputs. Finally run a complete workflow to verify integration.

Python annotations do not enforce these promises while the program runs. Mypy checks compatible signatures and declared types before execution. Tests verify results. Runtime checks cover custom distances and candidate outputs where they enter the graph.

Existing imports of `CandidateGenerationResult`, `CandidateRejection`, `SymbolicNote`, and `TrackInstrumentConfig` still work. Their shared definitions now live in `aimusic.core.component_types`. Use that module for new contract-facing code.

The combined CLI/UI test in `tests/test_artifact_injection.py` selects all four replacements together in both supported tunings. It checks changed candidate support, metric-derived diagnostics, decoder output, MIDI, and CLI packaging. See [manual checks before presenting](req-10-manual-checks.md) for the browser and listening checks that automation has not performed.
