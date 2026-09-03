import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any

from aimusic.core.diagnostics import (
    ManifestValidationError,
    RunManifest,
    build_run_manifest,
)
from aimusic.core.config import (
    DecodeConfig,
    EDOConfig,
    MicrotonalRendering,
    StyleConfig,
)
from aimusic.core.core_types import Score, ScoreValidationError
from aimusic.core.vocab import DEFAULT_GROOVE_FAMILIES, DEFAULT_METER_SIGNATURES
from aimusic.decode import decode_path_to_score
from aimusic.planning.plans import MethodARunConfig, run_method_a
from aimusic.render import TrackInstrumentConfig, render_midi
from aimusic.render.package import write_render_package
from aimusic.scoring.tension import (
    compare_tension_curves,
    realized_tension_curve,
    target_tension_curve,
)
from aimusic.theory.edo import EDO

def _json_ready(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _json_ready(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_json_ready(item) for item in value]
    if hasattr(value, "value"):
        enum_value = getattr(value, "value", None)
        if isinstance(enum_value, str):
            return enum_value
    return value


def _build_edo(args: argparse.Namespace) -> EDO:
    return EDO(
        EDOConfig(
            n=args.edo,
            base_tuning=args.base_tuning,
            pitch_bend_range=args.pitch_bend_range,
            microtonal_rendering_method=MicrotonalRendering[args.rendering_method],
        )
    )


def _parse_track_program(value: str) -> tuple[str, int]:
    track_name, separator, program_text = value.partition("=")
    if not separator or not track_name.strip():
        raise argparse.ArgumentTypeError("track program must be in the form track=program.")
    try:
        program = int(program_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("program must be an integer.") from exc
    if program < 0 or program > 127:
        raise argparse.ArgumentTypeError("program must be in MIDI range 0..127.")
    return track_name.strip(), program


def _build_track_instruments(args: argparse.Namespace) -> dict[str, TrackInstrumentConfig]:
    instruments: dict[str, TrackInstrumentConfig] = {}
    for track_name, program in args.track_program:
        instruments[track_name.strip().lower()] = TrackInstrumentConfig(program=program)
    for track_name in args.drum_track:
        normalized = track_name.strip().lower()
        existing = instruments.get(normalized)
        instruments[normalized] = TrackInstrumentConfig(
            program=None if existing is None else existing.program,
            is_drum=True,
        )
    return instruments


def handle_generate(args: argparse.Namespace) -> None:
    """Run the current Method A pipeline and export score, MIDI, and manifest artifacts."""
    style_config = StyleConfig(
        allowed_meters=(args.meter,),
        groove_families=(args.groove_family,),
    )
    decode_config = DecodeConfig(
        subbeats_per_beat=args.subbeats_per_beat,
        drum_density=args.drum_density,
        bass_density=args.bass_density,
        comping_density=args.comping_density,
        lead_density=args.lead_density,
    )
    run_config = MethodARunConfig(
        total_beats=args.beats,
        seed=args.seed,
        use_sampling=args.sample_path,
        style_config=style_config,
        decode_config=decode_config,
        edo=args.edo,
    )
    prior = None
    prior_bundle = getattr(args, "prior_bundle", None)
    if prior_bundle:
        try:
            from aimusic.ml.inference import load_trained_neural_prior
        except ImportError:
            print(
                "Error: --prior-bundle requires optional ML dependencies.\n"
                "Install with: pip install -e '.[ml]'",
                file=sys.stderr,
            )
            sys.exit(1)
        prior = load_trained_neural_prior(prior_bundle)
    plan_result = run_method_a(run_config, prior=prior)
    score = decode_path_to_score(
        plan_result.path,
        decode_config=decode_config,
        vocabularies=plan_result.vocabularies,
        edo=args.edo,
        tempo_bpm=args.tempo_bpm,
    )
    manifest = build_run_manifest(
        plan_result,
        seed=args.seed,
        config_dump=_json_ready(
            {
                "run_config": run_config,
                "meter": args.meter,
                "groove_family": args.groove_family,
                "tempo_bpm": args.tempo_bpm,
                "output_dir": args.out,
                "pitch_bend_range": args.pitch_bend_range,
                "rendering_method": args.rendering_method,
                "track_instruments": _build_track_instruments(args),
            }
        ),
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    score_path = out_dir / f"{manifest.run_id}_score.json"
    midi_path = out_dir / f"{manifest.run_id}.mid"
    manifest_path = out_dir / f"{manifest.run_id}_manifest.json"
    track_instruments = _build_track_instruments(args)

    with score_path.open("w", encoding="utf-8") as f:
        json.dump(score.to_dict(), f, indent=2)

    render_midi(
        score,
        _build_edo(args),
        str(midi_path),
        track_instruments=track_instruments,
    )

    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest.to_dict(), f, indent=2)

    track_programs = {
        name: (None if cfg.is_drum else cfg.program)
        for name, cfg in track_instruments.items()
    }
    package = write_render_package(
        out_dir,
        score=score,
        midi_path=midi_path,
        manifest=manifest,
        path=plan_result.path,
        vocabularies=plan_result.vocabularies,
        edo=args.edo,
        base_tuning=args.base_tuning,
        pitch_bend_range=args.pitch_bend_range,
        rendering_method=args.rendering_method,
        track_programs=track_programs,
        run_id=manifest.run_id,
    )

    print(f"Generated score JSON: {score_path}")
    print(f"Generated multitrack MIDI: {midi_path}")
    print(f"Generated manifest: {manifest_path}")
    print(f"Generated RenderPackage: {package.root}")

def _load_json_file(path: Path, *, kind: str) -> Any:
    """Load a JSON file, converting missing/malformed files into actionable exits."""
    if not path.exists():
        print(f"Error: Could not find {kind} at {path}")
        sys.exit(1)
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        print(f"Error: {path} is not valid JSON ({exc}).")
        sys.exit(1)
    if not isinstance(data, dict):
        print(f"Error: {path} must contain a JSON object at the top level, got {type(data).__name__}.")
        sys.exit(1)
    return data


def _print_timeline(title: str, events: list) -> None:
    """Print a structural timeline as its labeled segments, not just a count."""
    print(f"\n--- {title} ---")
    if not events:
        print("  (no segments)")
        return
    for event in events:
        start = event.get("start_time")
        end = event.get("end_time")
        label = event.get("label")
        print(f"  {start:>5.1f} -> {end:<5.1f} : {label}")


def handle_inspect(args: argparse.Namespace) -> None:
    """Handles the 'inspect' CLI command."""
    manifest_path = Path(args.file)
    data = _load_json_file(manifest_path, kind="manifest")

    try:
        manifest = RunManifest.from_dict(data)
    except ManifestValidationError as exc:
        print(f"Error: {manifest_path} is not a valid run manifest ({exc}).")
        sys.exit(1)

    print(f"\n=== Inspection Report for Run: {manifest.run_id} ===")
    print(f"Schema Version: {manifest.schema_version}")
    if manifest.migration_warnings:
        print("\n--- Legacy Migration Warnings ---")
        for warning in manifest.migration_warnings:
            print(f"- {warning}")

    graph = manifest.graph_stats
    print("\n--- Graph Candidate Flow ---")
    print(f"Layer sizes: {graph.layer_sizes}")
    print("Layer  Proposed  Legal  Scored  Duplicates  Candidate-pruned  Unique states  State-pruned  Retained")
    for layer in graph.per_layer_stats:
        print(
            f"{layer.time_index:>5}  {layer.proposed:>8}  {layer.legal:>5}  "
            f"{layer.scored:>6}  {layer.duplicate_proposals:>10}  "
            f"{layer.candidate_pruned:>16}  {layer.scored_unique_states:>13}  "
            f"{layer.pruned:>12}  {layer.retained:>8}"
        )
    print(
        f"Totals: proposed={graph.total_proposed}, legal={graph.total_legal}, "
        f"scored={graph.total_scored}, duplicates={graph.total_duplicate_proposals}, "
        f"candidate-pruned={graph.total_candidate_pruned}, "
        f"unique-states={graph.total_scored_unique_states}, "
        f"state-pruned={graph.total_pruned}, retained={graph.total_retained}"
    )
    print("\n--- Graph Edge Flow ---")
    print("Layer  Scored  Edge-pruned  Retained")
    for layer in graph.per_layer_stats:
        print(
            f"{layer.time_index:>5}  {layer.scored_edges:>6}  "
            f"{layer.pruned_edges:>11}  {layer.retained_edges:>8}"
        )
    if graph.rejection_summary:
        print(f"Rejection reasons: {graph.rejection_summary}")
    if graph.pruning_summary:
        print(f"Pruning reasons: {graph.pruning_summary}")

    endpoints = manifest.endpoint_stats
    print("\n--- Endpoint Support ---")
    print(
        f"Original support: pi0={endpoints.original_pi0_support_size}, "
        f"piT={endpoints.original_piT_support_size}"
    )
    print(
        f"Solver support:   pi0={endpoints.solver_pi0_support_size}, "
        f"piT={endpoints.solver_piT_support_size}"
    )
    print(
        f"Unreachable mass: pi0={endpoints.unreachable_pi0_mass:.6f}, "
        f"piT={endpoints.unreachable_piT_mass:.6f}"
    )

    # --- SB Math Diagnostics ---
    sb = manifest.sb_stats
    print("\n--- Schrödinger Bridge Health ---")
    status = "Converged" if sb.converged else "FAILED"
    print(f"Status:             {status} (in {sb.iterations_run} iterations)")
    print(f"Max Delta:          {sb.final_max_delta}")
    print(f"Residual history:   {sb.residual_history}")
    print(f"Entropy by layer:   {sb.layer_entropies}")
    print(f"Average entropy:    {sb.effective_entropy:.4f}")
    print(f"Disconnected nodes: {sb.disconnected_nodes}")

    path_stats = manifest.path_stats
    score = "unavailable" if path_stats.path_score is None else f"{path_stats.path_score:.6f}"
    print("\n--- Final Path Selection ---")
    print(f"Mode: {path_stats.path_mode}")
    print(f"Score: {score}")
    print(f"Transitions: {path_stats.path_transition_count}")

    # --- Structural Timelines ---
    structure = manifest.structural_stats.to_dict()
    _print_timeline("Key Timeline", structure.get("key_timeline", []))
    _print_timeline("Chord Timeline", structure.get("chord_timeline", []))
    _print_timeline("Role Timeline", structure.get("role_timeline", []))
    _print_timeline("Groove Timeline", structure.get("groove_timeline", []))

    print("\n--- Boundaries ---")
    boundaries = structure.get("boundaries", [])
    if boundaries:
        print("  " + ", ".join(f"{beat:.1f}" for beat in boundaries))
    else:
        print("  (none)")

    print("\n--- Tension Arc ---")
    for time_val, tension in structure.get("tension_curve", []):
        bar = "#" * int(tension * 20)
        print(f"Beat {time_val:04.1f}: {bar} ({tension:.3f})")
    print("=========================================================\n")


def handle_export(args: argparse.Namespace) -> None:
    """Handle the export command by rendering a serialized Score to MIDI."""
    score_path = Path(args.file)
    data = _load_json_file(score_path, kind="score file")

    try:
        score = Score.from_dict(data)
    except ScoreValidationError as exc:
        print(f"Error: {score_path} is not a valid score file ({exc}).")
        sys.exit(1)

    edo = _build_edo(args)
    output_path = Path(args.out) if args.out else score_path.with_suffix(".mid")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    render_midi(
        score,
        edo,
        str(output_path),
        track_instruments=_build_track_instruments(args),
    )
    print(f"Exported multitrack MIDI to: {output_path}")

def main() -> None:
    parser = argparse.ArgumentParser(description="GTTM + SB Symbolic Music Generator")
    subparsers = parser.add_subparsers(dest="command", required=True)

    gen_parser = subparsers.add_parser("generate", help="Generate a new score")
    gen_parser.add_argument("--seed", type=int, default=42)
    gen_parser.add_argument("--beats", type=int, default=8)
    gen_parser.add_argument("--edo", type=int, default=12)
    gen_parser.add_argument("--meter", choices=DEFAULT_METER_SIGNATURES, default="4/4")
    gen_parser.add_argument("--groove-family", choices=DEFAULT_GROOVE_FAMILIES, default="straight")
    gen_parser.add_argument("--tempo-bpm", type=float, default=120.0)
    gen_parser.add_argument("--sample-path", action="store_true")
    gen_parser.add_argument("--subbeats-per-beat", type=int, default=4)
    gen_parser.add_argument("--drum-density", type=float, default=0.75)
    gen_parser.add_argument("--bass-density", type=float, default=0.60)
    gen_parser.add_argument("--comping-density", type=float, default=0.55)
    gen_parser.add_argument("--lead-density", type=float, default=0.45)
    gen_parser.add_argument("--base-tuning", type=int, default=0)
    gen_parser.add_argument("--pitch-bend-range", type=int, default=2)
    gen_parser.add_argument(
        "--rendering-method",
        choices=[method.name for method in MicrotonalRendering],
        default=MicrotonalRendering.MPE.name,
    )
    gen_parser.add_argument(
        "--track-program",
        action="append",
        type=_parse_track_program,
        default=[],
        help="Override a symbolic track's GM program using track=program; repeatable.",
    )
    gen_parser.add_argument(
        "--drum-track",
        action="append",
        default=[],
        help="Treat the named symbolic track as percussion; repeatable.",
    )
    gen_parser.add_argument("--out", type=str, default="./outputs")
    gen_parser.add_argument(
        "--prior-bundle",
        type=str,
        default=None,
        help="Path to a trained prior artifact bundle (requires optional [ml] extra).",
    )
    gen_parser.set_defaults(func=handle_generate)

    ins_parser = subparsers.add_parser("inspect", help="Inspect diagnostics")
    ins_parser.add_argument("file", type=str)
    ins_parser.set_defaults(func=handle_inspect)


    exp_parser = subparsers.add_parser("export", help="Export a generated score to MIDI")
    exp_parser.add_argument("file", type=str, help="Path to the score data")
    exp_parser.add_argument("--out", type=str, default=None, help="Output MIDI path")
    exp_parser.add_argument("--edo", type=int, default=12, help="EDO division for rendering")
    exp_parser.add_argument(
        "--base-tuning",
        type=int,
        default=0,
        help="Base MIDI note used by the EDO converter",
    )
    exp_parser.add_argument(
        "--pitch-bend-range",
        type=int,
        default=2,
        help="Pitch-bend range in semitones for MPE rendering",
    )
    exp_parser.add_argument(
        "--rendering-method",
        choices=[method.name for method in MicrotonalRendering],
        default=MicrotonalRendering.MPE.name,
        help="Microtonal MIDI rendering method",
    )
    exp_parser.add_argument(
        "--track-program",
        action="append",
        type=_parse_track_program,
        default=[],
        help="Override a symbolic track's GM program using track=program; repeatable.",
    )
    exp_parser.add_argument(
        "--drum-track",
        action="append",
        default=[],
        help="Treat the named symbolic track as percussion; repeatable.",
    )
    exp_parser.set_defaults(func=handle_export)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
