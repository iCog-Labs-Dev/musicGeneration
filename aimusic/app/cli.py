import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any, Iterable, cast

from aimusic.core.diagnostics import (
    RunManifest,
    SBDiagnostics,
    StructuralDiagnostics,
    TimelineEvent,
)
from aimusic.core.config import DecodeConfig, EDOConfig, MicrotonalRendering, StyleConfig
from aimusic.core.core_types import Score
from aimusic.core.vocab import DEFAULT_GROOVE_FAMILIES, DEFAULT_METER_SIGNATURES
from aimusic.decode import decode_path_to_score
from aimusic.planning.plans import MethodARunConfig, run_method_a
from aimusic.render import render_midi
from aimusic.render.midi_render import TrackInstrumentConfig
from aimusic.theory.edo import EDO

ROLE_TENSION = {
    "hold": 0.20,
    "prep": 0.45,
    "change": 0.65,
    "cad": 0.90,
}


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


def _segment_timeline(values: Iterable[str]) -> list[TimelineEvent]:
    items = tuple(values)
    if not items:
        return []
    events: list[TimelineEvent] = []
    start = 0
    current = items[0]
    for index, label in enumerate(items[1:], start=1):
        if label == current:
            continue
        events.append(TimelineEvent(float(start), float(index), current))
        start = index
        current = label
    events.append(TimelineEvent(float(start), float(len(items)), current))
    return events


def _build_structural_diagnostics(path: tuple[Any, ...], vocabularies: Any) -> StructuralDiagnostics:
    decoded_states = path[:-1] if len(path) > 1 else path
    key_labels = [vocabularies.keys.token_for_id(state.key_id).label for state in decoded_states]
    chord_labels = [vocabularies.chords.token_for_id(state.chord_id).label for state in decoded_states]
    role_labels = [vocabularies.roles.token_for_id(state.role_id).label for state in decoded_states]
    groove_labels = [vocabularies.grooves.token_for_id(state.groove_id).label for state in decoded_states]
    boundaries = [float(index) for index, state in enumerate(decoded_states) if state.boundary_lvl > 0]
    tension_curve = [
        (
            float(index),
            min(
                1.0,
                ROLE_TENSION[vocabularies.roles.token_for_id(state.role_id).label]
                + (0.05 * state.boundary_lvl),
            ),
        )
        for index, state in enumerate(decoded_states)
    ]
    return StructuralDiagnostics(
        key_timeline=_segment_timeline(key_labels),
        chord_timeline=_segment_timeline(chord_labels),
        role_timeline=_segment_timeline(role_labels),
        groove_timeline=_segment_timeline(groove_labels),
        boundaries=boundaries,
        tension_curve=tension_curve,
    )


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
    plan_result = run_method_a(run_config)
    score = decode_path_to_score(
        plan_result.path,
        decode_config=decode_config,
        vocabularies=plan_result.vocabularies,
        edo=args.edo,
        tempo_bpm=args.tempo_bpm,
    )
    structural_stats = _build_structural_diagnostics(plan_result.path, plan_result.vocabularies)
    manifest = RunManifest(
        seed=args.seed,
        config_dump=_json_ready(
            {
                "run_config": run_config,
                "meter": args.meter,
                "groove_family": args.groove_family,
                "tempo_bpm": args.tempo_bpm,
                "output_dir": args.out,
                "track_instruments": _build_track_instruments(args),
            }
        ),
        structural_stats=structural_stats,
        sb_stats=SBDiagnostics.from_solution(plan_result.sb_solution),
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    score_path = out_dir / f"{manifest.run_id}_score.json"
    midi_path = out_dir / f"{manifest.run_id}.mid"
    manifest_path = out_dir / f"{manifest.run_id}_manifest.json"

    with score_path.open("w", encoding="utf-8") as f:
        json.dump(score.to_dict(), f, indent=2)

    render_midi(
        score,
        _build_edo(args),
        str(midi_path),
        track_instruments=_build_track_instruments(args),
    )

    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest.to_dict(), f, indent=2)

    print(f"Generated score JSON: {score_path}")
    print(f"Generated multitrack MIDI: {midi_path}")
    print(f"Generated manifest: {manifest_path}")

def handle_inspect(args: argparse.Namespace) -> None:
    """Handles the 'inspect' CLI command."""
    manifest_path = Path(args.file)
    if not manifest_path.exists():
        print(f"Error: Could not find manifest at {args.file}")
        sys.exit(1)
        
    with open(manifest_path, "r") as f:
        data = json.load(f)
        
    print(f"\n=== Inspection Report for Run: {data.get('run_id')} ===")
    
    # --- SB Math Diagnostics ---
    sb = data.get("sb_stats", {})
    print("\n--- Schrödinger Bridge Health ---")
    status = "🟢 Converged" if sb.get("converged") else "🔴 FAILED"
    print(f"Status:      {status} (in {sb.get('iterations_run')} iterations)")
    print(f"Max Delta:   {sb.get('final_max_delta')}")
    print(f"Entropy:     {sb.get('effective_entropy'):.4f} (Lower = More Confident)")
    print(f"Pruned dead: {sb.get('pruned_nodes')} nodes")
    print(f"Layer sizes: {sb.get('layer_sizes')}")

    # --- Structural Timelines ---
    structure = data.get("structure", {})
    print("\n--- Tension Arc ---")
    for time_val, tension in structure.get("tension_curve", []):
        bar = "█" * int(tension * 20)
        print(f"Beat {time_val:04.1f}: {bar} ({tension})")
    print("=========================================================\n")


def handle_export(args: argparse.Namespace) -> None:
    """Handle the export command by rendering a serialized Score to MIDI."""
    score_path = Path(args.file)
    if not score_path.exists():
        print(f"Error: Could not find score file at {args.file}")
        sys.exit(1)

    with score_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    score = cast(Any, Score).from_dict(data)
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
