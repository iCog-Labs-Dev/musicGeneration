"""Deterministic groove-spec applicator (PDF §3.1 / M1)."""

from __future__ import annotations

import io
import json
import math
from bisect import bisect_right
from pathlib import Path
from fractions import Fraction
from typing import Any, Mapping

import mido
import pretty_midi as pm

from aimusic.audio.groove.spec import GrooveSpec, parse_groove_spec
from aimusic.audio.manifest import StageManifest, content_hash_files
from aimusic.audio.groove.publication import publish_bundle
from aimusic.audio.midi_events import MidiEvents, NoteTiming, add_ghosts, rational, retime

# Fixed seed for deterministic jitter when jitter_sd is set.
_JITTER_SEED = 0x4D3241


def _role_for_instrument(instrument: pm.Instrument) -> str:
    name = (instrument.name or "").strip().lower()
    for role in ("drums", "bass", "lead", "comp", "pads", "ornament", "piano"):
        if role in name:
            return role
    if instrument.is_drum:
        return "drums"
    return "comp"


def _seconds_per_beat(midi: pm.PrettyMIDI) -> float:
    if hasattr(midi, "get_tempo_changes"):
        _times, tempos = midi.get_tempo_changes()
        if len(tempos):
            return 60.0 / float(tempos[0])
    return 0.5


def _position_16(start_sec: float, spb: float) -> int:
    """16th-note grid index within the beat (0..15 repeating across the piece)."""
    if spb <= 0:
        return 0
    sixteenths = start_sec / (spb / 4.0)
    return math.floor(sixteenths + 1e-9) % 16


def _swing_offset_sec(start_sec: float, spb: float, ratio: float, subdivision: int) -> float:
    """Delay the off-beat of each swing pair. ratio=0.5 → 0 delay; 2/3 → triplet feel."""
    if ratio <= 0.5 or subdivision <= 0:
        return 0.0
    pair_sec = spb * (8.0 / subdivision)
    if pair_sec <= 0:
        return 0.0
    pos_in_pair = start_sec % pair_sec
    half = pair_sec * 0.5
    if pos_in_pair >= half - 1e-9:
        return (ratio - 0.5) * pair_sec
    return 0.0


def _microtiming_ms_for_role(
    role: str,
    start_sec: float,
    spb: float,
    micro: Any,
    note_index: int,
) -> float:
    if micro is None:
        return 0.0
    role_spec = getattr(micro, role, None)
    if role_spec is None:
        return 0.0
    offset_ms = float(role_spec.global_offset or 0.0)
    if role_spec.by_position_16:
        pos = _position_16(start_sec, spb)
        arr = role_spec.by_position_16
        offset_ms += float(arr[pos % len(arr)])
    if role_spec.jitter_sd:
        u = math.sin((_JITTER_SEED + note_index) * 12.9898) * 43758.5453
        frac = u - math.floor(u)
        offset_ms += (frac * 2.0 - 1.0) * float(role_spec.jitter_sd)
    return offset_ms


def _velocity_scale(
    role: str,
    start_sec: float,
    spb: float,
    velocity_spec: Any,
    depth: float = 1.0,
) -> float:
    if velocity_spec is None or depth <= 0:
        return 1.0
    scale = 1.0
    if velocity_spec.accent_map_16:
        pos = _position_16(start_sec, spb)
        arr = velocity_spec.accent_map_16
        scale *= float(arr[pos % len(arr)])
    if velocity_spec.phrase_arc and velocity_spec.phrase_arc.depth:
        phase = (start_sec / (spb * 4.0)) % 1.0
        arc = math.sin(phase * math.pi)
        d = float(velocity_spec.phrase_arc.depth) * depth
        scale *= 1.0 + d * (arc - 0.5) * 2.0 * 0.15
    return scale


def _analysis_roles(midi: MidiEvents, analysis: dict[str, Any] | None) -> dict[int, str] | None:
    """Admit a complete source-track mapping; never treat instrument indices as SMF IDs."""
    if analysis is None:
        return None
    rows = analysis.get("tracks")
    if not isinstance(rows, list):
        raise ValueError("Groove analysis roles require a tracks list with source_track_index")
    roles: dict[int, str] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Groove analysis role must be a mapping with source_track_index")
        index, role = row.get("source_track_index"), row.get("role")
        if type(index) is not int or not 0 <= index < midi.track_count or index in roles:
            raise ValueError("Invalid/duplicate source_track_index in groove roles; reconcile or re-analyze first")
        if role not in ("drums", "bass", "lead", "comp", "comping", "pads", "ornament", "piano", "mixed"):
            raise ValueError(f"Invalid groove role for source_track_index {index}: {role!r}")
        roles[index] = "comping" if role == "comp" else role
    if set(roles) != {note.track for note in midi.notes}:
        raise ValueError("Groove roles must map every note-bearing source_track_index exactly once")
    return roles


def _apply_to_events(
    midi: MidiEvents,
    spec: GrooveSpec,
    *,
    light_cleanup: bool = False,
    roles: Mapping[int, str] | None = None,
) -> tuple[MidiEvents, dict[str, Any]]:
    """Transform source events in musical time, then validate channel state."""
    tempo = midi.tempo_map
    depth = 0.0 if light_cleanup else 1.0
    names = {e.track: e.message.name.lower() for e in midi.events
             if e.message.type == "track_name"}
    meters = [(0, 4, 4)] + [(e.tick, e.message.numerator, e.message.denominator)
                            for e in midi.events if e.message.type == "time_signature"]
    meter_ticks = [item[0] for item in meters]
    targets = {}
    ghosts = []
    note_diffs: list[dict[str, Any]] = []
    for counter, note in enumerate(sorted(midi.notes, key=lambda n: (n.track, n.on)), 1):
        name = names.get(note.track, "")
        role = (roles[note.track] if roles is not None else
                next((r for r in ("drums", "bass", "lead", "comp", "pads", "ornament", "piano")
                      if r in name), "drums" if note.channel == 9 else "comp"))
        # The analysis/report vocabulary is canonical; the GrooveSpec predates it.
        groove_role = "comp" if role == "comping" else role
        beat = note.start / midi.ticks_per_beat
        start_seconds = tempo.seconds(note.start)
        offset = Fraction(0)
        if depth and spec.swing is not None and groove_role in (spec.swing.applies_to or []):
            pair_ticks = Fraction(midi.ticks_per_beat * 8, spec.swing.subdivision)
            if note.start % pair_ticks >= pair_ticks / 2:
                swing_ticks = (rational(spec.swing.ratio) - Fraction(1, 2)) * pair_ticks
                offset += tempo.seconds(note.start + swing_ticks) - start_seconds
        if depth:
            offset += rational(_microtiming_ms_for_role(groove_role, beat, 1.0, spec.microtiming_ms, counter)) / 1000
        if depth and spec.anticipation is not None:
            ant = getattr(spec.anticipation, groove_role, None)
            if ant is not None and ant.push_ms and ant.prob >= 1.0:
                meter_tick, numerator, denominator = meters[bisect_right(meter_ticks, note.start) - 1]
                bar_ticks = Fraction(midi.ticks_per_beat * numerator * 4, denominator)
                bar_start = note.start - (note.start - meter_tick) % bar_ticks
                if start_seconds - tempo.seconds(bar_start) < Fraction(3, 100):
                    offset -= rational(ant.push_ms[0]) / 1000
        new_start = tempo.shift(note.start, offset)
        duration = tempo.seconds(note.end) - start_seconds
        if depth and spec.articulation is not None:
            art = getattr(spec.articulation, groove_role, None)
            if art is not None and art.duration_scale is not None:
                duration *= rational(art.duration_scale)
        new_end = note.end + round(tempo.tick(tempo.seconds(new_start) + duration) - note.end)
        velocity = max(1, min(127, round(note.velocity * _velocity_scale(groove_role, beat, 1.0, spec.velocity, depth))))
        targets[note.id] = NoteTiming(new_start, new_end, velocity)
        actual_offset = tempo.seconds(new_start) - start_seconds
        note_diffs.append({"role": role, "pitch": note.pitch, "offset_sec": float(actual_offset),
                           "offset_ticks": new_start - note.start, "velocity_in": note.velocity,
                           "velocity_out": velocity})
        if depth and spec.velocity and spec.velocity.ghost_notes and groove_role in spec.velocity.ghost_notes:
            ghost = spec.velocity.ghost_notes[groove_role]
            if ghost.prob >= 1.0:
                ghost_start = new_start + round(Fraction(midi.ticks_per_beat, 8))
                ghost_end = ghost_start + max(1, round(tempo.tick(tempo.seconds(ghost_start) + duration * Fraction(3, 10)) - ghost_start))
                ghosts.append((note.id, NoteTiming(ghost_start, ghost_end,
                                                   max(1, min(127, round(velocity * ghost.vel_scale))))))
    out = retime(midi, targets)
    if ghosts:
        out = add_ghosts(out, ghosts)
    return out, {
        "notes_in": len(midi.notes), "notes_out": len(out.notes), "ghost_inserted": len(ghosts),
        "light_cleanup": light_cleanup,
        "mean_abs_offset_ms": sum(abs(d["offset_sec"]) for d in note_diffs) / max(1, len(note_diffs)) * 1000,
        "note_diffs": note_diffs,
    }


def _apply_to_midi(
    midi: pm.PrettyMIDI,
    spec: GrooveSpec,
    *,
    light_cleanup: bool = False,
) -> tuple[pm.PrettyMIDI, dict[str, Any]]:
    """Legacy in-memory adapter; information already lost by PrettyMIDI is unrecoverable.

    The public file API never takes this path. Existing analysis callers keep
    their object/return type while using the same event transform underneath.
    """
    buffer = io.BytesIO()
    midi.write(buffer)
    buffer.seek(0)
    out, report = _apply_to_events(MidiEvents.from_mido(mido.MidiFile(file=buffer)), spec,
                                   light_cleanup=light_cleanup)
    rendered = io.BytesIO()
    out.to_mido().save(file=rendered)
    rendered.seek(0)
    return pm.PrettyMIDI(rendered), report


def apply_groove(
    midi_path: Path | str,
    spec: GrooveSpec | dict[str, Any],
    *,
    analysis: dict[str, Any] | None = None,
    output_path: Path | str | None = None,
    config: dict[str, Any] | None = None,
    light_cleanup: bool | None = None,
) -> Path:
    """Apply groove using complete analysis roles keyed by ``source_track_index``.

    Without analysis, retain legacy name/channel heuristics. Supplied analysis
    must identify every note-bearing source track uniquely; reconcile planner
    documents or regenerate legacy instrument-index analysis before calling.
    Reports retain canonical roles (``comping`` uses GrooveSpec's ``comp`` fields).
    """
    path = Path(midi_path)
    if not path.is_file():
        raise FileNotFoundError(f"MIDI file not found: {path}")

    validated = parse_groove_spec(spec)

    cfg = config or {}
    if light_cleanup is None:
        light_cleanup = bool(cfg.get("groove", {}).get("light_cleanup", False))

    midi = MidiEvents.load(path)
    roles = _analysis_roles(midi, analysis)
    out_midi, report = _apply_to_events(midi, validated, light_cleanup=light_cleanup, roles=roles)

    if output_path is None:
        out_dir = Path(cfg.get("paths", {}).get("output_dir", "outputs"))
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / f"{path.stem}.expressive.mid"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    report_path = Path(str(output_path) + ".diff.json")
    code_version = str(cfg.get("code_version", "0.1.0"))
    subset: dict[str, Any] = {"groove": validated.model_dump(mode="json"), "light_cleanup": light_cleanup}
    if roles is not None:
        subset["source_track_roles"] = roles
    manifest = StageManifest(
        stage="expressivization",
        inputs_hash=content_hash_files([path], json.dumps(subset, sort_keys=True), code_version),
        config_subset=subset,
        metadata={
            "notes_in": report["notes_in"],
            "notes_out": report["notes_out"],
            "ghost_inserted": report["ghost_inserted"],
            "analysis_hash": (analysis or {}).get("inputs_hash"),
            "diff_report": report_path.name,
        },
        code_version=code_version,
        output_path=output_path.name,
    )
    buffer = io.BytesIO()
    out_midi.to_mido().save(file=buffer)
    publish_bundle(path, output_path, {
        "": buffer.getvalue(),
        ".diff.json": json.dumps(report, indent=2, sort_keys=True).encode("utf-8"),
        ".json": json.dumps(manifest.to_dict(), indent=2, sort_keys=True).encode("utf-8"),
    })
    return output_path
