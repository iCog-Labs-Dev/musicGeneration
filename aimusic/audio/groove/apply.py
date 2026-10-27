"""Deterministic groove-spec applicator (PDF §3.1 / M1)."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pretty_midi as pm

from aimusic.audio.groove.spec import GrooveSpec, parse_groove_spec
from aimusic.audio.manifest import write_manifest

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


def _apply_to_midi(
    midi: pm.PrettyMIDI,
    spec: GrooveSpec,
    *,
    light_cleanup: bool = False,
) -> tuple[pm.PrettyMIDI, dict[str, Any]]:
    """Return new PrettyMIDI + diff report. light_cleanup skips expressive transforms."""
    spb = _seconds_per_beat(midi)
    ticks_per_beat = float(midi.resolution)
    sec_per_tick = spb / ticks_per_beat

    out = pm.PrettyMIDI(initial_tempo=60.0 / spb)
    out.time_signature_changes = list(midi.time_signature_changes)

    note_diffs: list[dict[str, Any]] = []
    ghost_inserted = 0
    total_in = 0
    total_out = 0
    note_counter = 0

    depth = 0.0 if light_cleanup else 1.0

    for inst in midi.instruments:
        role = _role_for_instrument(inst)
        new_inst = pm.Instrument(program=inst.program, is_drum=inst.is_drum, name=inst.name)
        for n in inst.notes:
            total_in += 1
            note_counter += 1
            start = float(n.start)
            end = float(n.end)
            vel = int(n.velocity)
            offset_sec = 0.0

            if depth > 0 and spec.swing is not None and role in (spec.swing.applies_to or []):
                offset_sec += _swing_offset_sec(
                    start, spb, float(spec.swing.ratio), int(spec.swing.subdivision)
                )

            if depth > 0:
                offset_sec += (
                    _microtiming_ms_for_role(
                        role, start, spb, spec.microtiming_ms, note_counter
                    )
                    / 1000.0
                )

            if depth > 0 and spec.anticipation is not None:
                ant = getattr(spec.anticipation, role, None)
                if ant is not None and ant.push_ms:
                    bar_sec = spb * 4.0
                    pos_in_bar = start % bar_sec
                    if pos_in_bar < 0.03 and ant.prob >= 1.0:
                        push = float(ant.push_ms[0]) / 1000.0
                        offset_sec -= push

            new_start = max(0.0, start + offset_sec)
            dur = max(1e-4, end - start)

            if depth > 0 and spec.articulation is not None:
                art = getattr(spec.articulation, role, None)
                if art is not None and art.duration_scale is not None:
                    dur *= float(art.duration_scale)

            new_end = new_start + dur

            vscale = _velocity_scale(role, start, spb, spec.velocity, depth=depth)
            new_vel = max(1, min(127, round(vel * vscale)))

            offset_ticks = offset_sec / sec_per_tick if sec_per_tick else 0.0
            note_diffs.append(
                {
                    "role": role,
                    "pitch": n.pitch,
                    "offset_sec": offset_sec,
                    "offset_ticks": offset_ticks,
                    "velocity_in": vel,
                    "velocity_out": new_vel,
                }
            )
            new_inst.notes.append(
                pm.Note(velocity=new_vel, pitch=n.pitch, start=new_start, end=new_end)
            )
            total_out += 1

            if (
                depth > 0
                and spec.velocity
                and spec.velocity.ghost_notes
                and role in spec.velocity.ghost_notes
            ):
                gn = spec.velocity.ghost_notes[role]
                if gn.prob >= 1.0:
                    g_start = new_start + (spb / 8.0)
                    g_vel = max(1, min(127, round(new_vel * float(gn.vel_scale))))
                    new_inst.notes.append(
                        pm.Note(
                            velocity=g_vel,
                            pitch=n.pitch,
                            start=g_start,
                            end=g_start + dur * 0.3,
                        )
                    )
                    ghost_inserted += 1
                    total_out += 1

        new_inst.notes.sort(key=lambda x: (x.start, x.pitch))
        out.instruments.append(new_inst)

    report = {
        "notes_in": total_in,
        "notes_out": total_out,
        "ghost_inserted": ghost_inserted,
        "light_cleanup": light_cleanup,
        "mean_abs_offset_ms": (
            sum(abs(d["offset_sec"]) for d in note_diffs) / max(1, len(note_diffs)) * 1000.0
        ),
        "note_diffs": note_diffs,
    }
    return out, report


def apply_groove(
    midi_path: Path | str,
    spec: GrooveSpec | dict[str, Any],
    *,
    analysis: dict[str, Any] | None = None,
    output_path: Path | str | None = None,
    config: dict[str, Any] | None = None,
    light_cleanup: bool | None = None,
) -> Path:
    """Apply a validated groove spec to MIDI; emit expressive MIDI + diff report."""
    path = Path(midi_path)
    if not path.is_file():
        raise FileNotFoundError(f"MIDI file not found: {path}")

    validated = parse_groove_spec(spec)

    cfg = config or {}
    if light_cleanup is None:
        light_cleanup = bool(cfg.get("groove", {}).get("light_cleanup", False))

    midi = pm.PrettyMIDI(str(path))
    out_midi, report = _apply_to_midi(midi, validated, light_cleanup=light_cleanup)

    if output_path is None:
        out_dir = Path(cfg.get("paths", {}).get("output_dir", "outputs"))
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / f"{path.stem}.expressive.mid"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    out_midi.write(str(output_path))
    report_path = Path(str(output_path) + ".diff.json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    code_version = str(cfg.get("code_version", "0.1.0"))
    write_manifest(
        "expressivization",
        output_path,
        [path],
        config_subset={"groove": validated.model_dump(mode="json"), "light_cleanup": light_cleanup},
        metadata={
            "notes_in": report["notes_in"],
            "notes_out": report["notes_out"],
            "ghost_inserted": report["ghost_inserted"],
            "analysis_hash": (analysis or {}).get("inputs_hash"),
            "diff_report": str(report_path),
        },
        code_version=code_version,
    )
    return output_path
