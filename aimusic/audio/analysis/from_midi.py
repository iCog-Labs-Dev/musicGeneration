"""STAGE 0: MIDI analysis and structured description."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pretty_midi as pm

from aimusic.audio.manifest import content_hash_files, write_manifest
from aimusic.audio.midi_events import MidiEvents

ANALYSIS_VERSION = "1.0.0"

_NAME_ROLES = ("drums", "bass", "lead", "comp", "pads", "ornament", "piano")


def _infer_role(instrument: pm.Instrument) -> str:
    return infer_role(instrument.name, instrument.program, instrument.is_drum)


def infer_role(name: str, program: int, is_drum: bool) -> str:
    """Pure name/program heuristic; imported evidence is never planner authority."""
    name = (name or "").strip().lower()
    for role in _NAME_ROLES:
        if role in name:
            return "comping" if role == "comp" else role
    if is_drum:
        return "drums"
    prog = int(program)
    if prog <= 7:
        return "piano"
    if 32 <= prog <= 39:
        return "bass"
    if 40 <= prog <= 51:
        return "comping"
    if 88 <= prog <= 95:
        return "pads"
    if 80 <= prog <= 103:
        return "lead"
    return "comping"


def infer_source_tracks(source: MidiEvents) -> list[dict[str, Any]]:
    """Infer roles once per SMF source track, not per PrettyMIDI instrument."""
    rows = []
    for index, isolated in source.isolate_tracks().items():
        name = next((e.message.name for e in source.events
                     if e.track == index and e.message.type == 'track_name'), '')
        channels = sorted({n.channel for n in isolated.notes})
        current = {ch: 0 for ch in channels}
        used: set[int] = set()
        changes = []
        for event in isolated.events:
            msg = event.message
            if event.channel not in current:
                continue
            if msg.type == 'program_change':
                current[msg.channel] = msg.program
                changes.append({'tick': event.tick, 'channel': msg.channel, 'program': msg.program})
            if msg.type == 'note_on' and msg.velocity:
                used.add(current[msg.channel])
        programs = sorted(used)
        percussion = channels == [9]
        roles = {infer_role(name, p, percussion) for p in programs}
        role = next(iter(roles)) if len(roles) == 1 and (9 not in channels or percussion) else 'mixed'
        rows.append({'source_track_index': index, 'name': name, 'channels': channels,
                     'role': role, 'role_source': 'inferred', 'count_source': 'MIDI events',
                     'program': programs[0] if len(programs) == 1 else None,
                     'programs': programs, 'program_changes': changes, 'is_drum': percussion,
                     'note_count': len(isolated.notes)})
    return rows


def _tempo_bpm(midi: pm.PrettyMIDI) -> float:
    if hasattr(midi, "get_tempo_changes"):
        _times, tempos = midi.get_tempo_changes()
        if len(tempos):
            return float(tempos[0])
    return 120.0


def _time_signature(midi: pm.PrettyMIDI) -> dict[str, int]:
    if midi.time_signature_changes:
        ts = midi.time_signature_changes[0]
        return {"numerator": int(ts.numerator), "denominator": int(ts.denominator)}
    return {"numerator": 4, "denominator": 4}


def _bar_boundaries_sec(midi: pm.PrettyMIDI, *, n_bars_hint: int | None = None) -> list[float]:
    """Return start times (seconds) of each bar plus the end time."""
    bpm = _tempo_bpm(midi)
    ts = _time_signature(midi)
    beat_sec = 60.0 / bpm
    bar_sec = beat_sec * ts["numerator"] * (4.0 / ts["denominator"])
    end = float(midi.get_end_time())
    if end <= 0:
        return [0.0]
    n_bars = n_bars_hint if n_bars_hint is not None else max(1, round(end / bar_sec))
    bounds = [i * bar_sec for i in range(n_bars + 1)]
    if bounds[-1] < end:
        bounds.append(end)
    else:
        bounds[-1] = end
    return bounds


def _track_entry(idx: int, track: dict[str, Any], pitches: list[int]) -> dict[str, Any]:
    return {
        **track,
        "index": idx,  # Legacy display order only; source_track_index is the identity.
        "pitch_min": min(pitches) if pitches else None,
        "pitch_max": max(pitches) if pitches else None,
    }


def build_analysis_document(
    midi_path: Path, midi: pm.PrettyMIDI, *, code_version: str
) -> dict[str, Any]:
    inputs_hash = content_hash_files([midi_path])
    # PrettyMIDI may split/reorder source tracks by program/channel. Use the same
    # event-based role authority as imported contexts, including explicit mixed roles.
    source = MidiEvents.load(midi_path)
    pitches: dict[int, list[int]] = {}
    for note in source.notes:
        pitches.setdefault(note.track, []).append(note.pitch)
    tracks = [_track_entry(i, track, pitches[track["source_track_index"]])
              for i, track in enumerate(infer_source_tracks(source))]
    return {
        "schema_version": ANALYSIS_VERSION,
        "code_version": code_version,
        "source_midi": midi_path.name,
        "inputs_hash": inputs_hash,
        "resolution": int(midi.resolution),
        "tempo_bpm": _tempo_bpm(midi),
        "time_signature": _time_signature(midi),
        "duration_sec": float(midi.get_end_time()),
        "bar_boundaries_sec": _bar_boundaries_sec(midi),
        "tracks": tracks,
        "total_notes": sum(t["note_count"] for t in tracks),
    }


def analyze_midi(
    midi_path: Path | str,
    *,
    config: dict[str, Any] | None = None,
    output_path: Path | str | None = None,
) -> dict[str, Any]:
    """Produce a versioned, content-hashed analysis JSON document (+ sidecar)."""
    import pretty_midi as pm

    path = Path(midi_path)
    if not path.is_file():
        raise FileNotFoundError(f"MIDI file not found: {path}")

    cfg = config or {}
    code_version = str(cfg.get("code_version", "0.1.0"))

    try:
        midi = pm.PrettyMIDI(str(path))
    except Exception as exc:
        raise ValueError(f"invalid MIDI file: {path}") from exc

    doc = build_analysis_document(path, midi, code_version=code_version)

    if output_path is None:
        out_dir = Path(cfg.get("paths", {}).get("output_dir", "outputs"))
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / f"{path.stem}.analysis.json"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(json.dumps(doc, indent=2, sort_keys=True), encoding="utf-8")
    write_manifest(
        "analysis",
        output_path,
        [path],
        config_subset={"code_version": code_version, "schema_version": ANALYSIS_VERSION},
        metadata={"total_notes": doc["total_notes"], "n_tracks": len(doc["tracks"])},
        code_version=code_version,
    )
    return doc
