"""RenderPackage contract producer (AUDIO_PIPELINE_ARCHITECTURE.pdf §3).

Emits a content-hashed ``run_<hash>/`` directory that is the sole artifact
boundary between the symbolic composer and the audio pipeline.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

from aimusic.core.core_types import BeatState, NoteEvent, Score
from aimusic.core.diagnostics import RunManifest
from aimusic.core.vocab import Vocabularies

STRUCTURE_SCHEMA = "aimusic.structure/1"
DEFAULT_PROGRAMS: dict[str, int | None] = {
    "bass": 33,
    "comping": 4,
    "lead": 81,
    "drums": None,
}
MICROTONAL_TOLERANCE_CENTS = 8.0


class ContractViolation(ValueError):
    """Raised when a RenderPackage fails a boundary invariant."""


@dataclass(frozen=True)
class RenderPackage:
    """Versioned, content-hashed directory handed to the audio pipeline."""

    root: Path
    midi_path: Path
    score_path: Path
    structure_path: Path
    tuning_path: Path
    manifest_path: Path
    content_hash: str
    beatstates_path: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "midi_path": str(self.midi_path),
            "score_path": str(self.score_path),
            "structure_path": str(self.structure_path),
            "tuning_path": str(self.tuning_path),
            "manifest_path": str(self.manifest_path),
            "content_hash": self.content_hash,
            "beatstates_path": str(self.beatstates_path) if self.beatstates_path else None,
        }


@dataclass
class StructureDoc:
    """STAGE 0 analysis artifact (schema v1)."""

    schema: str = STRUCTURE_SCHEMA
    provenance: str = "planner"
    source_hash: str = ""
    edo: int = 12
    base_tuning: float = 0.0
    tempo_map: list[list[float]] = field(default_factory=list)
    meter: list[list[Any]] = field(default_factory=list)
    key: list[dict[str, Any]] = field(default_factory=list)
    chords: list[dict[str, Any]] = field(default_factory=list)
    sections: list[dict[str, Any]] = field(default_factory=list)
    bar_table: list[list[float]] = field(default_factory=list)
    tracks: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _label(vocabularies: Vocabularies | None, attr: str, token_id: int) -> str:
    if vocabularies is None:
        return str(token_id)
    vocabulary = getattr(vocabularies, attr)
    if vocabulary.has_id(token_id):
        return vocabulary.token_for_id(token_id).label
    return str(token_id)


def _beats_per_bar(vocabularies: Vocabularies | None, meter_id: int) -> int:
    if vocabularies is None:
        return 4
    if not vocabularies.meters.has_id(meter_id):
        return 4
    return int(vocabularies.meters.token_for_id(meter_id).beats_per_bar)


def _parse_meter_beats(label: str) -> int:
    try:
        num, _den = label.split("/", maxsplit=1)
        return max(1, int(num))
    except (TypeError, ValueError):
        return 4


def _onset_profile_16(events: Sequence[NoteEvent], ticks_per_beat: int) -> list[float]:
    bins = [0.0] * 16
    if ticks_per_beat <= 0:
        return bins
    for event in events:
        sixteenth = int((event.ton % (ticks_per_beat * 4)) / max(1, ticks_per_beat / 4))
        sixteenth = max(0, min(15, sixteenth))
        bins[sixteenth] += 1.0
    total = sum(bins) or 1.0
    return [round(v / total, 4) for v in bins]


def _track_is_microtonal(events: Sequence[NoteEvent], edo: int, *, tolerance_cents: float) -> bool:
    if edo == 12 or not events:
        return False
    cents_per_step = 1200.0 / edo
    for event in events:
        pc = event.h % edo
        # Distance from nearest 12-TET pitch class in cents within one octave.
        absolute_cents = (pc * cents_per_step) % 1200.0
        nearest_12 = round(absolute_cents / 100.0) * 100.0
        if abs(absolute_cents - nearest_12) > tolerance_cents:
            return True
    return False


def build_structure(
    score: Score,
    path: Sequence[BeatState],
    *,
    edo: int = 12,
    base_tuning: float = 0.0,
    vocabularies: Vocabularies | None = None,
    provenance: str = "planner",
    source_hash: str = "",
    track_programs: dict[str, int | None] | None = None,
    tolerance_cents: float = MICROTONAL_TOLERANCE_CENTS,
) -> StructureDoc:
    """Build structure.json from planner BeatState path + decoded Score (B1)."""
    programs = {**DEFAULT_PROGRAMS, **(track_programs or {})}
    tempo = float(score.tempo_bpm)
    decoded = list(path[:-1] if len(path) > 1 else path)

    meter_label = "4/4"
    if decoded:
        meter_label = _label(vocabularies, "meters", decoded[0].meter_id)
    beats_per_bar = _parse_meter_beats(meter_label)
    if decoded:
        beats_per_bar = _beats_per_bar(vocabularies, decoded[0].meter_id)

    n_beats = max(len(decoded), 1)
    n_bars = max(1, math.ceil(n_beats / beats_per_bar))
    sec_per_beat = 60.0 / tempo

    tempo_map = [[0.0, tempo]]
    meter = [[0, meter_label]]
    bar_table = [[float(i), float(i * beats_per_bar * sec_per_beat)] for i in range(n_bars + 1)]

    # Key / chord timelines compressed to change points.
    key: list[dict[str, Any]] = []
    chords: list[dict[str, Any]] = []
    last_key: int | None = None
    last_chord: int | None = None
    for beat_idx, state in enumerate(decoded):
        bar = beat_idx // beats_per_bar
        beat_in_bar = beat_idx % beats_per_bar
        if state.key_id != last_key:
            key.append(
                {
                    "start_bar": bar,
                    "key_id": _label(vocabularies, "keys", state.key_id),
                }
            )
            last_key = state.key_id
        if state.chord_id != last_chord:
            chords.append(
                {
                    "bar": bar,
                    "beat": beat_in_bar,
                    "chord_id": _label(vocabularies, "chords", state.chord_id),
                }
            )
            last_chord = state.chord_id

    # Sections from boundary_lvl: split when boundary_lvl >= 2 (phrase-ish).
    sections: list[dict[str, Any]] = []
    if decoded:
        sec_start = 0
        sec_idx = 0
        for beat_idx, state in enumerate(decoded):
            is_boundary = state.boundary_lvl >= 2 and beat_idx > sec_start
            is_last = beat_idx == len(decoded) - 1
            if is_boundary or is_last:
                end_beat = beat_idx + (1 if is_last else 0)
                start_bar = sec_start // beats_per_bar
                end_bar = max(start_bar + 1, math.ceil(end_beat / beats_per_bar))
                window = decoded[sec_start:end_beat] or [state]
                max_boundary = max(s.boundary_lvl for s in window)
                sections.append(
                    {
                        "label": chr(ord("A") + (sec_idx % 26)),
                        "bars": [start_bar, end_bar],
                        "energy": round(min(1.0, 0.3 + 0.1 * max_boundary), 3),
                        "tension": round(min(1.0, 0.2 + 0.15 * max_boundary), 3),
                        "boundary_lvl": int(max_boundary),
                    }
                )
                sec_idx += 1
                sec_start = beat_idx
    else:
        sections.append(
            {
                "label": "A",
                "bars": [0, n_bars],
                "energy": 0.4,
                "tension": 0.3,
                "boundary_lvl": 0,
            }
        )

    by_track: dict[str, list[NoteEvent]] = {}
    for event in score.note_events:
        by_track.setdefault(event.track, []).append(event)

    tracks: list[dict[str, Any]] = []
    for name in sorted(by_track.keys()):
        events = by_track[name]
        pitches = [e.h for e in events]
        role = name if name in DEFAULT_PROGRAMS else "lead"
        percussion = role == "drums" or programs.get(name) is None and name == "drums"
        # GM drum keys are sample indices, not EDO pitch heights.
        microtonal = (
            False
            if percussion
            else _track_is_microtonal(events, edo, tolerance_cents=tolerance_cents)
        )
        tracks.append(
            {
                "name": name,
                "role": role,
                "program": programs.get(name, programs.get(role)),
                "percussion": bool(percussion),
                "quantized": True,
                "flat_velocity": True,
                "onset_profile_16": _onset_profile_16(events, score.ticks_per_beat),
                "pitch_range": [min(pitches), max(pitches)] if pitches else [0, 0],
                "microtonal": microtonal,
            }
        )

    return StructureDoc(
        schema=STRUCTURE_SCHEMA,
        provenance=provenance,
        source_hash=source_hash,
        edo=int(edo),
        base_tuning=float(base_tuning),
        tempo_map=tempo_map,
        meter=meter,
        key=key,
        chords=chords,
        sections=sections,
        bar_table=bar_table,
        tracks=tracks,
    )


def build_tuning(
    *,
    edo: int,
    base_tuning: float,
    structure: StructureDoc,
    pitch_bend_range: int = 2,
    rendering_method: str = "MPE",
) -> dict[str, Any]:
    any_micro = any(t.get("microtonal") for t in structure.tracks)
    if edo == 12 and not any_micro:
        method = "direct_12"
    else:
        method = rendering_method.lower() if rendering_method else "mpe"
    return {
        "edo": int(edo),
        "base_tuning": float(base_tuning),
        "pitch_bend_range": int(pitch_bend_range),
        "method": method,
        "per_track": {
            t["name"]: ("direct_12" if not t.get("microtonal") else method) for t in structure.tracks
        },
    }


def assert_contract_invariants(
    package: RenderPackage,
    *,
    score: Score | None = None,
) -> None:
    """Assert PDF §3.2 contract invariants 1–5; raise ContractViolation on failure."""
    structure = json.loads(package.structure_path.read_text(encoding="utf-8"))
    tuning = json.loads(package.tuning_path.read_text(encoding="utf-8"))

    if score is None:
        score = Score.from_dict(json.loads(package.score_path.read_text(encoding="utf-8")))

    # 1. Every NoteEvent.track appears exactly once in structure.tracks
    track_names = [t["name"] for t in structure.get("tracks", [])]
    if len(track_names) != len(set(track_names)):
        raise ContractViolation("structure.tracks contains duplicate track names")
    score_tracks = sorted(score.track_event_counts().keys())
    if sorted(track_names) != score_tracks:
        raise ContractViolation(
            f"structure.tracks {sorted(track_names)!r} != score tracks {score_tracks!r}"
        )

    # 2. bar_table monotonic and consistent with tempo_map
    bar_table = structure.get("bar_table") or []
    times = [row[1] for row in bar_table]
    if any(times[i] > times[i + 1] for i in range(len(times) - 1)):
        raise ContractViolation("bar_table times are not monotonic")
    if not structure.get("tempo_map"):
        raise ContractViolation("tempo_map is empty")
    if not structure.get("meter"):
        raise ContractViolation("meter is empty")

    # 3. tuning.edo == structure.edo
    if int(tuning.get("edo", -1)) != int(structure.get("edo", -2)):
        raise ContractViolation("tuning.json.edo != structure.edo")

    # 4. score.mid exists and is non-empty
    if not package.midi_path.is_file() or package.midi_path.stat().st_size == 0:
        raise ContractViolation("score.mid missing or empty")

    # 5. microtonal tracks never use direct_12 method
    for track in structure.get("tracks", []):
        if track.get("microtonal"):
            method = tuning.get("per_track", {}).get(track["name"], tuning.get("method"))
            if method == "direct_12":
                raise ContractViolation(
                    f"microtonal track {track['name']!r} has method direct_12"
                )


def _content_hash_bytes(*blobs: bytes) -> str:
    h = hashlib.sha256()
    for blob in blobs:
        h.update(blob)
    return h.hexdigest()


def _canonical_json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")


def write_render_package(
    out_dir: Path | str,
    *,
    score: Score,
    midi_path: Path | str,
    manifest: RunManifest,
    path: Sequence[BeatState] = (),
    vocabularies: Vocabularies | None = None,
    edo: int = 12,
    base_tuning: float = 0.0,
    pitch_bend_range: int = 2,
    rendering_method: str = "MPE",
    track_programs: dict[str, int | None] | None = None,
    run_id: str | None = None,
) -> RenderPackage:
    """Write ``run_<runId8>_<packageHash12>/`` containing the inter-system contract artifacts."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    midi_src = Path(midi_path)
    score_bytes = json.dumps(score.to_dict(), sort_keys=True).encode("utf-8")
    midi_bytes = midi_src.read_bytes()
    source_hash = _content_hash_bytes(score_bytes, midi_bytes, str(edo).encode())

    structure = build_structure(
        score,
        path,
        edo=edo,
        base_tuning=base_tuning,
        vocabularies=vocabularies,
        provenance="planner",
        source_hash=f"sha256:{source_hash}",
        track_programs=track_programs,
    )
    structure_bytes = _canonical_json_bytes(structure.to_dict())
    tuning = build_tuning(
        edo=edo,
        base_tuning=base_tuning,
        structure=structure,
        pitch_bend_range=pitch_bend_range,
        rendering_method=rendering_method,
    )
    tuning_bytes = _canonical_json_bytes(tuning)
    beatstates_bytes = "".join(
        json.dumps(state.to_dict(vocabularies)) + "\n" for state in path
    ).encode("utf-8")
    package_hash = _content_hash_bytes(
        source_hash.encode("utf-8"),
        structure_bytes,
        tuning_bytes,
        beatstates_bytes,
    )

    package_id = run_id or manifest.run_id
    root = out_dir / f"run_{package_id[:8]}_{package_hash[:12]}"
    if root.exists():
        existing = load_render_package(root)
        if existing.content_hash == package_hash:
            return existing
        raise ContractViolation(
            f"RenderPackage path {root} already exists with different package identity "
            f"(existing={existing.content_hash[:12]}, computed={package_hash[:12]})"
        )
    root.mkdir(parents=True)

    score_path = root / "score.json"
    midi_dst = root / "score.mid"
    structure_path = root / "structure.json"
    tuning_path = root / "tuning.json"
    manifest_path = root / "manifest.json"
    beatstates_path = root / "beatstates.jsonl"

    score_path.write_bytes(score_bytes)
    shutil.copy2(midi_src, midi_dst)
    structure_path.write_bytes(structure_bytes)
    tuning_path.write_bytes(tuning_bytes)
    beatstates_path.write_bytes(beatstates_bytes)

    manifest_data = manifest.to_dict()
    manifest_data["render_package"] = {
        "source_hash": source_hash,
        "package_hash": package_hash,
        "content_hash": package_hash,
        "structure_schema": STRUCTURE_SCHEMA,
    }
    manifest_path.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")

    package = RenderPackage(
        root=root,
        midi_path=midi_dst,
        score_path=score_path,
        structure_path=structure_path,
        tuning_path=tuning_path,
        manifest_path=manifest_path,
        content_hash=package_hash,
        beatstates_path=beatstates_path,
    )
    assert_contract_invariants(package, score=score)
    return package


def load_render_package(root: Path | str) -> RenderPackage:
    """Load and validate an existing RenderPackage directory."""
    root = Path(root)
    required = ("score.mid", "score.json", "structure.json", "tuning.json", "manifest.json")
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"RenderPackage incomplete at {root}: missing {missing}")
    beatstates = root / "beatstates.jsonl"
    manifest_data = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    render_meta = manifest_data.get("render_package") or {}
    package_hash = render_meta.get("package_hash")
    if isinstance(package_hash, str) and package_hash:
        content_hash = package_hash
    else:
        structure = json.loads((root / "structure.json").read_text(encoding="utf-8"))
        content_hash = str(structure.get("source_hash", "")).removeprefix("sha256:") or "unknown"
    package = RenderPackage(
        root=root,
        midi_path=root / "score.mid",
        score_path=root / "score.json",
        structure_path=root / "structure.json",
        tuning_path=root / "tuning.json",
        manifest_path=root / "manifest.json",
        content_hash=content_hash,
        beatstates_path=beatstates if beatstates.is_file() else None,
    )
    assert_contract_invariants(package)
    return package
