"""Tests for RenderPackage contract (AUDIO_PIPELINE_ARCHITECTURE.pdf §3)."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from aimusic.core.core_types import BeatState, NoteEvent, Score
from aimusic.core.diagnostics import RunManifest
from aimusic.render.package import (
    STRUCTURE_SCHEMA,
    ContractViolation,
    assert_contract_invariants,
    build_structure,
    load_render_package,
    write_render_package,
)
from aimusic.theory.edo import EDO
from aimusic.core.config import EDOConfig, MicrotonalRendering
from aimusic.render import render_midi


def _tiny_score(*, edo: int = 12) -> tuple[Score, tuple[BeatState, ...]]:
    # For 19-EDO, use step 1 on lead (~63¢) so microtonal detection fires (>8¢).
    events = (
        NoteEvent(ton=0, toff=240, h=12 if edo == 12 else 19, v=0.8, track="bass"),
        NoteEvent(ton=0, toff=120, h=36 if edo == 12 else 1, v=0.7, track="lead"),
        NoteEvent(ton=0, toff=60, h=40, v=0.9, track="drums"),
        NoteEvent(ton=240, toff=480, h=16 if edo == 12 else 21, v=0.75, track="comping"),
    )
    score = Score(note_events=events, ticks_per_beat=480, tempo_bpm=120.0)
    path = tuple(
        BeatState(
            meter_id=0,
            beat_in_bar=i % 4,
            boundary_lvl=2 if i % 4 == 0 else 0,
            key_id=0,
            chord_id=0,
            role_id=0,
            head_id=0,
            groove_id=0,
        )
        for i in range(8)
    )
    return score, path


class TestRenderPackage(unittest.TestCase):
    def test_write_and_load_package_passes_invariants(self) -> None:
        score, path = _tiny_score()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            midi_path = tmp_path / "tmp.mid"
            edo = EDO(EDOConfig(n=12, base_tuning=0, pitch_bend_range=2,
                                microtonal_rendering_method=MicrotonalRendering.MPE))
            render_midi(score, edo, str(midi_path))
            manifest = RunManifest(seed=1, config_dump={"edo": 12})
            package = write_render_package(
                tmp_path,
                score=score,
                midi_path=midi_path,
                manifest=manifest,
                path=path,
                edo=12,
                run_id="fixture01",
            )
            self.assertTrue(package.root.is_dir())
            for name in ("score.mid", "score.json", "structure.json", "tuning.json", "manifest.json"):
                self.assertTrue((package.root / name).is_file(), msg=name)
            structure = json.loads(package.structure_path.read_text(encoding="utf-8"))
            self.assertEqual(structure["schema"], STRUCTURE_SCHEMA)
            self.assertEqual(structure["provenance"], "planner")
            manifest = json.loads(package.manifest_path.read_text(encoding="utf-8"))
            rp = manifest["render_package"]
            self.assertIn("source_hash", rp)
            self.assertIn("package_hash", rp)
            self.assertEqual(rp["package_hash"], package.content_hash)
            self.assertEqual(structure["source_hash"], f"sha256:{rp['source_hash']}")
            loaded = load_render_package(package.root)
            assert_contract_invariants(loaded, score=score)
            self.assertEqual(loaded.content_hash, package.content_hash)

    def test_package_hash_diverges_on_tuning_params(self) -> None:
        score, path = _tiny_score()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            midi_path = tmp_path / "tmp.mid"
            edo = EDO(EDOConfig(n=12, base_tuning=0, pitch_bend_range=2,
                                microtonal_rendering_method=MicrotonalRendering.MPE))
            render_midi(score, edo, str(midi_path))
            manifest = RunManifest(seed=1, config_dump={"edo": 12})
            pkg_a = write_render_package(
                tmp_path,
                score=score,
                midi_path=midi_path,
                manifest=manifest,
                path=path,
                edo=12,
                run_id="hashdiv01",
                base_tuning=0.0,
                pitch_bend_range=2,
            )
            pkg_b = write_render_package(
                tmp_path,
                score=score,
                midi_path=midi_path,
                manifest=manifest,
                path=path,
                edo=12,
                run_id="hashdiv01",
                base_tuning=1.0,
                pitch_bend_range=2,
            )
            pkg_c = write_render_package(
                tmp_path,
                score=score,
                midi_path=midi_path,
                manifest=manifest,
                path=path,
                edo=12,
                run_id="hashdiv01",
                base_tuning=0.0,
                pitch_bend_range=48,
            )
            self.assertNotEqual(pkg_a.content_hash, pkg_b.content_hash)
            self.assertNotEqual(pkg_a.content_hash, pkg_c.content_hash)
            self.assertNotEqual(pkg_a.root, pkg_b.root)
            # source_hash (score+midi+edo) stays stable across tuning-only changes
            src_a = json.loads(pkg_a.structure_path.read_text(encoding="utf-8"))["source_hash"]
            src_b = json.loads(pkg_b.structure_path.read_text(encoding="utf-8"))["source_hash"]
            self.assertEqual(src_a, src_b)

    def test_write_render_package_idempotent_second_write(self) -> None:
        score, path = _tiny_score()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            midi_path = tmp_path / "tmp.mid"
            edo = EDO(EDOConfig(n=12, base_tuning=0, pitch_bend_range=2,
                                microtonal_rendering_method=MicrotonalRendering.MPE))
            render_midi(score, edo, str(midi_path))
            manifest = RunManifest(seed=1, config_dump={"edo": 12})
            first = write_render_package(
                tmp_path,
                score=score,
                midi_path=midi_path,
                manifest=manifest,
                path=path,
                edo=12,
                run_id="idempo001",
            )
            mtime_before = first.structure_path.stat().st_mtime_ns
            second = write_render_package(
                tmp_path,
                score=score,
                midi_path=midi_path,
                manifest=manifest,
                path=path,
                edo=12,
                run_id="idempo001",
            )
            self.assertEqual(first.root, second.root)
            self.assertEqual(first.content_hash, second.content_hash)
            self.assertEqual(first.structure_path.stat().st_mtime_ns, mtime_before)
            self.assertEqual(len(list(tmp_path.glob("run_*"))), 1)

    def test_write_render_package_conflict_raises(self) -> None:
        score, path = _tiny_score()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            midi_path = tmp_path / "tmp.mid"
            edo = EDO(EDOConfig(n=12, base_tuning=0, pitch_bend_range=2,
                                microtonal_rendering_method=MicrotonalRendering.MPE))
            render_midi(score, edo, str(midi_path))
            manifest = RunManifest(seed=1, config_dump={"edo": 12})
            package = write_render_package(
                tmp_path,
                score=score,
                midi_path=midi_path,
                manifest=manifest,
                path=path,
                edo=12,
                run_id="conflict1",
            )
            # Corrupt identity recorded in the existing package without renaming the dir.
            manifest_data = json.loads(package.manifest_path.read_text(encoding="utf-8"))
            manifest_data["render_package"]["package_hash"] = "0" * 64
            package.manifest_path.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")
            with self.assertRaises(ContractViolation):
                write_render_package(
                    tmp_path,
                    score=score,
                    midi_path=midi_path,
                    manifest=manifest,
                    path=path,
                    edo=12,
                    run_id="conflict1",
                )

    def test_structure_marks_19edo_lead_microtonal(self) -> None:
        score, path = _tiny_score(edo=19)
        structure = build_structure(score, path, edo=19, base_tuning=0.0)
        by_name = {t["name"]: t for t in structure.tracks}
        self.assertTrue(by_name["lead"]["microtonal"] or by_name["bass"]["microtonal"])

    def test_structure_19edo_drums_never_microtonal(self) -> None:
        """GM drum keys must not trip microtonal detection in non-12 EDO."""
        events = (
            NoteEvent(ton=0, toff=240, h=19, v=0.8, track="bass"),
            NoteEvent(ton=0, toff=120, h=1, v=0.7, track="lead"),
            NoteEvent(ton=0, toff=60, h=36, v=0.9, track="drums"),
            NoteEvent(ton=60, toff=120, h=38, v=0.9, track="drums"),
            NoteEvent(ton=120, toff=180, h=42, v=0.85, track="drums"),
        )
        score = Score(note_events=events, ticks_per_beat=480, tempo_bpm=120.0)
        path = (
            BeatState(
                meter_id=0,
                beat_in_bar=0,
                boundary_lvl=2,
                key_id=0,
                chord_id=0,
                role_id=0,
                head_id=0,
                groove_id=0,
            ),
        )
        structure = build_structure(score, path, edo=19, base_tuning=0.0)
        by_name = {t["name"]: t for t in structure.tracks}
        self.assertTrue(by_name["drums"]["percussion"])
        self.assertFalse(by_name["drums"]["microtonal"])
        self.assertTrue(by_name["lead"]["microtonal"])

    def test_cli_generate_emits_render_package(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "aimusic.app.cli",
                    "generate",
                    "--seed",
                    "11",
                    "--beats",
                    "4",
                    "--out",
                    tmp,
                ],
                capture_output=True,
                text=True,
                cwd=str(Path(__file__).resolve().parents[1]),
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr + result.stdout)
            packages = list(Path(tmp).glob("run_*"))
            self.assertEqual(len(packages), 1, msg=result.stdout)
            load_render_package(packages[0])


class TestPackageIntegrity(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.score, self.path = _tiny_score()
        self.midi = self.root / 'source.mid'
        render_midi(self.score, EDO(EDOConfig(n=12)), str(self.midi))
        self.manifest = RunManifest(seed=1, config_dump={})
        self.package = self.write()

    def write(self):
        return write_render_package(
            self.root, score=self.score, midi_path=self.midi,
            manifest=self.manifest, path=self.path, run_id='integrity',
        )

    def metadata(self):
        return json.loads(self.package.manifest_path.read_text())

    def save_metadata(self, data):
        self.package.manifest_path.write_text(json.dumps(data))

    def test_new_digests_cover_payload_bytes_not_manifest(self) -> None:
        import hashlib

        digests = self.metadata()['render_package']['file_digests']
        self.assertEqual(set(digests), {
            'score.json', 'score.mid', 'structure.json', 'tuning.json', 'beatstates.jsonl',
        })
        for name, digest in digests.items():
            self.assertEqual(digest, hashlib.sha256((self.package.root / name).read_bytes()).hexdigest())
        # Provenance edits are deliberately not self-hashed.
        data = self.metadata()
        data['audio'] = {'status': 'pending'}
        self.save_metadata(data)
        self.assertEqual(load_render_package(self.package.root).content_hash, self.package.content_hash)

    def test_v1_serialization_identity_and_single_package_aliases(self) -> None:
        import hashlib

        source = hashlib.sha256(
            json.dumps(self.score.to_dict(), sort_keys=True).encode()
            + self.midi.read_bytes() + b"12"
        ).hexdigest()
        package = hashlib.sha256(
            source.encode()
            + self.package.structure_path.read_bytes()
            + self.package.tuning_path.read_bytes()
            + (self.package.root / "beatstates.jsonl").read_bytes()
        ).hexdigest()
        self.assertEqual(self.package.source_hash, source)
        self.assertEqual(self.package.content_hash, package)
        original = self.metadata()
        for alias in ("package_hash", "content_hash"):
            data = json.loads(json.dumps(original))
            data["render_package"].pop(alias)
            data["render_package"].pop("file_digests")
            self.save_metadata(data)
            # v1 canonical JSON ignores indentation, but not metadata values.
            structure = json.loads(self.package.structure_path.read_text())
            self.package.structure_path.write_text(json.dumps(structure))
            self.assertEqual(load_render_package(self.package.root).content_hash, package)

    def test_corrupt_payload_reports_file_expected_and_actual(self) -> None:
        for name in ('score.json', 'score.mid', 'structure.json', 'tuning.json', 'beatstates.jsonl'):
            with self.subTest(name=name):
                path = self.package.root / name
                original = path.read_bytes()
                path.write_bytes(original + b' ')
                with self.assertRaisesRegex(ContractViolation, rf'{name}.*expected=.*actual='):
                    load_render_package(self.package.root)
                path.write_bytes(original)

    def test_legacy_identity_and_aggregate_group_diagnostics(self) -> None:
        data = self.metadata()
        data['render_package'].pop('file_digests', None)
        self.save_metadata(data)
        loaded = load_render_package(self.package.root)
        self.assertEqual(loaded.content_hash, self.package.content_hash)
        for name, group in (('score.json', 'source'), ('score.mid', 'source'),
                            ('structure.json', 'package'), ('tuning.json', 'package'),
                            ('beatstates.jsonl', 'package')):
            with self.subTest(name=name):
                path = self.package.root / name
                original = path.read_bytes()
                if name.endswith('.json'):
                    payload = json.loads(original)
                    payload['corruption'] = True
                    path.write_text(json.dumps(payload))
                else:
                    path.write_bytes(original + b' ')
                with self.assertRaisesRegex(ContractViolation, rf'{group} aggregate.*expected=.*actual=') as error:
                    load_render_package(self.package.root)
                self.assertIn(name, str(error.exception))
                path.write_bytes(original)

    def test_manifest_aliases_and_structure_source_must_agree(self) -> None:
        original = self.metadata()
        for field in ('source_hash', 'package_hash', 'content_hash'):
            with self.subTest(field=field):
                data = self.metadata()
                data['render_package'][field] = '0' * 64
                self.save_metadata(data)
                with self.assertRaisesRegex(ContractViolation, 'expected=.*actual='):
                    load_render_package(self.package.root)
                self.save_metadata(original)
        data = self.metadata()
        data['render_package'].pop('file_digests', None)
        self.save_metadata(data)
        structure = json.loads(self.package.structure_path.read_text())
        structure['source_hash'] = 'sha256:' + '0' * 64
        self.package.structure_path.write_text(json.dumps(structure))
        with self.assertRaisesRegex(ContractViolation, r'structure.source_hash.*expected=.*actual='):
            load_render_package(self.package.root)

    def test_reused_corrupt_directory_is_not_returned_or_overwritten(self) -> None:
        self.package.midi_path.write_bytes(b'corrupt')
        with self.assertRaisesRegex(ContractViolation, 'score.mid'):
            self.write()
        self.assertEqual(self.package.midi_path.read_bytes(), b'corrupt')

    def test_absent_legacy_beatstates_only_matches_empty_aggregate(self) -> None:
        data = self.metadata()
        data['render_package'].pop('file_digests', None)
        self.save_metadata(data)
        (self.package.root / 'beatstates.jsonl').unlink()
        with self.assertRaisesRegex(ContractViolation, 'package aggregate'):
            load_render_package(self.package.root)
        empty = write_render_package(
            self.root, score=self.score, midi_path=self.midi, manifest=self.manifest,
            path=(), run_id='empty',
        )
        data = json.loads(empty.manifest_path.read_text())
        data['render_package'].pop('file_digests', None)
        empty.manifest_path.write_text(json.dumps(data))
        (empty.root / 'beatstates.jsonl').unlink()
        self.assertEqual(load_render_package(empty.root).content_hash, empty.content_hash)


if __name__ == "__main__":
    unittest.main()
