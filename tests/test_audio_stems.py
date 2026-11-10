"""Logical event isolation, safe publication and simple preview regressions.

Retains Bug A from PR #47 review: duplicate instrument names must produce
distinct WAV files and the rough mix must sum each stem exactly once.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path


def _audio_render_available() -> bool:
    try:
        import pretty_midi  # noqa: F401
        import soundfile  # noqa: F401
        return True
    except ImportError:
        return False


def _make_two_pianos_one_bass_midi(path: Path) -> Path:
    """Create a MIDI with two 'Piano' instruments and one 'Bass' instrument."""
    import pretty_midi as pm

    midi = pm.PrettyMIDI(initial_tempo=120.0)
    # Piano #1: middle C, one quarter note
    inst_a = pm.Instrument(program=0, is_drum=False, name="Piano")
    inst_a.notes.append(pm.Note(velocity=80, pitch=60, start=0.0, end=0.5))
    # Piano #2: same role name, distinct pitch (E)
    inst_b = pm.Instrument(program=0, is_drum=False, name="Piano")
    inst_b.notes.append(pm.Note(velocity=80, pitch=64, start=0.0, end=0.5))
    # Bass: distinct role
    inst_c = pm.Instrument(program=32, is_drum=False, name="Bass")
    inst_c.notes.append(pm.Note(velocity=80, pitch=40, start=0.0, end=0.5))
    midi.instruments.extend([inst_a, inst_b, inst_c])
    midi.write(str(path))
    return path


@unittest.skipUnless(_audio_render_available(), "requires pip install -e '.[audio]'")
class TestRenderRoleStemsDuplicates(unittest.TestCase):
    def test_duplicate_role_names_produce_distinct_wav_files(self) -> None:
        from aimusic.audio.render.stems import render_role_stems

        with tempfile.TemporaryDirectory() as tmp_s:
            tmp = Path(tmp_s)
            midi_path = tmp / "src.mid"
            _make_two_pianos_one_bass_midi(midi_path)

            out_dir = tmp / "stems"
            stems = render_role_stems(midi_path, output_dir=out_dir)

            # Three role WAVs on disk
            self.assertTrue((out_dir / "track_0001_piano.wav").is_file())
            self.assertTrue((out_dir / "track_0002_piano.wav").is_file())
            self.assertTrue((out_dir / "track_0003_bass.wav").is_file())

            # Dict keys distinct
            self.assertIn("track_0001_piano", stems)
            self.assertIn("track_0002_piano", stems)
            self.assertIn("track_0003_bass", stems)

            # Dict values point to distinct paths
            self.assertNotEqual(stems["track_0001_piano"], stems["track_0002_piano"])
            self.assertEqual(stems["track_0001_piano"].name, "track_0001_piano.wav")
            self.assertEqual(stems["track_0002_piano"].name, "track_0002_piano.wav")
            self.assertEqual(stems["track_0003_bass"].name, "track_0003_bass.wav")

    def test_duplicate_role_rough_mix_sums_each_stem_once(self) -> None:
        """Regression: the second piano must not be summed twice into rough_mix.

        Build a ground-truth mix by rendering each instrument to its own WAV
        (via ``render_stem`` with ``instrument_index``) so each piano truly has
        its own audio. The rough_mix written by ``render_role_stems`` must
        equal the sum of those distinct stems — modulo the global peak
        attenuation-only 0.95 ceiling that both paths apply.
        """
        import numpy as np
        import pretty_midi as pm
        import soundfile as sf

        from aimusic.audio.render.simple_r import render_stem
        from aimusic.audio.render.stems import render_role_stems

        with tempfile.TemporaryDirectory() as tmp_s:
            tmp = Path(tmp_s)
            midi_path = tmp / "src.mid"
            _make_two_pianos_one_bass_midi(midi_path)

            # Ground truth: render each instrument to its OWN WAV.
            truth_dir = tmp / "truth"
            truth_dir.mkdir()
            truth_paths: dict[str, Path] = {}
            midi = pm.PrettyMIDI(str(midi_path))
            for idx, inst in enumerate(midi.instruments):
                p = truth_dir / f"inst_{idx}.wav"
                render_stem(
                    midi_path, output_path=p, instrument_index=idx, sample_rate=48000
                )
                truth_paths[f"inst_{idx}"] = p

            # Independent sum and attenuation-only 0.95 ceiling oracle.
            truth_waves: list[np.ndarray] = []
            sr: int | None = None
            for p in truth_paths.values():
                data, file_sr = sf.read(str(p), always_2d=False)
                truth_waves.append(np.asarray(data, dtype=np.float64))
                sr = int(file_sr)
            assert sr is not None
            max_len = max(w.shape[0] for w in truth_waves)
            expected = np.zeros(max_len, dtype=np.float64)
            for w in truth_waves:
                expected[: w.shape[0]] += w
            expected *= min(1.0, 0.95 / np.max(np.abs(expected)))

            # Now exercise the buggy path
            out_dir = tmp / "stems"
            stems = render_role_stems(midi_path, output_dir=out_dir)

            mix_path = stems["rough_mix"]
            mix, mix_sr = sf.read(str(mix_path), always_2d=False)
            mix = np.asarray(mix, dtype=np.float64)
            self.assertEqual(int(mix_sr), sr)

            # Compare shapes and audio similarity. The bug (double-add of
            # piano_2 into the mix) inflates the rough_mix's energy by ~3 dB
            # relative to the ground-truth sum of three distinct stems.
            n = min(mix.shape[0], expected.shape[0])
            diff = mix[:n] - expected[:n]
            rms_diff = float(np.sqrt(np.mean(diff * diff) + 1e-12))
            rms_truth = float(np.sqrt(np.mean(expected[:n] * expected[:n]) + 1e-12))
            ratio = rms_diff / max(rms_truth, 1e-12)
            self.assertLess(
                ratio,
                0.10,
                msg=(
                    f"rough_mix differs from ground-truth sum by {ratio:.3f} "
                    f"(normalized RMS diff); piano_2 likely summed twice "
                    "or written over piano (regression of Bug A)."
                ),
            )


def _event_track(*events):
    import mido
    result = mido.MidiTrack()
    previous = 0
    for tick, message in events:
        result.append(message.copy(time=tick - previous))
        previous = tick
    return result


def _logical_fixture():
    """Two member channels, a drum channel and setup on a separate track."""
    import mido
    midi = mido.MidiFile(type=1, ticks_per_beat=480)
    setup = [(0, mido.MetaMessage('set_tempo', tempo=500000)),
             (0, mido.MetaMessage('time_signature', numerator=4, denominator=4)),
             (0, mido.Message('sysex', data=(0x7e, 0x7f, 8, 2, 0, 1, 60, 60, 0, 0)))]
    for channel in (2, 3):
        for cc, value in ((0, 1), (32, 2), (101, 0), (100, 0), (6, 12), (38, 50)):
            setup.append((0, mido.Message('control_change', channel=channel, control=cc, value=value)))
        setup.append((0, mido.Message('program_change', channel=channel, program=81)))
    setup.extend([(480, mido.MetaMessage('set_tempo', tempo=750000)),
                  (480, mido.MetaMessage('time_signature', numerator=3, denominator=4)),
                  (1000, mido.MetaMessage('end_of_track'))])
    voices = [(0, mido.MetaMessage('track_name', name='misleading bass'))]
    for channel, pitch in ((2, 60), (3, 67)):
        voices.extend([(100, mido.Message('pitchwheel', channel=channel, pitch=172)),
                       (100, mido.Message('note_on', channel=channel, note=pitch, velocity=90)),
                       (150, mido.Message('aftertouch', channel=channel, value=42)),
                       (175, mido.Message('polytouch', channel=channel, note=pitch, value=51)),
                       (200, mido.Message('control_change', channel=channel, control=64, value=127)),
                       (500, mido.Message('note_off', channel=channel, note=pitch)),
                       (600, mido.Message('pitchwheel', channel=channel, pitch=200)),
                       (700, mido.Message('control_change', channel=channel, control=64, value=0))])
    voices.sort(key=lambda e: e[0])
    voices.append((1000, mido.MetaMessage('end_of_track')))
    midi.tracks = [_event_track(*setup), _event_track(*voices), _event_track(
        (0, mido.MetaMessage('track_name', name='kit')),
        (100, mido.Message('note_on', channel=9, note=36, velocity=100)),
        (400, mido.Message('note_off', channel=9, note=36)),
        # Global events need not be in the conductor track.
        (480, mido.MetaMessage('key_signature', key='D')),
        (1000, mido.MetaMessage('end_of_track')))]
    return midi


def _channel_trajectory(source, channel):
    """Independent state oracle: banks/programs, RPN cents, pedal and pitches."""
    program, bank, lsb, rpn, bend, width, cents, pedal = 0, 0, 0, [127, 127], 0, 2, 0, 0
    result = []
    for event in source.events:
        m = event.message
        if event.channel != channel:
            continue
        if m.type == 'program_change':
            program = m.program
        if m.type == 'pitchwheel':
            bend = m.pitch
        if m.type == 'control_change':
            if m.control == 0:
                bank = m.value
            if m.control == 32:
                lsb = m.value
            if m.control == 101:
                rpn[0] = m.value
            if m.control == 100:
                rpn[1] = m.value
            if m.control == 6 and rpn == [0, 0]:
                width = m.value
            if m.control == 38 and rpn == [0, 0]:
                cents = m.value
            if m.control == 64:
                pedal = m.value
        result.append((event.tick, program, bank, lsb, pedal,
                       bend * (width + cents / 100) / 8192))
    return result


@unittest.skipUnless(_audio_render_available(), "requires pip install -e '.[audio]'")
class TestLogicalStemIsolation(unittest.TestCase):
    def test_complete_channel_and_global_streams_survive_serialization(self):
        import io
        import mido
        from aimusic.audio.midi_events import MidiEvents
        from aimusic.audio.render.stems import isolate_source_stems
        source = MidiEvents.from_mido(_logical_fixture())
        analysis = {'tracks': [
            {'source_track_index': 1, 'role': 'comp', 'role_source': 'structure.json'},
            {'source_track_index': 2, 'role': 'drums', 'role_source': 'structure.json'}]}
        stems = isolate_source_stems(source, analysis=analysis)
        self.assertEqual([s.source_track_index for s in stems], [1, 2])
        self.assertEqual([s.role for s in stems], ['comping', 'drums'])
        self.assertEqual(stems[0].role_source, 'structure.json')
        self.assertEqual(stems[0].channels, (2, 3))
        self.assertEqual(stems[1].channels, (9,))
        for stem in stems:
            buffer = io.BytesIO()
            stem.midi.to_mido().save(file=buffer)
            buffer.seek(0)
            actual = MidiEvents.from_mido(mido.MidiFile(file=buffer))
            signature = lambda e: (e.tick, e.track, e.message.dict())
            expected = [signature(e) for e in source.events
                        if e.channel is None or e.channel in stem.channels]
            self.assertEqual([signature(e) for e in actual.events], expected)
            self.assertEqual(actual.tempo_map, source.tempo_map)
            note_signature = lambda n: (n.track, n.channel, n.pitch, n.velocity,
                                        n.start, n.end, n.sounding_end)
            self.assertEqual([note_signature(n) for n in actual.notes],
                             [note_signature(n) for n in source.notes if n.track == stem.source_track_index])
            for channel in stem.channels:
                self.assertEqual(_channel_trajectory(actual, channel), _channel_trajectory(source, channel))
        self.assertEqual([n.sounding_end for n in stems[0].midi.notes], [700, 700])
        self.assertAlmostEqual(_channel_trajectory(stems[0].midi, 2)[-1][-1], 200 * 12.5 / 8192)

    def test_isolated_note_ids_remain_valid_event_model_indices(self):
        from aimusic.audio.midi_events import MidiEvents
        from aimusic.audio.render.stems import isolate_source_stems
        source = MidiEvents.from_mido(_logical_fixture())
        for stem in isolate_source_stems(source):
            self.assertEqual([n.id for n in stem.midi.notes], list(range(len(stem.midi.notes))))
            for event in stem.midi.events:
                for note_id in event.note_ids:
                    self.assertEqual(stem.midi.notes[note_id].channel, event.channel)

    def test_unowned_conductor_channel_setup_is_not_discarded(self):
        import mido
        from aimusic.audio.midi_events import MidiEvents
        from aimusic.audio.render.stems import isolate_source_stems
        midi = _logical_fixture()
        setup = mido.Message('control_change', channel=0, control=7, value=91)
        midi.tracks[0].insert(0, setup)
        source = MidiEvents.from_mido(midi)
        for stem in isolate_source_stems(source):
            self.assertIn(setup, [e.message for e in stem.midi.events])

    def test_setup_only_track_without_tempo_does_not_claim_member_channels(self):
        import mido
        from aimusic.audio.midi_events import MidiEvents
        from aimusic.audio.render.stems import isolate_source_stems
        for master in (False, True):
            with self.subTest(master=master):
                setup = [(0, mido.MetaMessage('track_name', name='setup')),
                         (0, mido.Message('control_change', channel=0, control=7, value=91))]
                if master:
                    setup.extend((0, mido.Message('control_change', channel=0, control=cc, value=value))
                                 for cc, value in ((101, 0), (100, 6), (6, 4)))
                setup.append((0, mido.Message('program_change', channel=2, program=81)))
                midi = mido.MidiFile(type=1)
                midi.tracks = [_event_track(*setup), _event_track(
                    (0, mido.MetaMessage('track_name', name='lead')),
                    (1, mido.Message('note_on', channel=2, note=60)),
                    (10, mido.Message('note_off', channel=2, note=60)))]
                source = MidiEvents.from_mido(midi)
                stems = isolate_source_stems(source)
                self.assertEqual([s.source_track_index for s in stems], [1])
                self.assertEqual(stems[0].channels, (2,))
                self.assertEqual(stems[0].midi.events, source.events)

    def test_master_setup_remains_shared_without_a_silent_stem(self):
        import mido
        from aimusic.audio.midi_events import MidiEvents
        from aimusic.audio.render.stems import isolate_source_stems
        midi = mido.MidiFile(type=1)
        setup = [(0, mido.Message('control_change', channel=0, control=cc, value=value))
                 for cc, value in ((101, 0), (100, 6), (6, 4))]
        midi.tracks = [_event_track(*setup)]
        for channel in (2, 3):
            midi.tracks.append(_event_track(
                (1, mido.Message('note_on', channel=channel, note=60)),
                (10, mido.Message('note_off', channel=channel, note=60))))
        source = MidiEvents.from_mido(midi)
        stems = isolate_source_stems(source)
        self.assertEqual([s.source_track_index for s in stems], [1, 2])
        for stem in stems:
            self.assertEqual([e.message for e in stem.midi.events if e.channel == 0],
                             [msg for _, msg in setup])
            self.assertEqual(len(stem.midi.notes), 1)

    def test_named_metadata_track_is_not_an_undeclared_silent_instrument(self):
        import mido
        from aimusic.audio.midi_events import MidiEvents
        from aimusic.audio.render.stems import isolate_source_stems
        midi = _logical_fixture()
        midi.tracks.append(_event_track((0, mido.MetaMessage('track_name', name='session setup'))))
        self.assertEqual([s.source_track_index for s in isolate_source_stems(MidiEvents.from_mido(midi))], [1, 2])

    def test_declared_setup_track_has_authoritative_channel_ownership(self):
        import mido
        from aimusic.audio.midi_events import MidiEvents, MidiEventError
        from aimusic.audio.render.stems import isolate_source_stems
        midi = _logical_fixture()
        midi.tracks.append(_event_track((0, mido.Message('program_change', channel=2))))
        with self.assertRaisesRegex(MidiEventError, '[Oo]wnership.*track.*channel 2'):
            isolate_source_stems(MidiEvents.from_mido(midi), analysis={'tracks': [
                {'source_track_index': 3, 'role': 'pads', 'role_source': 'structure.json'}]})

    def test_rejects_cross_track_channel_setup_ownership(self):
        import mido
        from aimusic.audio.midi_events import MidiEvents, MidiEventError
        from aimusic.audio.render.stems import isolate_source_stems
        for kind in ('program_change', 'pitchwheel', 'control_change', 'note_on'):
            with self.subTest(kind=kind):
                midi = _logical_fixture()
                midi.tracks[2].insert(1, mido.Message(kind, channel=2))
                with self.assertRaisesRegex(MidiEventError, '[Oo]wnership.*track.*channel 2'):
                    isolate_source_stems(MidiEvents.from_mido(midi))

    def test_type_zero_is_one_source_stem_not_one_per_channel(self):
        import mido
        from aimusic.audio.midi_events import MidiEvents
        from aimusic.audio.render.stems import isolate_source_stems
        midi = _logical_fixture()
        midi.type = 0
        midi.tracks = [mido.merge_tracks(midi.tracks)]
        source = MidiEvents.from_mido(midi)
        stems = isolate_source_stems(source)
        self.assertEqual(len(stems), 1)
        self.assertEqual(stems[0].channels, (2, 3, 9))
        self.assertEqual(stems[0].midi.events, source.events)

    def test_declared_silent_track_survives_without_fabricating_notes(self):
        import mido
        from aimusic.audio.midi_events import MidiEvents
        from aimusic.audio.render.stems import isolate_source_stems
        midi = _logical_fixture()
        midi.tracks.append(_event_track((0, mido.MetaMessage('track_name', name='silent'))))
        stems = isolate_source_stems(MidiEvents.from_mido(midi), analysis={'tracks': [
            {'source_track_index': 3, 'role': 'pads', 'role_source': 'structure.json'}]})
        self.assertEqual([s.source_track_index for s in stems], [1, 2, 3])
        self.assertEqual(stems[-1].midi.notes, ())


    def test_19edo_groove_then_serialized_isolation_keeps_expected_pitches(self):
        import io
        import mido
        from aimusic.audio.midi_events import MidiEvents, NoteTiming, retime
        from aimusic.audio.render.stems import isolate_source_stems
        from aimusic.theory.edo import EDO, EDOConfig
        edo = EDO(EDOConfig(n=19, pitch_bend_range=2))
        midi = mido.MidiFile(type=1, ticks_per_beat=480)
        midi.tracks = [_event_track((0, mido.MetaMessage('set_tempo', tempo=500000)),
                                   (480, mido.MetaMessage('set_tempo', tempo=750000)),
                                   (1000, mido.MetaMessage('end_of_track')))]
        events = [(0, mido.MetaMessage('track_name', name='lead'))]
        expected = {}
        for channel, height in enumerate((96, 98, 101), start=2):
            pitch, bend = edo.to_midi(height)
            expected[channel] = edo.config.base_tuning + height * 12 / 19
            for cc, value in ((101, 0), (100, 0), (6, 2), (38, 0)):
                events.append((0, mido.Message('control_change', channel=channel, control=cc, value=value)))
            events.extend([(200, mido.Message('pitchwheel', channel=channel, pitch=bend)),
                           (200, mido.Message('note_on', channel=channel, note=pitch, velocity=80)),
                           (700, mido.Message('note_off', channel=channel, note=pitch))])
        events.sort(key=lambda e: e[0])
        events.append((1000, mido.MetaMessage('end_of_track')))
        midi.tracks.append(_event_track(*events))
        source = MidiEvents.from_mido(midi)
        grooved = retime(source, {n.id: NoteTiming(n.start + 17, n.end + 17, n.velocity) for n in source.notes})
        stem = isolate_source_stems(grooved)[0]
        buffer = io.BytesIO()
        stem.midi.to_mido().save(file=buffer)
        buffer.seek(0)
        isolated = MidiEvents.from_mido(mido.MidiFile(file=buffer))
        self.assertEqual(len(isolated.notes), 3)
        self.assertEqual([(n.channel, n.pitch, n.start, n.end) for n in isolated.notes],
                         [(n.channel, n.pitch, n.start, n.end) for n in grooved.notes])
        bends = {}
        for event in isolated.events:
            msg = event.message
            if msg.type == 'pitchwheel':
                bends[msg.channel] = msg.pitch
            if msg.type == 'note_on' and msg.velocity:
                observed = msg.note + bends[msg.channel] * 2 / 8192
                self.assertLessEqual(abs(observed - expected[msg.channel]) * 100, 1.0)

    def test_invalid_authoritative_identity_fails_instead_of_guessing(self):
        from aimusic.audio.midi_events import MidiEvents, MidiEventError
        from aimusic.audio.render.stems import isolate_source_stems
        source = MidiEvents.from_mido(_logical_fixture())
        for rows in ([{'role': 'lead', 'role_source': 'structure.json'}],
                     [{'source_track_index': 99, 'role': 'lead'}],
                     [{'source_track_index': 1}, {'source_track_index': 1}]):
            with self.subTest(rows=rows), self.assertRaisesRegex(MidiEventError, 'source.track'):
                isolate_source_stems(source, analysis={'tracks': rows})


@unittest.skipUnless(_audio_render_available(), "requires pip install -e '.[audio]'")
class TestStemNamespace(unittest.TestCase):
    def test_untrusted_names_are_unique_contained_indexed_and_mixed(self):
        import json
        import mido
        import numpy as np
        import soundfile as sf
        from aimusic.audio.render.stems import render_role_stems
        names = ['Piano', 'Piano', 'click', 'rough_mix', 'bar_boundaries', '../escape',
                 '/absolute', r'..\escape', 'café', 'cafe', '音色', 'a' * 400, 'role_index']
        with tempfile.TemporaryDirectory() as tmp_s:
            tmp = Path(tmp_s)
            midi = mido.MidiFile(type=1, charset='utf8')
            midi.tracks.append(_event_track((0, mido.MetaMessage('set_tempo', tempo=500000))))
            for i, name in enumerate(names):
                midi.tracks.append(_event_track(
                    (0, mido.MetaMessage('track_name', name=name)),
                    (0, mido.Message('note_on', channel=i, note=40 + i, velocity=70)),
                    (120, mido.Message('note_off', channel=i, note=40 + i))))
            midi_path = tmp / 'src.mid'
            midi.save(midi_path)
            source_bytes = midi_path.read_bytes()
            outputs = render_role_stems(midi_path, output_dir=tmp / 'stems')
            index = json.loads(outputs['role_index'].read_text())
            self.assertEqual(len(index['stems']), len(names))
            self.assertEqual(len({row['id'] for row in index['stems']}), len(names))
            self.assertEqual(sorted(k for ids in index['roles'].values() for k in ids),
                             sorted(row['id'] for row in index['stems']))
            waves = []
            for row in index['stems']:
                path = outputs[row['id']]
                self.assertEqual(path.name, row['wav'])
                self.assertEqual(path.resolve().parent, (tmp / 'stems').resolve())
                self.assertRegex(path.name, r'^track_[0-9]{4}_[a-z0-9_]+\.wav$')
                self.assertLess(len(path.name), 100)
                self.assertTrue((path.parent / row['midi']).is_file())
                waves.append(sf.read(path)[0])
            expected = np.zeros(max(len(w) for w in waves))
            for wave in waves:
                expected[:len(wave)] += wave
            expected *= min(1.0, 0.95 / np.max(np.abs(expected)))
            np.testing.assert_allclose(sf.read(outputs['rough_mix'])[0], expected, atol=1e-7)
            self.assertEqual(midi_path.read_bytes(), source_bytes)
            second = render_role_stems(midi_path, output_dir=tmp / 'repeat')
            self.assertEqual(set(outputs), set(second))
            self.assertEqual(outputs['role_index'].read_bytes(), second['role_index'].read_bytes())

    def test_existing_files_directories_and_symlinks_are_never_overwritten(self):
        from aimusic.audio.render.stems import render_role_stems
        artifacts = ['track_0001_piano.wav', 'track_0001_piano.mid',
                     'track_0001_piano.wav.json', 'click.wav', 'rough_mix.wav',
                     'bar_boundaries.json', 'role_index.json']
        for name in artifacts:
            for kind in ('file', 'directory', 'inside_link', 'outside_link', 'dangling_link'):
                with self.subTest(name=name, kind=kind), tempfile.TemporaryDirectory() as tmp_s:
                    tmp = Path(tmp_s)
                    path = _make_two_pianos_one_bass_midi(tmp / 'src.mid')
                    out = tmp / 'stems'
                    out.mkdir()
                    target = out / name
                    outside = tmp / 'outside'
                    outside.write_bytes(b'KEEP OUTSIDE')
                    if kind == 'file':
                        target.write_bytes(b'KEEP')
                    elif kind == 'directory':
                        target.mkdir()
                    else:
                        link = outside if kind == 'outside_link' else out / 'keep'
                        if kind == 'inside_link':
                            link.write_bytes(b'KEEP')
                        try:
                            target.symlink_to(link)
                        except OSError as exc:
                            if getattr(exc, 'winerror', None) == 1314:
                                self.skipTest('Native symlink privilege unavailable; file/directory and mocked reparse cases remain required')
                            raise
                    before = sorted(out.iterdir())
                    with self.assertRaisesRegex((ValueError, FileExistsError), 'exist|symlink|contain'):
                        render_role_stems(path, output_dir=out)
                    self.assertEqual(sorted(out.iterdir()), before)
                    self.assertEqual(outside.read_bytes(), b'KEEP OUTSIDE')
                    if kind == 'file':
                        self.assertEqual(target.read_bytes(), b'KEEP')
                    if kind.endswith('link'):
                        self.assertTrue(target.is_symlink())

    def test_output_root_symlink_is_refused(self):
        from aimusic.audio.render.stems import render_role_stems
        with tempfile.TemporaryDirectory() as tmp_s:
            tmp = Path(tmp_s)
            path = _make_two_pianos_one_bass_midi(tmp / 'src.mid')
            outside = tmp / 'outside'
            outside.mkdir()
            out = tmp / 'stems'
            try:
                out.symlink_to(outside, target_is_directory=True)
            except OSError as exc:
                if getattr(exc, 'winerror', None) == 1314:
                    self.skipTest('Native symlink privilege unavailable; native junction and mocked reparse cases remain required')
                raise
            with self.assertRaisesRegex(ValueError, 'symlink|reparse'):
                render_role_stems(path, output_dir=out)
            self.assertEqual(list(outside.iterdir()), [])

    def test_simple_render_uses_all_member_channels_and_retains_silent_stem(self):
        import json
        import mido
        import numpy as np
        import soundfile as sf
        from aimusic.audio.midi_events import MidiEvents
        from aimusic.audio.render.stems import render_role_stems
        from aimusic.audio.render.simple_r import render_stem
        with tempfile.TemporaryDirectory() as tmp_s:
            tmp = Path(tmp_s)
            midi = _logical_fixture()
            # C5 now refuses lossy simple rendering. Event-fidelity tests above
            # retain the original MPE/MTS fixture; this preview is direct-12.
            for track in midi.tracks:
                for i in range(len(track) - 1, -1, -1):
                    msg = track[i]
                    if (msg.type in ('pitchwheel', 'sysex') or
                            msg.type == 'control_change' and msg.control in (6, 38, 100, 101)):
                        if i + 1 < len(track):
                            track[i + 1].time += msg.time
                        del track[i]
            midi.tracks.append(_event_track((0, mido.MetaMessage('track_name', name='silent'))))
            path = tmp / 'src.mid'
            midi.save(path)
            analysis = {'tracks': [
                {'source_track_index': 1, 'role': 'comping', 'role_source': 'structure.json'},
                {'source_track_index': 3, 'role': 'pads', 'role_source': 'structure.json'}]}
            outputs = render_role_stems(path, output_dir=tmp / 'stems', analysis=analysis)
            index = json.loads(outputs['role_index'].read_text())
            self.assertEqual(len(index['stems']), 3)
            self.assertEqual(index['roles']['comping'], ['track_0001_misleading_bass'])
            member = index['stems'][0]
            isolated = tmp / 'stems' / member['midi']
            self.assertEqual({n.channel for n in MidiEvents.load(isolated).notes}, {2, 3})
            truth = render_stem(isolated, output_path=tmp / 'truth.wav')
            np.testing.assert_array_equal(sf.read(truth)[0], sf.read(outputs[member['id']])[0])
            self.assertFalse(np.any(sf.read(outputs['track_0003_silent'])[0]))


    def test_orchestrator_does_not_delete_existing_stem_outputs_on_cold_or_warm_copy(self):
        from aimusic.audio.orchestrator import Orchestrator
        with tempfile.TemporaryDirectory() as tmp_s:
            tmp = Path(tmp_s)
            source = _make_two_pianos_one_bass_midi(tmp / 'src.mid')
            cfg = {'stages': {'analysis': False, 'expressivization': False, 'render': True}}
            for warm in (False, True):
                with self.subTest(warm=warm):
                    out = tmp / ('warm' if warm else 'cold')
                    (out / 'stems').mkdir(parents=True)
                    existing = out / 'stems' / 'rough_mix.wav'
                    existing.write_bytes(b'USER MIX')
                    with self.assertRaises(FileExistsError):
                        Orchestrator(cfg, tmp / 'cache').run(source, output_dir=out)
                    self.assertEqual(existing.read_bytes(), b'USER MIX')
                    self.assertEqual(list((out / 'stems').iterdir()), [existing])

    def test_orchestrator_retains_unrelated_files_and_copies_index_on_cache_hit(self):
        from aimusic.audio.orchestrator import Orchestrator
        with tempfile.TemporaryDirectory() as tmp_s:
            tmp = Path(tmp_s)
            source = _make_two_pianos_one_bass_midi(tmp / 'src.mid')
            cfg = {'stages': {'analysis': False, 'expressivization': False, 'render': True}}
            for target in ('cold', 'warm'):
                out = tmp / target
                (out / 'stems').mkdir(parents=True)
                keep = out / 'stems' / 'user.txt'
                keep.write_bytes(b'USER')
                orch = Orchestrator(cfg, tmp / 'cache')
                orch.run(source, output_dir=out)
                self.assertTrue((out / 'stems' / 'role_index.json').is_file())
                self.assertEqual(keep.read_bytes(), b'USER')
                if target == 'warm':
                    self.assertEqual(orch.cache_hits['render'], 1)
            cold_files = {p.name: p.read_bytes() for p in (tmp / 'cold' / 'stems').iterdir()}
            self.assertEqual(cold_files, {p.name: p.read_bytes() for p in (tmp / 'warm' / 'stems').iterdir()})


    def test_initial_default_tempo_is_used_before_a_later_change(self):
        import json
        import mido
        from aimusic.audio.render.stems import render_role_stems
        with tempfile.TemporaryDirectory() as tmp_s:
            tmp = Path(tmp_s)
            midi = mido.MidiFile(type=1, ticks_per_beat=480)
            midi.tracks = [_event_track((480, mido.MetaMessage('set_tempo', tempo=750000))),
                           _event_track((0, mido.Message('note_on', note=60)),
                                        (2400, mido.Message('note_off', note=60)))]
            source = tmp / 'src.mid'
            midi.save(source)
            outputs = render_role_stems(source, output_dir=tmp / 'stems')
            table = json.loads(outputs['bar_boundaries'].read_text())
            self.assertEqual([row['start_sec'] for row in table], [0, 2, 3.5])

    def test_render_or_publication_failure_leaves_no_partial_stems(self):
        import os
        from unittest.mock import patch
        from aimusic.audio.render.stems import render_role_stems
        with tempfile.TemporaryDirectory() as tmp_s:
            tmp = Path(tmp_s)
            source = _make_two_pianos_one_bass_midi(tmp / 'src.mid')
            out = tmp / 'stems'
            with patch('aimusic.audio.render.stems.render_stem', side_effect=RuntimeError('render failed')):
                with self.assertRaisesRegex(RuntimeError, 'render failed'):
                    render_role_stems(source, output_dir=out)
            self.assertEqual(list(out.iterdir()), [])
            if os.name == 'nt':
                from aimusic.audio.windows_native import WindowsAPI
                real_link = WindowsAPI.rename
                mutation = 'aimusic.audio.windows_native.WindowsAPI.rename'
            else:
                real_link = os.link
                mutation = 'os.link'
            calls = 0

            def fail_second_link(*args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError('publication failed')
                return real_link(*args, **kwargs)

            with patch(mutation, new=fail_second_link):
                with self.assertRaisesRegex(OSError, 'publication failed'):
                    render_role_stems(source, output_dir=out)
            self.assertEqual(list(out.iterdir()), [])


if __name__ == '__main__':
    unittest.main()
