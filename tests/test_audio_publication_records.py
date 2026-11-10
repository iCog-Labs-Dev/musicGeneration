"""Record, staged-input and caller contracts for shared publication."""
from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


class TestPublicationRecords(unittest.TestCase):
    def test_records_are_frozen_and_serialization_is_canonical(self):
        from aimusic.audio.publication import FileIdentity, Ownership, Journal, Phase, canonical
        identity = FileIdentity(7, 11)
        owned = Ownership('café.wav', identity)
        record = Journal('fixed-id', ('café.wav',), False, Phase.PREPARED,
                         staged=(owned,))
        for value, field, replacement in ((identity, 'inode', 12), (owned, 'name', 'other'),
                                           (record, 'phase', Phase.COMMITTED)):
            with self.assertRaises(dataclasses.FrozenInstanceError):
                setattr(value, field, replacement)
        serialized = canonical(record)
        self.assertEqual(serialized, canonical(record))
        self.assertEqual(json.loads(serialized)['staged'][0]['identity'], {'device': 7, 'inode': 11})
        self.assertEqual(canonical({'z': 2, 'a': {'y': 1, 'x': 0}}),
                         canonical({'a': {'x': 0, 'y': 1}, 'z': 2}))
        with self.assertRaises(ValueError):
            canonical({'bad': float('nan')})

    def test_staged_symlink_refused_without_private_residue(self):
        from aimusic.audio.publication import pin_output
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp).resolve()
            output, staging = base / 'out', base / 'stage'
            output.mkdir()
            staging.mkdir()
            source = base / 'source'
            source.write_bytes(b'PRIVATE')
            try:
                (staging / 'a.wav').symlink_to(source)
            except OSError as exc:
                self.skipTest(f'symlinks unavailable: {exc}')
            with pin_output(output) as root:
                with self.assertRaises((ValueError, OSError)):
                    root.publish(staging, ['a.wav'], replace=False)
            self.assertEqual(list(output.iterdir()), [])
            self.assertEqual(source.read_bytes(), b'PRIVATE')


class TestPreviewPublication(unittest.TestCase):
    def test_existing_output_is_not_overwritten(self):
        import mido
        from aimusic.audio.preview import render_midi_preview
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            source, output = base / 'source.mid', base / 'preview.wav'
            midi = mido.MidiFile()
            midi.tracks.append(mido.MidiTrack([
                mido.Message('note_on', note=60), mido.Message('note_off', note=60, time=120)]))
            midi.save(source)
            output.write_bytes(b'KEEP USER AUDIO')
            before = source.read_bytes()
            with self.assertRaises(FileExistsError):
                render_midi_preview(source, output)
            self.assertEqual(output.read_bytes(), b'KEEP USER AUDIO')
            self.assertEqual(source.read_bytes(), before)

    @unittest.skipIf(os.name == 'nt', 'POSIX fault injection; Windows backend pending group 3')
    def test_failed_publication_leaves_source_and_unrelated_bytes(self):
        import mido
        from aimusic.audio.preview import render_midi_preview
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            source, output = base / 'source.mid', base / 'preview.wav'
            midi = mido.MidiFile()
            midi.tracks.append(mido.MidiTrack([
                mido.Message('note_on', note=60), mido.Message('note_off', note=60, time=120)]))
            midi.save(source)
            before = source.read_bytes()
            (base / 'keep').write_bytes(b'KEEP')
            with patch('os.link', side_effect=OSError('publish failed')):
                with self.assertRaisesRegex(OSError, 'publish failed'):
                    render_midi_preview(source, output)
            self.assertEqual({p.name for p in base.iterdir()}, {'source.mid', 'keep'})
            self.assertEqual(source.read_bytes(), before)
            self.assertEqual((base / 'keep').read_bytes(), b'KEEP')

class TestAdditionalPublicationBoundaries(unittest.TestCase):
    def test_cache_restore_rejects_portable_aliases_before_reading_cache(self):
        from aimusic.audio.cache import restored_entry
        with tempfile.TemporaryDirectory() as tmp:
            for names in (['mix.wav', 'MIX.WAV'], ['CON.wav'], ['../escape.wav']):
                with self.subTest(names=names), self.assertRaises(ValueError):
                    with restored_entry(Path(tmp), 'unused', names):
                        self.fail('Unsafe cache plan accepted')

    def test_standalone_mix_never_overwrites_a_user_file_or_input(self):
        import numpy as np
        import soundfile as sf
        from aimusic.audio.render.stems import rough_mix
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            source, output = base / 'stem.wav', base / 'mix.wav'
            sf.write(source, np.zeros(128), 48000)
            before = source.read_bytes()
            output.write_bytes(b'KEEP USER MIX')
            for path in (output, source):
                with self.subTest(path=path), self.assertRaises(FileExistsError):
                    rough_mix([source], output_path=path)
            self.assertEqual(source.read_bytes(), before)
            self.assertEqual(output.read_bytes(), b'KEEP USER MIX')

    @unittest.skipIf(os.name == 'nt', 'POSIX backend primitive boundary')
    def test_missing_required_posix_primitive_is_actionable(self):
        from aimusic.audio.publication import pin_output
        with tempfile.TemporaryDirectory() as tmp, patch.object(os, 'O_NOFOLLOW'):
            del os.O_NOFOLLOW
            with self.assertRaisesRegex(RuntimeError, 'unavailable.*Use a supported'):
                with pin_output(tmp):
                    self.fail('Missing no-follow primitive accepted')
            self.assertEqual(list(Path(tmp).iterdir()), [])

    @unittest.skipIf(os.name == 'nt', 'POSIX ownership/backup rollback boundary')
    def test_replacement_rollback_preserves_substitution_and_only_old_copy(self):
        from aimusic.audio.publication import pin_output
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            output, staging = base / 'out', base / 'stage'
            output.mkdir()
            staging.mkdir()
            (output / 'a.wav').write_bytes(b'ONLY OLD COPY')
            (staging / 'a.wav').write_bytes(b'NEW')
            intruder = base / 'intruder'
            intruder.write_bytes(b'UNOWNED')
            with self.assertRaisesRegex(OSError, 'recovery files retained'):
                with pin_output(output) as root:
                    with root.publication(staging, ['a.wav'], replace=True):
                        intruder.replace(output / 'a.wav')
                        raise RuntimeError('caller failure')
            self.assertEqual((output / 'a.wav').read_bytes(), b'UNOWNED')
            self.assertTrue(any(p.read_bytes() == b'ONLY OLD COPY' for p in output.rglob('backup-*')))
            with self.assertRaisesRegex(RuntimeError, 'transaction residue'):
                with pin_output(output):
                    self.fail('Retained recovery was ignored')

    @unittest.skipIf(os.name == 'nt', 'POSIX groove leaf-link replacement semantics')
    def test_replacement_can_replace_unrelated_leaf_link_and_restore_it_on_failure(self):
        from aimusic.audio.publication import pin_output
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            output, staging = base / 'out', base / 'stage'
            output.mkdir()
            staging.mkdir()
            target = base / 'unrelated'
            target.write_bytes(b'UNRELATED TARGET')
            leaf = output / 'a.mid'
            leaf.symlink_to(target)
            (staging / 'a.mid').write_bytes(b'NEW MIDI')
            with self.assertRaisesRegex(RuntimeError, 'caller failure'):
                with pin_output(output) as root:
                    with root.publication(staging, ['a.mid'], replace=True):
                        self.assertEqual(leaf.read_bytes(), b'NEW MIDI')
                        raise RuntimeError('caller failure')
            self.assertTrue(leaf.is_symlink())
            self.assertEqual(leaf.readlink(), target)
            with pin_output(output) as root:
                root.publish(staging, ['a.mid'], replace=True)
            self.assertFalse(leaf.is_symlink())
            self.assertEqual(leaf.read_bytes(), b'NEW MIDI')
            self.assertEqual(target.read_bytes(), b'UNRELATED TARGET')
            self.assertEqual([p.name for p in output.iterdir()], ['a.mid'])

    @unittest.skipIf(os.name == 'nt', 'POSIX case-sensitive directory race')
    def test_late_casefold_collision_preserves_the_other_writer(self):
        from aimusic.audio.publication import pin_output
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            output, staging = base / 'out', base / 'stage'
            output.mkdir()
            staging.mkdir()
            (staging / 'mix.wav').write_bytes(b'OURS')
            link = os.link

            def collision(src, dst, **kwargs):
                if dst == 'mix.wav':
                    (output / 'MIX.WAV').write_bytes(b'OTHER WRITER')
                return link(src, dst, **kwargs)

            with patch('os.link', side_effect=collision):
                with self.assertRaises(FileExistsError):
                    with pin_output(output) as root:
                        root.publish(staging, ['mix.wav'], replace=False)
            self.assertEqual({p.name: p.read_bytes() for p in output.iterdir()}, {'MIX.WAV': b'OTHER WRITER'})
