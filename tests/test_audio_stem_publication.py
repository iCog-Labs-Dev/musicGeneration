"""POSIX race regressions. Native Windows/ABI tests live in test_audio_windows_native."""
from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tests.test_audio_stems import _audio_render_available, _make_two_pianos_one_bass_midi


@unittest.skipUnless(_audio_render_available(), "requires pip install -e '.[audio]'")
@unittest.skipIf(os.name == 'nt', 'POSIX rename races; Windows pins deny renames')
class TestStemPublicationRaces(unittest.TestCase):
    def test_ancestor_swap_cannot_redirect_staging_publication_or_cleanup(self):
        from aimusic.audio.render import stems
        for phase in ('pin', 'render', 'link'):
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as tmp:
                base = Path(tmp).resolve()
                ancestor, moved, outside = (base / n for n in ('parent', 'moved', 'outside'))
                out = ancestor / 'stems'
                out.mkdir(parents=True)
                (outside / 'stems').mkdir(parents=True)
                source = _make_two_pianos_one_bass_midi(base / 'src.mid')
                swapped = False
                snapshot = {}

                def swap():
                    nonlocal swapped, snapshot
                    if swapped:
                        return
                    swapped = True
                    # A decoy tree makes path-based publication/cleanup dangerous,
                    # rather than merely making the source file disappear.
                    shutil.copytree(out, outside / 'stems', dirs_exist_ok=True)
                    (outside / 'stems' / 'KEEP').write_bytes(b'OUTSIDE')
                    snapshot = {p.relative_to(outside): p.read_bytes()
                                for p in outside.rglob('*') if p.is_file()}
                    ancestor.rename(moved)
                    ancestor.symlink_to(outside, target_is_directory=True)

                real_open, real_render, real_link = os.open, stems.render_stem, os.link

                def opening(path, flags, *args, **kwargs):
                    if phase == 'pin' and Path(path).name == 'stems' and flags & os.O_DIRECTORY:
                        swap()
                    return real_open(path, flags, *args, **kwargs)

                def rendering(*args, **kwargs):
                    result = real_render(*args, **kwargs)
                    if phase == 'render':
                        swap()
                    return result

                def linking(*args, **kwargs):
                    if phase == 'link':
                        swap()
                    return real_link(*args, **kwargs)

                with patch('os.open', side_effect=opening), patch.object(stems, 'render_stem', rendering), \
                        patch('os.link', side_effect=linking):
                    with self.assertRaises((ValueError, OSError)):
                        stems.render_role_stems(source, output_dir=out)
                self.assertTrue(swapped)
                self.assertEqual(list((moved / 'stems').iterdir()), [])
                self.assertEqual({p.relative_to(outside): p.read_bytes()
                                  for p in outside.rglob('*') if p.is_file()}, snapshot)

    def test_cache_copy_ancestor_swap_does_not_leave_or_delete_artifacts(self):
        from aimusic.audio.render import stems
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp).resolve()
            source, ancestor, moved, outside = (base / n for n in ('cache', 'parent', 'moved', 'outside'))
            source.mkdir()
            (source / 'a.wav').write_bytes(b'AUDIO')
            out = ancestor / 'stems'
            out.mkdir(parents=True)
            (outside / 'stems').mkdir(parents=True)
            (outside / 'stems' / 'KEEP').write_bytes(b'OUTSIDE')
            real_copy = shutil.copyfile

            def copying(*args, **kwargs):
                ancestor.rename(moved)
                ancestor.symlink_to(outside, target_is_directory=True)
                return real_copy(*args, **kwargs)

            with patch.object(stems.shutil, 'copyfile', side_effect=copying):
                with self.assertRaises((ValueError, OSError)):
                    stems.copy_stem_artifacts(source, out)
            self.assertEqual(list((moved / 'stems').iterdir()), [])
            self.assertEqual([p.name for p in (outside / 'stems').iterdir()], ['KEEP'])
            self.assertEqual((outside / 'stems' / 'KEEP').read_bytes(), b'OUTSIDE')

    def test_ancestor_swap_during_final_staging_cleanup_rolls_back(self):
        from aimusic.audio.render import stems
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp).resolve()
            source, ancestor, moved, outside = (base / n for n in ('cache', 'parent', 'moved', 'outside'))
            source.mkdir()
            (source / 'a.wav').write_bytes(b'AUDIO')
            out = ancestor / 'stems'
            out.mkdir(parents=True)
            (outside / 'stems').mkdir(parents=True)
            real_unlink = os.unlink
            swapped = False

            def unlink(path, **kwargs):
                nonlocal swapped
                if Path(path).name.startswith('.stems-') and not swapped:
                    swapped = True
                    ancestor.rename(moved)
                    ancestor.symlink_to(outside, target_is_directory=True)
                    (outside / 'stems' / Path(path).name).write_bytes(b'KEEP')
                return real_unlink(path, **kwargs)

            with patch('os.unlink', side_effect=unlink):
                with self.assertRaisesRegex(ValueError, 'identity changed'):
                    stems.copy_stem_artifacts(source, out)
            self.assertTrue(swapped)
            self.assertEqual(list((moved / 'stems').iterdir()), [])
            self.assertEqual([p.read_bytes() for p in (outside / 'stems').iterdir()], [b'KEEP'])
