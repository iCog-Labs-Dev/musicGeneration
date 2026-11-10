"""Offline wheel smoke test: installed resources, no editable/checkout imports."""

from __future__ import annotations

import os
from email import message_from_bytes
from zipfile import ZipFile
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


class TestAudioWheel(unittest.TestCase):
    def test_installed_profiles_outside_checkout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            source, wheels, installed = (root / name for name in ("source", "wheels", "installed"))
            source.mkdir()
            shutil.copytree(REPO_ROOT / "aimusic", source / "aimusic",
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            for name in ("pyproject.toml", "readme.md"):
                shutil.copy2(REPO_ROOT / name, source / name)
            env = {key: value for key, value in os.environ.items()
                   if key not in {"PYTHONPATH", "PYTHONHOME"}}

            def run(*args: str) -> None:
                result = subprocess.run(
                    [sys.executable, *args], cwd=root, env=env,
                    text=True, capture_output=True, timeout=120,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

            run("-I", "-m", "pip", "wheel", "--no-deps", "--no-build-isolation",
                "--no-index", "--wheel-dir", str(wheels), str(source))
            built = list(wheels.glob("aimusic-*.whl"))
            self.assertEqual(len(built), 1)
            with ZipFile(built[0]) as wheel:
                meta = message_from_bytes(wheel.read('aimusic-0.1.0.dist-info/METADATA'))
            requirements = meta.get_all('Requires-Dist', [])
            self.assertEqual([r for r in requirements if ';' not in r], ['numpy', 'mido'])
            self.assertIn('pyfluidsynth==1.3.4; extra == "audio-render"', requirements)
            self.assertIn('aimusic[audio]; extra == "audio-render"', requirements)
            run("-I", "-m", "pip", "install", "--no-deps", "--no-index",
                "--target", str(installed), str(built[0]))
            # Supply only the already-installed YAML dependency, offline. Do not
            # expose the host site-packages (which may contain editable aimusic).
            shutil.copytree(Path(yaml.__file__).parent, installed / "yaml")
            shutil.rmtree(source)
            run("-I", "-S", "-c", textwrap.dedent("""\
                import os
                import sys
                from pathlib import Path
                installed, checkout = map(Path, sys.argv[1:])
                sys.path.insert(0, str(installed))
                assert 'PYTHONPATH' not in os.environ
                assert not Path.cwd().is_relative_to(checkout)
                assert all(not Path(p).resolve().is_relative_to(checkout) for p in sys.path)
                import aimusic
                from aimusic.audio.config import load_audio_config
                assert Path(aimusic.__file__).is_relative_to(installed)
                for profile, backend, fallback in (
                    (None, 'simple', True), ('preview', 'simple', True),
                    ('listening', 'fluidsynth', False),
                ):
                    config = load_audio_config(profile)
                    raw = config.to_dict()
                    assert config.source_path is None
                    assert raw['render']['backend'] == backend
                    assert raw['render']['allow_simple_fallback'] is fallback
                    assert not raw['stages']['restyle']
                    assert not raw['groove']['models']['midi_gpt']
                    assert raw['render']['routing']['comping']['default'] == backend
                assert not any(name in sys.modules for name in (
                    'fluidsynth', 'pretty_midi', 'soundfile', 'torch',
                ))
                # The report module works with stdlib only, from the installed
                # wheel under -I -S (no audio/native dependencies or checkout).
                import base64
                from html.parser import HTMLParser
                import runpy
                import wave
                names = ('fixture-12', 'generated-12', 'reference-19', 'multitrack-19')
                expected = []
                for i, name in enumerate(names):
                    path = Path('four-workflows', name, 'cold', 'stems', 'rough_mix.wav')
                    path.parent.mkdir(parents=True)
                    with wave.open(str(path), 'wb') as stream:
                        stream.setparams((1, 2, 8000, 0, 'NONE', 'none'))
                        stream.writeframes(bytes([i, 0]) * 80)
                    expected.append(path.read_bytes())
                sys.argv = ['listening_review', '--evidence-root', '.', '--out', 'review.html']
                runpy.run_module('aimusic.audio.listening_review', run_name='__main__')
                class Players(HTMLParser):
                    sources = []
                    def handle_starttag(self, tag, attrs):
                        if tag == 'audio':
                            self.sources.append(dict(attrs)['src'])
                players = Players()
                players.feed(Path('review.html').read_text())
                assert len(players.sources) == 4
                assert [base64.b64decode(s.split(',', 1)[1], validate=True)
                        for s in players.sources] == expected
                print('installed wheel profiles and self-contained listening review OK')
                """), str(installed), str(REPO_ROOT))
            # Exercise the installed module command and real preview CLI with
            # host audio dependencies, still without checkout/editable aimusic.
            run("-I", "-c", textwrap.dedent("""\
                import os
                import runpy
                import sys
                from pathlib import Path
                installed, checkout = map(Path, sys.argv[1:])
                sys.path.insert(0, str(installed))
                assert 'PYTHONPATH' not in os.environ
                assert all(not Path(p).resolve().is_relative_to(checkout) for p in sys.path)
                import aimusic
                assert Path(aimusic.__file__).is_relative_to(installed)
                os.environ['AIMUSIC_AUDIO_BACKEND'] = 'native'
                os.environ.pop('AIMUSIC_SOUNDFONT', None)
                for name in ('fixture-12', 'reference-19', 'multitrack-19'):
                    sys.argv = ['demo_sources', name, '--out', name]
                    runpy.run_module('aimusic.audio.demo_sources', run_name='__main__')
                    assert Path(name, 'source.mid').is_file()
                    assert Path(name, 'expected.json').is_file()
                sys.argv = ['aimusic', 'render-midi', 'fixture-12/source.mid',
                            '--profile', 'preview', '--out', 'preview', '--no-validate-only']
                try:
                    runpy.run_module('aimusic.app.cli', run_name='__main__')
                except SystemExit as exc:
                    assert exc.code == 0
                import soundfile
                pcm, rate = soundfile.read('preview/stems/rough_mix.wav')
                assert rate == 48000 and len(pcm) == 780000
                print('installed demo generator and preview OK')
                """), str(installed), str(REPO_ROOT))
