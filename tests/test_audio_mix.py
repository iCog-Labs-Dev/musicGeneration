"""Common render horizon and bounded, attenuation-only rough mixing."""
from __future__ import annotations

import io
import json
import struct
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import mido
import numpy as np
import soundfile as sf

from aimusic.audio.render import registry
from aimusic.audio.render.stems import prepare_stem_render, render_role_stems, rough_mix
from tests.test_audio_stems import _event_track, _make_two_pianos_one_bass_midi


class TestAlignedMix(unittest.TestCase):
    def test_planned_and_configured_horizon_tail_click_and_silent_stems(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = _make_two_pianos_one_bass_midi(root / 'source.mid')
            midi = mido.MidiFile(source)
            midi.tracks.append(_event_track((0, mido.MetaMessage('track_name', name='silent'))))
            midi.save(source)
            analysis = {'bar_boundaries_sec': [0, 1, 2], 'tracks': [
                {'source_track_index': 4, 'role': 'pads', 'role_source': 'structure.json'}]}
            for duration in (0, 3):
                config = {'render': {'sample_rate': 8000, 'duration_sec': duration, 'release_tail_sec': 0.5}}
                plan = prepare_stem_render(source, config=config, analysis=analysis)
                horizon = max(2, duration)
                self.assertEqual(plan.duration_sec, horizon)
                out = render_role_stems(source, output_dir=root / str(duration), plan=plan)
                index = json.loads(out['role_index'].read_text())
                waves = []
                for row in index['stems']:
                    data, sr = sf.read(out[row['id']])
                    self.assertEqual((len(data), sr), (int((horizon + 0.5) * 8000), 8000))
                    self.assertTrue(np.isfinite(data).all())
                    self.assertLessEqual(np.max(np.abs(data)), 0.95)
                    if row['note_count']:
                        self.assertGreater(np.sqrt(np.mean(data[100:3000] ** 2)), 1e-5)
                    else:
                        np.testing.assert_array_equal(data, np.zeros_like(data))
                    waves.append(data)
                summed = sum(waves)
                expected = summed * min(1.0, 0.95 / np.max(np.abs(summed)))
                mix, _ = sf.read(out['rough_mix'])
                np.testing.assert_allclose(mix, expected, atol=1e-7)
                self.assertGreater(np.sqrt(np.mean(mix[100:3000] ** 2)), 1e-5)
                self.assertLessEqual(np.max(np.abs(mix)), 0.95)
                click, _ = sf.read(out['click'])
                self.assertEqual(len(click), len(mix))
                self.assertGreater(np.max(np.abs(click)), 0)
                self.assertFalse(np.any(mix[8000:]))  # Click at t=1 is NOT in the mix.

    def test_one_gain_only_when_needed_with_stereo_padding_and_rate_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for level in (0.02, 0.8):
                sf.write(root / 'a.wav', np.full((100, 2), level), 8000, subtype='FLOAT')
                sf.write(root / 'b.wav', np.full(50, level), 8000, subtype='FLOAT')
                mix = rough_mix([root / 'a.wav', root / 'b.wav'], output_path=root / f'mix-{level}.wav')
                data, _ = sf.read(mix)
                expected = np.r_[np.full(50, 2 * level), np.full(50, level)]
                expected *= min(1, 0.95 / max(expected))
                np.testing.assert_allclose(data, expected, atol=1e-7)
                self.assertLessEqual(np.max(np.abs(data)), 0.95)
                first = mix.read_bytes()
                repeated = rough_mix([root / 'a.wav', root / 'b.wav'], output_path=root / f'repeat-{level}.wav')
                self.assertEqual(first, repeated.read_bytes())
                with self.assertRaises(FileExistsError):
                    rough_mix([root / 'a.wav', root / 'b.wav'], output_path=mix)
                self.assertEqual(first, mix.read_bytes())
            sf.write(root / 'b.wav', [0.1], 16000, subtype='FLOAT')
            with self.assertRaisesRegex(ValueError, 'sample rate'):
                rough_mix([root / 'a.wav', root / 'b.wav'], output_path=root / 'bad.wav')
            sf.write(root / 'b.wav', [np.nan], 8000, subtype='FLOAT')
            with self.assertRaisesRegex(ValueError, 'nonfinite'):
                rough_mix([root / 'a.wav', root / 'b.wav'], output_path=root / 'bad.wav')

    def test_contract_pads_short_output_limits_peak_and_hashes_final_wav(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = _make_two_pianos_one_bass_midi(root / 'source.mid')
            selected = registry.resolve_renderer('bass', config={})
            request = registry.RenderRequest(source, root / 'out.wav', {}, 'direct_12', 8000, 2, 0.5, None, {})

            def short(request):
                sf.write(request.output_path, np.array([[2.0, -1.0], [0.0, 0.5]]), 8000, subtype='FLOAT')
                return request.output_path

            selected = replace(selected, backend=replace(selected.backend, render=short))
            result = registry.render_request(request, selected)
            data, _ = sf.read(result.output_path)
            self.assertEqual(data.shape, (20000, 2))
            self.assertLessEqual(np.max(np.abs(data)), 0.95)
            np.testing.assert_allclose(data[:2], [[0.95, -0.475], [0, 0.2375]], atol=1e-7)
            self.assertFalse(np.any(data[2:]))
            self.assertEqual(result.output_sha256, registry.file_digest(result.output_path))
            self.assertEqual(json.loads(Path(str(result.output_path) + '.json').read_text())['frames'], 20000)
            with self.assertRaisesRegex(ValueError, 'horizon'):
                registry.render_request(replace(request, duration_sec=0, release_tail_sec=0), selected)

    def test_invalid_common_timing_and_peak_fail_before_render(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = _make_two_pianos_one_bass_midi(root / 'source.mid')
            for settings in ({'release_tail_sec': -1}, {'duration_sec': float('nan')},
                             {'sample_rate': 0}, {'mix_peak': 1.1}, {'mix_peak': 0}):
                with self.subTest(settings=settings), self.assertRaises(ValueError):
                    prepare_stem_render(source, config={'render': settings})

    def test_float_wav_bytes_ignore_wall_clock_peak_timestamp_not_pcm(self):
        from unittest.mock import patch
        from aimusic.audio.render.util import write_float_wav
        original_write = sf.write
        with tempfile.TemporaryDirectory() as tmp:
            hashes = []
            for timestamp in (12345678, 87654321):
                def stamped(path, data, rate, **kwargs):
                    original_write(path, data, rate, **kwargs)
                    payload = bytearray(Path(path).read_bytes())
                    offset = payload.index(b'PEAK')
                    payload[offset + 12:offset + 16] = timestamp.to_bytes(4, 'little')
                    Path(path).write_bytes(payload)
                path = Path(tmp) / f'{timestamp}.wav'
                pcm = np.array([[.25, -.5], [.125, .375]], dtype=np.float32)
                with patch.object(sf, 'write', side_effect=stamped):
                    write_float_wav(path, pcm, 48000)
                hashes.append(path.read_bytes())
                actual, rate = sf.read(path)
                np.testing.assert_array_equal(actual, pcm)
                self.assertEqual(rate, 48000)
            self.assertEqual(*hashes)

    def test_peak_normalization_walks_odd_chunks_without_touching_pcm_markers(self):
        from unittest.mock import patch
        from aimusic.audio.render.util import write_float_wav

        # Finite FLOAT samples whose bytes also impersonate a PEAK v1 chunk.
        pcm_bytes = b'PEAK' + struct.pack('<III', 16, 1, 0x12345678)
        pcm = np.frombuffer(pcm_bytes, dtype='<f4')
        fmt = b'fmt ' + struct.pack('<IHHIIHH', 16, 3, 1, 48000, 192000, 4, 32)
        fact = b'fact' + struct.pack('<II', 4, len(pcm))
        ancillary = b'JUNK' + struct.pack('<I', 3) + b'odd\x00'
        data = b'data' + struct.pack('<I', len(pcm_bytes)) + pcm_bytes
        peak = b'PEAK' + struct.pack('<IIIfI', 16, 1, 0x12345678, float(np.max(np.abs(pcm))), 0)
        with tempfile.TemporaryDirectory() as tmp:
            for peak_after_data in (False, True):
                with self.subTest(peak_after_data=peak_after_data):
                    # Exercise PCM markers on either side of the real chunk.
                    prefix = b'WAVE' + fmt + fact + ancillary + (data if peak_after_data else b'')
                    body = prefix + peak + (b'' if peak_after_data else data)
                    before = b'RIFF' + struct.pack('<I', len(body)) + body
                    timestamp_offset = 8 + len(prefix) + 12  # Known from construction, never byte-search.
                    decoded, rate = sf.read(io.BytesIO(before), dtype='float32')
                    np.testing.assert_array_equal(decoded, pcm)
                    self.assertEqual(rate, 48000)
                    path = Path(tmp) / 'odd-chunks.wav'
                    with patch.object(sf, 'write', side_effect=lambda output, *args, **kwargs: Path(output).write_bytes(before)):
                        write_float_wav(path, pcm, 48000)
                    after = path.read_bytes()
                    # Full byte equality protects chunk sizes, padding, peak values/positions and PCM.
                    expected = before[:timestamp_offset] + b'\x00' * 4 + before[timestamp_offset + 4:]
                    self.assertEqual(after, expected)
                    decoded, rate = sf.read(path, dtype='float32')
                    np.testing.assert_array_equal(decoded, pcm)
                    self.assertEqual(rate, 48000)
