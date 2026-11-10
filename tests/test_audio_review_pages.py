"""Asset-free reviewer-page contracts; no Chrome, renderer or licensed assets."""
from __future__ import annotations

import base64
from dataclasses import replace
import hashlib
from html.parser import HTMLParser
import importlib
import io
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import tempfile
from threading import Thread
from urllib.error import HTTPError
from urllib.parse import quote, urljoin, urlsplit
from urllib.request import ProxyHandler, build_opener
import wave
import zipfile
import unittest
from unittest.mock import patch

from aimusic.audio.review_blinding import ReviewAssignment, create_assignment
from tests.test_audio_review_blinding import prepared_bundle


class PageParser(HTMLParser):
    def __init__(self, data):
        super().__init__()
        self.tags = []
        self.text = []
        self.styles = []
        self.in_style = False
        self.in_audio = False
        self.outside_audio_text = []
        self.feed(data.decode('utf-8'))
        self.close()

    def handle_starttag(self, tag, attrs):
        self.tags.append((tag, dict(attrs)))
        if tag == 'style':
            self.in_style = True
        if tag == 'audio':
            self.in_audio = True

    def handle_endtag(self, tag):
        if tag == 'style':
            self.in_style = False
        if tag == 'audio':
            self.in_audio = False

    def handle_data(self, data):
        (self.styles if self.in_style else self.text).append(data)
        if not self.in_style and not self.in_audio:
            self.outside_audio_text.append(data)

    def attrs(self, tag):
        return [a for t, a in self.tags if t == tag]


class TestReviewPages(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.bundle, self.blind = prepared_bundle(self.root)
        # A legal pseudonym can itself reveal a condition: do not echo it in HTML.
        self.assignment = create_assignment(self.bundle, 'groove-on', root=self.root,
                                            expected_sha256=self.bundle.sha256)

    def render(self, assignment=None, expected=None):
        try:
            module = importlib.import_module('aimusic.audio.review_pages')
        except ModuleNotFoundError as exc:
            if exc.name != 'aimusic.audio.review_pages':
                raise
            self.fail('Missing task 6.4 reviewer-page renderer')
        assignment = self.assignment if assignment is None else assignment
        return module.render_review_pages(assignment, expected_sha256=expected or self.assignment.sha256)

    def test_exact_six_pages_each_exposure_once_in_assignment_order(self):
        pages = self.render()
        expected_names = {'index.html', 'canonical.html', 'roles.html', 'parity.html', 'timing.html', 'tuning.html'}
        self.assertEqual(set(pages), expected_names)
        self.assertFalse(PageParser(pages['index.html']).attrs('audio'))
        seen = []
        for panel in self.assignment.panels:
            page = PageParser(pages[panel.purpose + '.html'])
            audio = page.attrs('audio')
            self.assertEqual([a['src'] for a in audio], [f'media/{s}.wav' for s in panel.stimuli])
            seen.extend(a['src'] for a in audio)
            self.assertEqual(len(page.attrs('h1')), 1)
            self.assertTrue(all(s in ''.join(page.text) for s in panel.stimuli))
            if len(panel.stimuli) == 2:
                self.assertIn('Clip A', page.text)
                self.assertIn('Clip B', page.text)
        self.assertEqual(len(seen), 14)
        self.assertEqual(len(set(seen)), 14)
        for filename, data in pages.items():
            self.assertIsInstance(data, bytes)
            page = PageParser(data)
            links = {a['href'] for a in page.attrs('a')}
            self.assertTrue(expected_names <= links, filename)
            self.assertIn(self.assignment.sha256, ''.join(page.text))
            self.assertIn(self.bundle.sha256, ''.join(page.text))

    def test_pure_deterministic_renderer_does_not_read_or_serialize_private_data(self):
        before = self.assignment.to_bytes()
        with patch.object(Path, 'open', side_effect=AssertionError('page renderer must not access files')):
            pages = self.render()
            self.assertEqual(pages, self.render())
        self.assertEqual(self.assignment.to_bytes(), before)
        public = '\n'.join(pages) + '\n' + b'\n'.join(pages.values()).decode()
        for forbidden in (*[c for _, c in self.blind.conditions], self.blind.order_seed, str(self.root),
                          'order_seed', 'conditions', 'blind_map', 'blind-map', 'private/', 'raw.wav',
                          'source_wav', 'render_sha256', 'presentation_sha256', 'backend', 'soundfont',
                          'native', 'simple', 'runtime', self.assignment.reviewer):
            self.assertNotIn(forbidden, public)

    def test_untrusted_identity_private_graph_and_forged_assignment_rejected(self):
        with self.assertRaisesRegex(ValueError, 'digest'):
            self.render(expected='0' * 64)
        for invalid in (self.bundle, self.assignment.to_dict(), None):
            with self.subTest(kind=type(invalid).__name__), self.assertRaises(ValueError):
                # Explicit call: None is invalid too, rather than the helper default.
                module = importlib.import_module('aimusic.audio.review_pages')
                module.render_review_pages(invalid, expected_sha256=self.assignment.sha256)
        for malicious in ('https://example.test/a.wav', '//example.test/a.wav', '../private/raw.wav',
                          'media/s00.wav?source=groove-on', 'media/s00.wav#condition',
                          'media/s00.wav" onerror="alert(1)', 'media/%2e%2e/raw.wav', 'media/groove-on.wav'):
            forged = ReviewAssignment.from_bytes(self.assignment.to_bytes())
            object.__setattr__(forged.stimuli[0].media, 'path', malicious)
            with self.subTest(path=malicious), self.assertRaises(ValueError):
                self.render(forged, forged.sha256)
        forged = ReviewAssignment.from_bytes(self.assignment.to_bytes())
        object.__setattr__(forged, 'reviewer', '<script>alert(1)</script>')
        with self.assertRaises(ValueError):
            self.render(forged, forged.sha256)

    def test_reversed_pairs_follow_assignment_not_sorted_inventory(self):
        changed = replace(self.assignment, panels=tuple(
            replace(p, stimuli=tuple(reversed(p.stimuli))) if len(p.stimuli) == 2 else p
            for p in self.assignment.panels))
        pages = self.render(changed, changed.sha256)
        for panel in changed.panels:
            self.assertEqual([a['src'] for a in PageParser(pages[panel.purpose + '.html']).attrs('audio')],
                             [f'media/{s}.wav' for s in panel.stimuli])

    def test_strict_csp_authorizes_only_exact_style_and_local_media(self):
        for filename, data in self.render().items():
            with self.subTest(page=filename):
                page = PageParser(data)
                policies = [a['content'] for a in page.attrs('meta')
                            if a.get('http-equiv') == 'Content-Security-Policy']
                self.assertEqual(len(policies), 1)
                policy = dict((parts[0], parts[1:]) for directive in policies[0].split(';')
                              if (parts := directive.strip().split()))
                for directive in ('default-src', 'script-src', 'connect-src', 'object-src',
                                  'base-uri', 'form-action', 'style-src-attr'):
                    self.assertEqual(policy[directive], ["'none'"])
                self.assertEqual(policy['media-src'], ["'self'"])
                self.assertEqual(len(page.styles), 1)
                style_hash = base64.b64encode(hashlib.sha256(page.styles[0].encode()).digest()).decode()
                self.assertEqual(policy['style-src'], [f"'sha256-{style_hash}'"])
                self.assertNotIn('unsafe-inline', policies[0])
                self.assertNotIn('unsafe-eval', policies[0])
                self.assertTrue(any(a.get('name') == 'referrer' and a['content'] == 'no-referrer'
                                    for a in page.attrs('meta')))
                html = data.decode()
                self.assertLess(html.index('Content-Security-Policy'), html.index('<style>'))
                for tag, attrs in page.tags:
                    self.assertNotIn(tag, ('script', 'iframe', 'object', 'embed', 'base', 'link', 'form', 'img'))
                    self.assertFalse(any(a.startswith('on') or a == 'style' for a in attrs))
                    for attribute in ('href', 'src'):
                        if attribute not in attrs:
                            continue
                        value = attrs[attribute]
                        self.assertFalse(urlsplit(value).scheme or urlsplit(value).netloc)
                        self.assertNotIn('..', value)
                        self.assertNotIn('\\', value)
                self.assertNotIn('url(', page.styles[0])
                self.assertNotIn('@import', page.styles[0])

    def test_keyboard_structure_accessible_names_and_low_volume_guidance(self):
        for filename, data in self.render().items():
            with self.subTest(page=filename):
                page = PageParser(data)
                self.assertEqual(page.attrs('html'), [{'lang': 'en'}])
                self.assertTrue(any(a.get('name') == 'viewport' for a in page.attrs('meta')))
                ids = [a['id'] for _, a in page.tags if 'id' in a]
                self.assertEqual(len(ids), len(set(ids)))
                self.assertEqual(page.attrs('a')[0]['href'], '#main')
                self.assertEqual(page.attrs('main'), [{'id': 'main', 'tabindex': '-1'}])
                self.assertTrue(all('aria-label' in a for a in page.attrs('nav')))
                current = [a for a in page.attrs('a') if a.get('aria-current') == 'page']
                self.assertEqual([a['href'] for a in current], [filename])
                for _, attrs in page.tags:
                    for attribute in ('aria-labelledby', 'aria-describedby'):
                        if attribute in attrs:
                            self.assertTrue(set(attrs[attribute].split()) <= set(ids))
                    if 'tabindex' in attrs:
                        self.assertIn(attrs['tabindex'], ('0', '-1'))
                    self.assertNotIn('accesskey', attrs)  # No browser/assistive-key conflicts.
                for attrs in page.attrs('audio'):
                    self.assertIn('controls', attrs)
                    self.assertEqual(attrs['preload'], 'none')
                    self.assertEqual(attrs['tabindex'], '0')
                    self.assertIn('aria-labelledby', attrs)
                    self.assertIn('low-volume', attrs['aria-describedby'].split())
                    for forbidden in ('autoplay', 'loop', 'muted', 'hidden'):
                        self.assertNotIn(forbidden, attrs)
                text = ''.join(page.text)
                for instruction in ('low volume', 'Tab', 'Shift+Tab', 'Enter', 'Pause', 'breaks'):
                    self.assertIn(instruction, text)
                style = ''.join(page.styles)
                for rule in (':focus-visible', 'outline:', 'overflow-wrap: anywhere', 'width: 100%'):
                    self.assertIn(rule, style)

    def test_fallback_visible_outside_audio_and_no_implied_review_completion(self):
        for filename, data in self.render().items():
            page = PageParser(data)
            text = ''.join(page.outside_audio_text)
            self.assertIn('not mastered', text)
            self.assertIn('no human approval', text)
            self.assertIn('not saved', text)
            downloads = [a for a in page.attrs('a') if 'download' in a]
            self.assertEqual([a['href'] for a in downloads], [a['src'] for a in page.attrs('audio')])
            for audio, download in zip(page.attrs('audio'), downloads):
                self.assertEqual(download['download'], audio['src'].split('/')[-1])
                self.assertTrue(any(id_.startswith('fallback-') for id_ in audio['aria-describedby'].split()))
            if filename != 'index.html':
                for instruction in ('If playback fails or media is missing', 'local audio player',
                                    're-extract', 'not reviewed', 'Do not substitute'):
                    self.assertIn(instruction, text)
                self.assertNotIn('hidden', [key for _, attrs in page.tags for key in attrs])
            else:
                for instruction in ('python -m http.server', '--bind 127.0.0.1', 'http://127.0.0.1:8000/index.html'):
                    self.assertIn(instruction, text)

    def test_extracted_public_tree_serves_every_link_and_decodable_wav_and_survives_missing_media(self):
        # Transport/decoder smoke, not proof of a particular browser's audio UI.
        # Create only owned synthetic bytes: no publication/ZIP-extraction API is under test here.
        data = io.BytesIO()
        with wave.open(data, 'wb') as wav:
            wav.setparams((1, 2, 8000, 0, 'NONE', 'not compressed'))
            wav.writeframes(b'\x00\x01' * 80)
        payload = data.getvalue()
        media = tuple(replace(s, media=replace(s.media, sha256=hashlib.sha256(payload).hexdigest(), size=len(payload)))
                      for s in self.assignment.stimuli)
        assignment = replace(self.assignment, stimuli=media)
        pages = self.render(assignment, assignment.sha256)
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, 'w') as zipped:
            for filename, content in pages.items():
                zipped.writestr(filename, content)
            for stimulus in assignment.stimuli:
                zipped.writestr(stimulus.media.path, payload)
        extracted = self.root / 'public only écoute' / 'review'
        with zipfile.ZipFile(archive) as zipped:
            zipped.extractall(extracted)
        self.assertFalse((extracted / 'private').exists())
        assignment.verify(extracted, expected_sha256=assignment.sha256)

        class QuietHandler(SimpleHTTPRequestHandler):
            def log_message(self, format, *args):
                pass

        server = ThreadingHTTPServer(('127.0.0.1', 0), partial(QuietHandler, directory=str(extracted.parent)))
        thread = Thread(target=server.serve_forever, kwargs={'poll_interval': .01}, daemon=True)
        thread.start()
        opener = build_opener(ProxyHandler({}))  # Never send loopback requests through ambient proxies.
        base = f'http://127.0.0.1:{server.server_port}/review/'
        try:
            for filename, content in pages.items():
                with opener.open(base + filename, timeout=3) as response:
                    self.assertEqual(response.headers.get_content_type(), 'text/html')
                    self.assertEqual(response.read(), content)
                page = PageParser(content)
                for tag, attrs in page.tags:
                    for attribute in ('href', 'src'):
                        if attribute not in attrs:
                            continue
                        url = urljoin(base + filename, quote(attrs[attribute], safe='/#'))
                        self.assertEqual(urlsplit(url).netloc, urlsplit(base).netloc)
                        with opener.open(url, timeout=3) as response:
                            body = response.read()
                            if tag == 'audio' or 'download' in attrs:
                                self.assertEqual(body, payload)
                                self.assertIn(response.headers.get_content_type(), ('audio/wav', 'audio/x-wav', 'audio/vnd.wave'))
                                with wave.open(io.BytesIO(body)) as wav:
                                    self.assertEqual((wav.getnchannels(), wav.getnframes(), wav.getframerate()), (1, 80, 8000))
            missing = assignment.stimuli[0].media.path
            (extracted / missing).unlink()
            with self.assertRaises(HTTPError) as failure:
                opener.open(base + missing, timeout=3)
            self.assertEqual(failure.exception.code, 404)
            failure.exception.close()
            panel = next(p for p in assignment.panels if assignment.stimuli[0].id in p.stimuli)
            with opener.open(base + panel.purpose + '.html', timeout=3) as response:
                page = PageParser(response.read())
                self.assertIn('If playback fails or media is missing', ''.join(page.outside_audio_text))
                self.assertIn('not reviewed', ''.join(page.outside_audio_text))
        finally:
            server.shutdown()
            thread.join(timeout=3)
            server.server_close()
