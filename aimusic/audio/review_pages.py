"""Pure, reviewer-safe HTML generation; no media reads or bundle publication."""
from __future__ import annotations

import base64
import hashlib
from html import escape

from aimusic.audio.review_blinding import AssignedStimulus, ReviewAssignment

_TITLES = {'index': 'Listening review', 'canonical': 'Canonical listening', 'roles': 'Role stems',
           'parity': 'Parity comparison', 'timing': 'Timing comparison', 'tuning': 'Tuning comparison'}
_PROMPTS = {
    'canonical': 'Listen for overall audibility, timing, lost voices, clicks, distortion and cut-off tails.',
    'roles': 'Listen to each isolated part for audibility, missing notes, stuck notes and cut-off tails.',
    'parity': 'Compare A and B for audible differences. Report defects separately from preference.',
    'timing': 'Compare A and B for timing differences. A difference need not be an improvement.',
    'tuning': 'Compare pitch relationships and perceived differences without guessing clip identities.',
}
_STYLE = '''
:root { color-scheme: light; --ink: #18212b; --paper: #fff; --rule: #bbc3ca; --link: #005a9c; }
body { font: 1rem/1.5 system-ui, sans-serif; color: var(--ink); background: var(--paper); margin: 0; }
main, header { max-width: 52rem; margin: auto; padding: 1rem; overflow-wrap: anywhere; }
h1 { font-size: 1.75rem; } h2 { font-size: 1.25rem; }
section { border-top: 1px solid var(--rule); padding: 1rem 0; }
a { color: var(--link); text-decoration: underline; }
nav ul { display: flex; flex-wrap: wrap; gap: .5rem 1rem; list-style: none; padding: 0; }
nav a, .skip { display: inline-block; padding: .5rem .25rem; }
[aria-current="page"] { font-weight: bold; }
audio { display: block; width: 100%; max-width: 100%; }
.digest { font-size: .875rem; overflow-wrap: anywhere; }
:focus-visible { outline: 3px solid var(--link); outline-offset: 3px; }
'''
_STYLE_HASH = base64.b64encode(hashlib.sha256(_STYLE.encode('utf-8')).digest()).decode('ascii')
_CSP = ("default-src 'none'; script-src 'none'; connect-src 'none'; object-src 'none'; "
        f"style-src 'sha256-{_STYLE_HASH}'; style-src-attr 'none'; media-src 'self'; "
        "base-uri 'none'; form-action 'none'")
_GUIDANCE = '''
<p id="low-volume">Set your device to low volume before playing. Stop if listening is uncomfortable.
Pause one clip before starting another; keep volume unchanged within a pair. Take breaks between panels.</p>
<p id="keyboard">Use Tab and Shift+Tab to move between links and audio controls; Enter opens a link.
Use the focused player's play/pause, seek and volume controls (Space or Enter where supported).
No clip starts automatically. Browser player shortcuts vary; WAV downloads are also keyboard accessible.</p>
<p>These are rough listening examples, not mastered audio. This page grants no human approval.
Listening does not establish objective scores or polyphonic pitch accuracy.
Responses are not saved here; record findings separately using the assignment and clip IDs.</p>
'''
_INDEX = '''
<p>14 clips in five panels: four canonical clips, four role stems, and three two-clip comparisons.
Follow the panel navigation in order. A/B labels apply only within their panel.</p>
<h2>Open locally</h2>
<p>Extract the whole reviewer bundle first. Keep these pages beside the media/ folder.
No internet connection, credentials or repository checkout is needed.
Serve only the extracted reviewer folder, never a directory containing unrelated files.</p>
<p>In that folder, run <code>python -m http.server 8000 --bind 127.0.0.1</code> and open
<code>http://127.0.0.1:8000/index.html</code> in your browser. Stop the server with Ctrl+C when finished.
Direct file opening is browser-dependent; local serving is the supported method.</p>
<p>If a player or download fails, follow the recovery instructions beside that clip.
Do not treat missing or unplayable media as silence or a completed review.</p>
'''


def _clip(stimulus: AssignedStimulus, label: str) -> str:
    # Values are schema-validated, but still encode every dynamic HTML context.
    sid, path = escape(stimulus.id, quote=True), escape(stimulus.media.path, quote=True)
    label = escape(label)
    return f'''<section aria-labelledby="label-{sid}">
<h2 id="label-{sid}">{label}</h2>
<p id="id-{sid}" class="digest">Clip ID: <code>{sid}</code></p>
<audio controls preload="none" tabindex="0" src="{path}"
 aria-labelledby="label-{sid} id-{sid}" aria-describedby="low-volume keyboard fallback-{sid}">
Your browser does not support embedded audio. Use the WAV download below.</audio>
<p id="fallback-{sid}">If playback fails or media is missing,
<a href="{path}" download="{sid}.wav">Download WAV for {label}</a> and try a local audio player at low volume.
If the download also fails, re-extract the full bundle, keep media/ beside these pages, and reload.
If it still fails, stop this panel and request a verified copy from the organizer.
Record the clip as not reviewed, not silence or an audio defect. Do not substitute another file.</p>
<p class="digest">WAV SHA-256: <code>{escape(stimulus.media.sha256)}</code></p>
</section>'''


def render_review_pages(assignment: ReviewAssignment, *, expected_sha256: str) -> dict[str, bytes]:
    """Return fixed relative HTML filenames and UTF-8 bytes for one assignment.

    Pass an assignment made via create_assignment (which uses the admitted public
    projection), with its independently trusted digest. This function validates
    metadata, not admission or media bytes: the assembler must verify those before
    publication. It never accepts the private graph, writes files or certifies a
    complete bundle. Place the six pages beside media/, not beside private files.

    Scripts are forbidden. Recovery text is always visible outside the audio tag:
    that tag's fallback is hidden by browsers that support audio but cannot play a
    particular file. Playback and file integrity are not automatically diagnosed.
    """
    if type(assignment) is not ReviewAssignment:
        raise ValueError('reviewer-safe assignment required')
    assignment.__post_init__()
    if assignment.sha256 != expected_sha256:
        raise ValueError('assignment digest mismatch')
    media = {s.id: s for s in assignment.stimuli}
    bodies = {'index': _INDEX}
    for panel in assignment.panels:
        clips = [_clip(media[s], f'Clip {"AB"[i] if len(panel.stimuli) == 2 else i + 1}')
                 for i, s in enumerate(panel.stimuli)]
        bodies[panel.purpose] = f'<p>{escape(_PROMPTS[panel.purpose])}</p>\n' + '\n'.join(clips)
    pages = {}
    for name, body in bodies.items():
        navigation = '\n'.join(
            f'<li><a href="{key}.html"' + (' aria-current="page"' if key == name else '') + f'>{escape(title)}</a></li>'
            for key, title in _TITLES.items())
        title = escape(_TITLES[name])
        pages[name + '.html'] = f'''<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta http-equiv="Content-Security-Policy" content="{escape(_CSP, quote=True)}">
<meta name="referrer" content="no-referrer">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>{_STYLE}</style>
</head><body><header>
<a class="skip" href="#main">Skip to listening content</a>
<nav aria-label="Listening panels"><ul>{navigation}</ul></nav>
</header><main id="main" tabindex="-1"><h1>{title}</h1>
{_GUIDANCE}{body}
<p class="digest">Bundle SHA-256: <code>{escape(assignment.bundle_sha256)}</code></p>
<p class="digest">Assignment SHA-256: <code>{escape(assignment.sha256)}</code></p>
</main></body></html>
'''.encode('utf-8')
    return pages
