# Proposal

## Why

The audible-demo branch cannot publish groove bundles or rendered stems reliably on native Windows, despite documenting Windows setup, because its publication primitives depend on POSIX descriptor APIs or conflict with Windows directory-sharing rules. The existing four-mix review also proves only a narrow milestone and does not provide a broad, blinded, hash-linked evaluation set for repeatable human testing.

## What Changes

- Add native Windows-safe publication for exclusive stem artifacts, demo sources, cache restoration, and transactional groove bundles while preserving containment, collision refusal, rollback, and recovery guarantees.
- Audit adjacent Windows path boundaries for reparse points, no-follow behavior, portable names, case-insensitive collisions, Unicode/long paths, and unsupported filesystems; fail closed rather than silently weakening publication.
- Add native Windows asset-free CI and an opt-in pinned FluidSynth verification lane distinct from API simulation and symlink-privilege-dependent tests.
- Add a bounded review-bundle workflow that generates admitted audio stimuli, objective-gate evidence, immutable provenance, blinded comparison assignments, reviewer templates, and digest-linked human findings, delivered as one deterministic sealed archive with strict verification before extraction.
- Expand the listening set beyond the historical four mixes with role stems, generated-package/import parity, meaningful timing comparisons, a valid-versus-flattened tuning control, and an explicitly limited polyphonic 19-EDO example.
- Preserve rough-demo limits: no mastering claim, no inherited human approval, no fabricated CLAP/FAD/LUFS/rhythm/note/polyphonic-f0 result, and no committed SoundFont or generated WAV assets.
- Use focused Conventional Commit checkpoints during implementation and retain the fork-first, separately authorized PR boundary.

## Capabilities

### New Capabilities

- `audio-output-publication`: Cross-platform publication of complete audio artifacts with platform-appropriate containment, collision, rollback, recovery, and verification behavior.
- `audio-human-review`: Reproducible generation, admission, blinding, sealed-archive delivery, and reporting of broad human-listening evidence linked to exact audio and provenance.

### Modified Capabilities

None. The repository has no synchronized main OpenSpec capabilities; the completed audible-demo change remains historical delta context rather than a main spec to modify.

## Impact

- Primary code areas: `aimusic/audio/groove/publication.py`, `aimusic/audio/render/stem_publication.py`, adjacent cache/source/listening safety helpers, orchestration, demo generation, and review reporting.
- Tests: native Windows publication and reparse/race coverage, platform-neutral safety tests, review-bundle schema and tamper tests, installed-wheel checks, and opt-in native FluidSynth evidence.
- Automation and documentation: Windows CI, PowerShell reproduction, supported-filesystem and threat-model notes, sample-matrix instructions, and a final evidence report format.
- Core symbolic dependencies remain unchanged; native APIs and DSP remain quarantined under `aimusic.audio`. No paid services, model weights, SoundFonts, generated WAVs, credentials, MUS-36 scorers, MUS-37 restyle, MUS-38 mastering, or organization PR are included.
