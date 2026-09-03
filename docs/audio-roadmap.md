# Audio pipeline roadmap (M0–M6)

Derived from `AUDIO_PIPELINE_ARCHITECTURE.pdf` §7 (13-week quarter plan).
Executed as fork PRs into `arsenylosev/musicGeneration:main` first; org PR to
iCog is a separate human step after fork CI is green.

## Sequencing rule

**Do not begin M3 before M2 is merged.** A restyle layer without a scorer produces
unaccountable output and wastes paid calls.

## Milestones

| Milestone | Deliverables | Exit criterion | Fork PR phase |
|-----------|--------------|----------------|---------------|
| **M0 — Contract** | `render_package`, structure schema v1, DECISIONS, extras packaging, architecture tests, fixture path | `generate` emits valid RenderPackage; dependency-rule tests pass; core install has no torch | **PR1 (this)** |
| **M1 — Deterministic spine** | `from_score`/`from_midi`/`reconcile`; groove spec+apply; fluidsynth/simple render; orchestrator+cache; CLI `expressivize` | Fixture MIDI → grooving stems; property tests on offsets; reconcile within tolerance | PR2–PR3 |
| **M1.5 — Microtonal spine (B3)** | `theory.edo` scala/MTS helpers; `render/tuning.py`; tuning_check | 19-EDO stem f0s within ~5¢ of lattice | PR4 |
| **M2 — Scoring** | rhythm/notes/CLAP/FAD/combine; corruption suite | Metrics degrade monotonically under known corruptions | PR4 |
| **M3 — One endpoint E2E** | RestyleEndpoint + Stable Audio; chunking on `boundary_lvl`; prompts; search+budget; vcrpy | Best candidate per stem without human input, inside budget, from fixtures | PR5 |
| **M4 — References** | groove extract; stylecard; FAD corpus; mixmaster+Matchering; HTML audit | Unattended master + audit from refs + RenderPackage | Later |
| **M5 — Breadth** | Suno cover blend; MusicGen search branch; GrooVAE drums (flagged); inpaint | Cover blend improves objective on ≥1 fixture without breaching floors | Later |
| **M6a — Single-ref seams** | host analysis; lattice re-quant; single_ref/mixfit scores; profile | Self-consistency fixture; lattice re-quant near no-op on aligned input | Later |
| **Hardening** | Docs consolidation; CI budget; next-quarter B4/closed-loop scope | Fluidsynth-only E2E fixture under ~60s in CI | Later |

## Deferred to next quarter

- **B4** host-conditioned planning (`candidates` gate + `E_host`)
- `hostsampler.py` auto-SFZ
- Closed-loop groove / GTTM-weight optimization
- MIDI-DDSP lead path beyond prototype

## Parallel symbolic track (historical)

The PDF listed Method A / decoder / MPE as parallel blockers. On current fork
`main` those symbolic pieces largely exist; keep developing audio against
fixtures so halves stay decoupled.

## Related

- [audio-next-steps.md](audio-next-steps.md)
- [audio-stages.md](audio-stages.md)
- [audio-pipeline.md](audio-pipeline.md)
