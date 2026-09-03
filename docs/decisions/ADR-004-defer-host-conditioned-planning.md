# ADR-004: Defer host-conditioned planning (B4)

## Status
Accepted (implementation deferred)

## Date
2026-08-20

## Context
Host-compatible rhythm can be repaired post-hoc (lattice re-quant) or enforced
at plan time (`E_host` in SB search). Plan-time is stronger but couples audio
host analysis to symbolic planning and heavy deps (madmom/demucs).

## Decision
Cut file seams (`host_lattice.json`, lattice helpers) now. Do **not** implement
planning `E_host` / candidate gates in the current quarter. When done later,
`E_host` joins the weighted energy sum (not a post-hoc edge mask only).

## Alternatives considered
- Implement B4 immediately with the audio port — rejected (blast radius)
- Only ever post-hoc re-quant — acceptable fallback, not the long-term design

## Consequences
Agents must not start B4 unless explicitly tasked. Record details in
DECISIONS.md when work begins.
