"""REQ-13 benchmark: quality/runtime tradeoffs for candidate proposal budgets.

Measures, for a range of ``proposal_budget`` values, both:
  - runtime per call to ``get_valid_next_states`` (with prior-guided ranking
    enabled, so scoring cost is included), and
  - quality, as top-k recall of the bounded search against an exhaustive
    (very large budget) search on the same source state, averaged over
    several candidate-proposal RNG seeds.

Run with:
    python scripts/benchmark_proposal_budget.py

Results are also written up in docs/decisions/ADR-005-candidate-proposal-budget.md.
"""
from __future__ import annotations

import statistics
import time

from aimusic.core.config import StyleConfig
from aimusic.core.core_types import BeatState
from aimusic.core.rng import RNGKey
from aimusic.core.vocab import DEFAULT_VOCABULARIES
from aimusic.planning.candidates import get_valid_next_states
from aimusic.scoring.priors import NeuralPrior

VOCABS = DEFAULT_VOCABULARIES
STYLE = StyleConfig(
    allowed_meters=("4/4", "5/4", "7/4"),
    groove_families=("straight", "syncopated", "swing"),
)
PRIOR = NeuralPrior()

PREV_STATE = BeatState(
    meter_id=VOCABS.meters.token_for_label("4/4").id,
    beat_in_bar=3,
    boundary_lvl=VOCABS.boundaries.token_for_label("none").id,
    key_id=VOCABS.keys.token_for_label("C").id,
    chord_id=VOCABS.chords.token_for_label("G7").id,
    role_id=VOCABS.roles.token_for_label("prep").id,
    head_id=VOCABS.heads.token_for_label("upper_approach").id,
    groove_id=VOCABS.grooves.token_for_label("straight_8ths").id,
)

BUDGETS = (32, 64, 128, 256, 512, 1024, 2048)
SEEDS = range(10)
TOP_K = 8
GROUND_TRUTH_BUDGET = 8192


def _top_k(*, proposal_budget: int, seed: int) -> tuple[set[BeatState], float]:
    start = time.perf_counter()
    result, _ = get_valid_next_states(
        PREV_STATE, 4, RNGKey(seed=seed), 8,
        style_config=STYLE, vocabularies=VOCABS, prior=PRIOR,
        proposal_budget=proposal_budget, prior_guided_proposals=True,
    )
    elapsed = time.perf_counter() - start
    return set(result.states[:TOP_K]), elapsed


def main() -> None:
    ground_truth, _ = _top_k(proposal_budget=GROUND_TRUTH_BUDGET, seed=SEEDS[0])

    print(f"{'budget':>8}  {'avg_recall':>10}  {'avg_ms':>8}  {'p95_ms':>8}")
    for budget in BUDGETS:
        recalls = []
        timings = []
        for seed in SEEDS:
            top_k, elapsed = _top_k(proposal_budget=budget, seed=seed)
            recalls.append(len(top_k & ground_truth) / TOP_K)
            timings.append(elapsed * 1000.0)
        avg_recall = statistics.mean(recalls)
        avg_ms = statistics.mean(timings)
        p95_ms = sorted(timings)[int(0.95 * (len(timings) - 1))]
        print(f"{budget:>8}  {avg_recall:>10.3f}  {avg_ms:>8.2f}  {p95_ms:>8.2f}")


if __name__ == "__main__":
    main()
