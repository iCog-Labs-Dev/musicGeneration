"""Candidate substitution retains the existing proposal call and result types."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from aimusic.core.component_types import CandidateGenerationResult
from aimusic.core.config import StyleConfig
from aimusic.core.core_types import BeatState
from aimusic.core.rng import RNGKey
from aimusic.core.vocab import Vocabularies

if TYPE_CHECKING:
    from aimusic.scoring.priors import Prior, PriorContext


class CandidateGenerator(Protocol):
    """Return legal successors and diagnostics, or an empty valid result."""

    def __call__(
        self,
        prev_state: BeatState,
        t: int,
        key: RNGKey,
        d_max: int,
        style_config: StyleConfig | None = None,
        vocabularies: Vocabularies | None = None,
        prior: Prior | None = None,
        context: PriorContext | None = None,
        edo: int | None = None,
        proposal_budget: int | None = None,
        prior_guided_proposals: bool = False,
    ) -> tuple[CandidateGenerationResult, RNGKey]: ...
