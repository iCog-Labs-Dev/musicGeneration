from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Optional, Sequence, Tuple


RegisterRange = Tuple[int, int]
ValueRange = Tuple[float, float]


def _require_int(name: str, value: int, *, minimum: Optional[int] = None) -> None:
    """Validate that *value* is an integer and optionally above a floor."""
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an int.")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}.")


def _require_real(name: str, value: float, *, minimum: Optional[float] = None) -> None:
    """Validate that *value* is a finite real number."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{name} must be a real number.")
    if not isfinite(float(value)):
        raise ValueError(f"{name} must be finite.")
    if minimum is not None and float(value) < minimum:
        raise ValueError(f"{name} must be >= {minimum}.")


def _coerce_non_empty_str_tuple(name: str, values: Sequence[str]) -> Tuple[str, ...]:
    items = tuple(values)
    if not items:
        raise ValueError(f"{name} must not be empty.")
    if any(not isinstance(item, str) or not item.strip() for item in items):
        raise ValueError(f"{name} entries must be non-empty strings.")
    return items


def _coerce_positive_int_tuple(name: str, values: Sequence[int]) -> Tuple[int, ...]:
    items = tuple(values)
    if not items:
        raise ValueError(f"{name} must not be empty.")
    for item in items:
        _require_int(name, item, minimum=1)
    return items


def _coerce_register_range(name: str, values: Sequence[int]) -> RegisterRange:
    items = tuple(values)
    if len(items) != 2:
        raise ValueError(f"{name} must contain exactly two integer bounds.")
    low, high = items
    _require_int(f"{name}[0]", low)
    _require_int(f"{name}[1]", high)
    if low >= high:
        raise ValueError(f"{name} lower bound must be < upper bound.")
    return int(low), int(high)


def _coerce_unit_range(name: str, values: Sequence[float]) -> ValueRange:
    items = tuple(values)
    if len(items) != 2:
        raise ValueError(f"{name} must contain exactly two numeric bounds.")
    low, high = float(items[0]), float(items[1])
    _require_real(f"{name}[0]", low, minimum=0.0)
    _require_real(f"{name}[1]", high, minimum=0.0)
    if high > 1.0:
        raise ValueError(f"{name} upper bound must be <= 1.0.")
    if low > high:
        raise ValueError(f"{name} lower bound must be <= upper bound.")
    return low, high


class MicrotonalRendering(Enum):
    """How to render non-12-EDO pitches to MIDI."""

    MPE = "mpe"
    MTS = "mts"


class SBBackend(Enum):
    """Numerical backend used by the Schrödinger Bridge solver."""

    NUMPY = "numpy"
    JAX = "jax"


class PlanMethod(Enum):
    """High-level generation plan selection."""

    METHOD_A = "method_a"
    METHOD_B = "method_b"


class SectioningStrategy(Enum):
    """Whether planning runs as one pass or in explicit sections."""

    SINGLE_PASS = "single_pass"
    SECTION_WISE = "section_wise"


class PriorFactorization(Enum):
    """How a learned prior consumes structural tokens.

    `FACTORIZED` is the default contract for EPIC 3 because it keeps the interface stable
    across vocabulary growth and avoids an exploding whole-state token space.
    """

    WHOLE_STATE = "whole_state"
    FACTORIZED = "factorized"
    MIXED = "mixed"


class PlaceholderPriorMode(Enum):
    """How the placeholder neural prior behaves before a real model is integrated."""

    NEUTRAL = "neutral"
    STRUCTURED = "structured"


@dataclass(frozen=True)
class EDOConfig:
    """Configuration for an Equal Division of the Octave (EDO) system."""

    n: int
    base_tuning: float = 60.0
    microtonal_rendering_method: MicrotonalRendering = MicrotonalRendering.MPE
    pitch_bend_range: int = 48

    def __post_init__(self) -> None:
        _require_int("n", self.n, minimum=1)
        _require_real("base_tuning", self.base_tuning)
        _require_int("pitch_bend_range", self.pitch_bend_range, minimum=1)
        if not isinstance(self.microtonal_rendering_method, MicrotonalRendering):
            raise TypeError("microtonal_rendering_method must be a MicrotonalRendering value.")


@dataclass(frozen=True)
class StyleConfig:
    """Style-specific vocabulary and register constraints."""

    allowed_meters: Tuple[str, ...] = ("4/4", "5/4", "7/4")
    subdivision_patterns: Tuple[int, ...] = (3, 4)
    groove_families: Tuple[str, ...] = ("straight", "syncopated")
    chord_vocabulary_size: int = 48
    key_vocabulary_size: int = 12
    bass_register: RegisterRange = (28, 52)
    comping_register: RegisterRange = (45, 72)
    lead_register: RegisterRange = (60, 88)
    typical_density_range: ValueRange = (0.25, 0.85)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "allowed_meters", _coerce_non_empty_str_tuple("allowed_meters", self.allowed_meters)
        )
        object.__setattr__(
            self,
            "subdivision_patterns",
            _coerce_positive_int_tuple("subdivision_patterns", self.subdivision_patterns),
        )
        object.__setattr__(
            self, "groove_families", _coerce_non_empty_str_tuple("groove_families", self.groove_families)
        )
        _require_int("chord_vocabulary_size", self.chord_vocabulary_size, minimum=1)
        _require_int("key_vocabulary_size", self.key_vocabulary_size, minimum=1)
        object.__setattr__(
            self, "bass_register", _coerce_register_range("bass_register", self.bass_register)
        )
        object.__setattr__(
            self, "comping_register", _coerce_register_range("comping_register", self.comping_register)
        )
        object.__setattr__(
            self, "lead_register", _coerce_register_range("lead_register", self.lead_register)
        )
        object.__setattr__(
            self,
            "typical_density_range",
            _coerce_unit_range("typical_density_range", self.typical_density_range),
        )


@dataclass(frozen=True)
class PriorWeights:
    """Relative weighting for corpus priors and GTTM feature families."""

    lambda_data: float = 1.0
    lambda_gttm: float = 1.0
    meter: float = 1.0
    grouping: float = 1.0
    harmonic: float = 1.0
    prolongational_role: float = 1.0
    melodic_head: float = 1.0
    groove: float = 1.0

    def __post_init__(self) -> None:
        for name in (
            "lambda_data",
            "lambda_gttm",
            "meter",
            "grouping",
            "harmonic",
            "prolongational_role",
            "melodic_head",
            "groove",
        ):
            _require_real(name, getattr(self, name), minimum=0.0)
        if self.lambda_data == 0.0 and self.lambda_gttm == 0.0:
            raise ValueError("At least one of lambda_data or lambda_gttm must be > 0.")


@dataclass(frozen=True)
class NeuralPriorConfig:
    """Runtime configuration and artifact metadata for an external neural prior."""

    model_family: str = "external_neural_prior"
    model_version: str = "placeholder-v1"
    factorization_mode: PriorFactorization = PriorFactorization.FACTORIZED
    checkpoint_path: Optional[str] = None
    tokenizer_path: Optional[str] = None
    manifest_path: Optional[str] = None
    supports_batch_scoring: bool = True
    batch_size: int = 32
    placeholder_mode: PlaceholderPriorMode = PlaceholderPriorMode.STRUCTURED
    default_logp: float = 0.0

    def __post_init__(self) -> None:
        for name in ("model_family", "model_version"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string.")

        if not isinstance(self.factorization_mode, PriorFactorization):
            raise TypeError("factorization_mode must be a PriorFactorization value.")

        for name in ("checkpoint_path", "tokenizer_path", "manifest_path"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{name} must be None or a non-empty string.")

        if not isinstance(self.supports_batch_scoring, bool):
            raise TypeError("supports_batch_scoring must be a bool.")
        _require_int("batch_size", self.batch_size, minimum=1)

        if not isinstance(self.placeholder_mode, PlaceholderPriorMode):
            raise TypeError("placeholder_mode must be a PlaceholderPriorMode value.")
        _require_real("default_logp", self.default_logp)


@dataclass(frozen=True)
class SBConfig:
    """Numerical controls and sparsity limits for SB inference."""

    horizon_t: int = 64
    max_iterations: int = 200
    tolerance: float = 1e-6
    temperature: float = 1.0
    k_max: int = 64
    d_max: int = 8
    backend_selection: SBBackend = SBBackend.NUMPY

    def __post_init__(self) -> None:
        _require_int("horizon_t", self.horizon_t, minimum=1)
        _require_int("max_iterations", self.max_iterations, minimum=1)
        _require_real("tolerance", self.tolerance, minimum=0.0)
        if self.tolerance == 0.0:
            raise ValueError("tolerance must be > 0.")
        _require_real("temperature", self.temperature, minimum=0.0)
        if self.temperature == 0.0:
            raise ValueError("temperature must be > 0.")
        _require_int("k_max", self.k_max, minimum=1)
        _require_int("d_max", self.d_max, minimum=1)
        if not isinstance(self.backend_selection, SBBackend):
            raise TypeError("backend_selection must be an SBBackend value.")


@dataclass(frozen=True)
class DecodeConfig:
    """Decoder controls for sub-beat realization and expressive mapping."""

    subbeats_per_beat: int = 4
    drum_density: float = 0.75
    bass_density: float = 0.60
    comping_density: float = 0.55
    lead_density: float = 0.45
    bass_register: RegisterRange = (28, 52)
    comping_register: RegisterRange = (45, 72)
    lead_register: RegisterRange = (60, 88)
    min_comping_voices: int = 3
    max_comping_voices: int = 5
    max_lead_leap_steps: int = 7
    tension_velocity_range: ValueRange = (0.55, 1.0)
    tension_expression_range: ValueRange = (0.0, 1.0)

    def __post_init__(self) -> None:
        _require_int("subbeats_per_beat", self.subbeats_per_beat, minimum=1)
        for name in ("drum_density", "bass_density", "comping_density", "lead_density"):
            value = getattr(self, name)
            _require_real(name, value, minimum=0.0)
            if value > 1.0:
                raise ValueError(f"{name} must be <= 1.0.")
        object.__setattr__(
            self, "bass_register", _coerce_register_range("bass_register", self.bass_register)
        )
        object.__setattr__(
            self, "comping_register", _coerce_register_range("comping_register", self.comping_register)
        )
        object.__setattr__(
            self, "lead_register", _coerce_register_range("lead_register", self.lead_register)
        )
        _require_int("min_comping_voices", self.min_comping_voices, minimum=1)
        _require_int("max_comping_voices", self.max_comping_voices, minimum=1)
        if self.min_comping_voices > self.max_comping_voices:
            raise ValueError("min_comping_voices must be <= max_comping_voices.")
        _require_int("max_lead_leap_steps", self.max_lead_leap_steps, minimum=1)
        object.__setattr__(
            self,
            "tension_velocity_range",
            _coerce_unit_range("tension_velocity_range", self.tension_velocity_range),
        )
        object.__setattr__(
            self,
            "tension_expression_range",
            _coerce_unit_range("tension_expression_range", self.tension_expression_range),
        )


@dataclass(frozen=True)
class PlanConfig:
    """High-level generation-plan controls."""

    method: PlanMethod = PlanMethod.METHOD_A
    sectioning_strategy: SectioningStrategy = SectioningStrategy.SINGLE_PASS
    loop_midpoint: Optional[int] = None
    endpoint_top_k: int = 8
    endpoint_temperature: float = 1.0
    start_anchor_weight: float = 1.0
    end_anchor_weight: float = 1.0
    section_names: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.method, PlanMethod):
            raise TypeError("method must be a PlanMethod value.")
        if not isinstance(self.sectioning_strategy, SectioningStrategy):
            raise TypeError("sectioning_strategy must be a SectioningStrategy value.")
        if self.loop_midpoint is not None:
            _require_int("loop_midpoint", self.loop_midpoint, minimum=1)
        _require_int("endpoint_top_k", self.endpoint_top_k, minimum=1)
        _require_real("endpoint_temperature", self.endpoint_temperature, minimum=0.0)
        if self.endpoint_temperature == 0.0:
            raise ValueError("endpoint_temperature must be > 0.")
        _require_real("start_anchor_weight", self.start_anchor_weight, minimum=0.0)
        _require_real("end_anchor_weight", self.end_anchor_weight, minimum=0.0)
        if self.start_anchor_weight == 0.0 and self.end_anchor_weight == 0.0:
            raise ValueError("At least one endpoint anchor weight must be > 0.")

        section_names = tuple(self.section_names)
        if any(not isinstance(name, str) or not name.strip() for name in section_names):
            raise ValueError("section_names entries must be non-empty strings.")
        object.__setattr__(self, "section_names", section_names)

        if self.method is PlanMethod.METHOD_B and self.loop_midpoint is None:
            raise ValueError("loop_midpoint is required when method is METHOD_B.")
        if self.method is PlanMethod.METHOD_A and self.loop_midpoint is not None:
            raise ValueError("loop_midpoint is only valid when method is METHOD_B.")

        if self.sectioning_strategy is SectioningStrategy.SECTION_WISE and not self.section_names:
            raise ValueError("section_names are required for SECTION_WISE planning.")
