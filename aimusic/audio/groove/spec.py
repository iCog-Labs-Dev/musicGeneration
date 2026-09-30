"""Groove specification dataclasses + JSON schema (PDF §3.1)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class SwingSpec(BaseModel):
    ratio: float = Field(ge=0.5, le=0.75, description="0.5=straight, ~0.66=triplet swing")
    subdivision: int = Field(default=8, description="Subdivision of the beat pair")
    applies_to: list[str] = Field(default_factory=lambda: ["drums", "bass", "comp"])


class MicrotimingRole(BaseModel):
    by_position_16: list[float] | None = None
    global_offset: float | None = None
    jitter_sd: float | None = Field(default=None, le=4.0)
    phrase_rubato: bool | None = None


class MicrotimingSpec(BaseModel):
    drums: MicrotimingRole | None = None
    bass: MicrotimingRole | None = None
    lead: MicrotimingRole | None = None
    comp: MicrotimingRole | None = None


class GhostNotesSpec(BaseModel):
    prob: float = Field(ge=0.0, le=1.0)
    vel_scale: float = Field(ge=0.0, le=1.0)


class PhraseArcSpec(BaseModel):
    shape: str
    depth: float = Field(ge=0.0, le=1.0)


class VelocitySpec(BaseModel):
    accent_map_16: list[float] | None = None
    ghost_notes: dict[str, GhostNotesSpec] | None = None
    phrase_arc: PhraseArcSpec | None = None


class AnticipationRole(BaseModel):
    prob: float = Field(ge=0.0, le=1.0)
    push_ms: list[float]
    at: Literal["chord_changes"] | str = "chord_changes"


class AnticipationSpec(BaseModel):
    comp: AnticipationRole | None = None
    bass: AnticipationRole | None = None


class TempoCurveSpec(BaseModel):
    model_config = {"extra": "allow"}


class ArticulationRole(BaseModel):
    duration_scale: float | None = None
    strum_ms: float | None = None


class ArticulationSpec(BaseModel):
    bass: ArticulationRole | None = None
    comp: ArticulationRole | None = None
    lead: ArticulationRole | None = None
    drums: ArticulationRole | None = None


class GrooveSpec(BaseModel):
    """Parametric groove transform — never raw notes."""

    swing: SwingSpec | None = None
    microtiming_ms: MicrotimingSpec | None = None
    velocity: VelocitySpec | None = None
    anticipation: AnticipationSpec | None = None
    tempo_curve: TempoCurveSpec | dict[str, Any] | None = None
    articulation: ArticulationSpec | None = None


def groove_spec_json_schema() -> dict[str, Any]:
    """Export JSON Schema for LLM constrained decoding / validation."""
    return GrooveSpec.model_json_schema()


def parse_groove_spec(data: dict[str, Any] | GrooveSpec) -> GrooveSpec:
    """Validate and return a GrooveSpec; raises pydantic.ValidationError."""
    if isinstance(data, GrooveSpec):
        return data
    return GrooveSpec.model_validate(data)
