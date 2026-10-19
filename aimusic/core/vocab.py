from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Generic, Iterator, Mapping, Protocol, Sequence, Tuple, TypeVar

from aimusic.core.config import StyleConfig


PITCH_CLASS_NAMES_12: Tuple[str, ...] = (
    "C",
    "C#",
    "D",
    "Eb",
    "E",
    "F",
    "F#",
    "G",
    "Ab",
    "A",
    "Bb",
    "B",
)

DEFAULT_METER_SIGNATURES: Tuple[str, ...] = ("4/4", "3/4", "5/4", "7/4")
DEFAULT_GROOVE_FAMILIES: Tuple[str, ...] = ("straight", "syncopated", "swing")
CORE_CHORD_QUALITIES: Tuple[str, ...] = ("maj", "min", "7", "dim")


class _TokenLike(Protocol):
    id: int
    label: str


TokenT = TypeVar("TokenT", bound=_TokenLike)


def _require_non_negative_int(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an int.")
    if value < 0:
        raise ValueError(f"{name} must be >= 0.")


def _require_non_empty_str(name: str, value: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    if not value.strip():
        raise ValueError(f"{name} must not be empty.")


def _coerce_int_tuple(name: str, values: Sequence[int]) -> Tuple[int, ...]:
    items = tuple(values)
    for idx, value in enumerate(items):
        _require_non_negative_int(f"{name}[{idx}]", value)
    if len(items) != len(set(items)):
        raise ValueError(f"{name} must not contain duplicate values.")
    return items


def _parse_meter_signature(signature: str) -> Tuple[int, int]:
    _require_non_empty_str("meter signature", signature)
    try:
        numerator_str, denominator_str = signature.split("/", maxsplit=1)
        numerator = int(numerator_str)
        denominator = int(denominator_str)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid meter signature: {signature!r}") from exc

    if numerator <= 0 or denominator <= 0:
        raise ValueError(f"Invalid meter signature: {signature!r}")
    return numerator, denominator


def _strong_beats_for_meter(beats_per_bar: int) -> Tuple[int, ...]:
    if beats_per_bar == 4:
        return (0, 2)
    if beats_per_bar == 5:
        return (0, 3)
    if beats_per_bar == 7:
        return (0, 4)
    return (0,)


def _pitch_class_labels(cardinality: int) -> Tuple[str, ...]:
    if cardinality == 12:
        return PITCH_CLASS_NAMES_12
    return tuple(f"pc_{idx}" for idx in range(cardinality))


@dataclass(frozen=True)
class MeterToken:
    id: int
    label: str
    beats_per_bar: int
    strong_beats: Tuple[int, ...] = ()

    def __post_init__(self) -> None:
        _require_non_negative_int("id", self.id)
        _require_non_empty_str("label", self.label)
        _require_non_negative_int("beats_per_bar", self.beats_per_bar)
        if self.beats_per_bar == 0:
            raise ValueError("beats_per_bar must be > 0.")
        strong_beats = _coerce_int_tuple("strong_beats", self.strong_beats)
        if any(beat >= self.beats_per_bar for beat in strong_beats):
            raise ValueError("strong_beats must be valid beat indices within the bar.")
        object.__setattr__(self, "strong_beats", strong_beats)


@dataclass(frozen=True)
class BeatPositionToken:
    id: int
    label: str
    index: int

    def __post_init__(self) -> None:
        _require_non_negative_int("id", self.id)
        _require_non_empty_str("label", self.label)
        _require_non_negative_int("index", self.index)


@dataclass(frozen=True)
class BoundaryToken:
    id: int
    label: str
    level: int

    def __post_init__(self) -> None:
        _require_non_negative_int("id", self.id)
        _require_non_empty_str("label", self.label)
        _require_non_negative_int("level", self.level)


@dataclass(frozen=True)
class KeyToken:
    id: int
    label: str
    root_pc: int

    def __post_init__(self) -> None:
        _require_non_negative_int("id", self.id)
        _require_non_empty_str("label", self.label)
        _require_non_negative_int("root_pc", self.root_pc)


@dataclass(frozen=True)
class ChordToken:
    id: int
    label: str
    root_pc: int
    quality: str

    def __post_init__(self) -> None:
        _require_non_negative_int("id", self.id)
        _require_non_empty_str("label", self.label)
        _require_non_negative_int("root_pc", self.root_pc)
        _require_non_empty_str("quality", self.quality)


@dataclass(frozen=True)
class RoleToken:
    id: int
    label: str
    description: str

    def __post_init__(self) -> None:
        _require_non_negative_int("id", self.id)
        _require_non_empty_str("label", self.label)
        _require_non_empty_str("description", self.description)


@dataclass(frozen=True)
class HeadToken:
    id: int
    label: str
    description: str

    def __post_init__(self) -> None:
        _require_non_negative_int("id", self.id)
        _require_non_empty_str("label", self.label)
        _require_non_empty_str("description", self.description)


@dataclass(frozen=True)
class GrooveToken:
    id: int
    label: str
    family: str
    subdivision: int

    def __post_init__(self) -> None:
        _require_non_negative_int("id", self.id)
        _require_non_empty_str("label", self.label)
        _require_non_empty_str("family", self.family)
        _require_non_negative_int("subdivision", self.subdivision)
        if self.subdivision == 0:
            raise ValueError("subdivision must be > 0.")


@dataclass(frozen=True)
class TokenVocabulary(Generic[TokenT]):
    """Immutable vocabulary of typed token objects."""

    name: str
    tokens: Tuple[TokenT, ...]
    _id_map: Mapping[int, TokenT] = field(init=False, repr=False, compare=False)
    _label_map: Mapping[str, TokenT] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        _require_non_empty_str("name", self.name)
        tokens = tuple(self.tokens)
        if not tokens:
            raise ValueError(f"{self.name} vocabulary must not be empty.")

        ids = [token.id for token in tokens]
        labels = [token.label for token in tokens]
        if len(ids) != len(set(ids)):
            raise ValueError(f"{self.name} vocabulary contains duplicate token ids.")
        if len(labels) != len(set(labels)):
            raise ValueError(f"{self.name} vocabulary contains duplicate labels.")

        object.__setattr__(self, "tokens", tokens)
        object.__setattr__(self, "_id_map", MappingProxyType({token.id: token for token in tokens}))
        object.__setattr__(self, "_label_map", MappingProxyType({token.label: token for token in tokens}))

    @property
    def id_map(self) -> Mapping[int, TokenT]:
        return self._id_map

    @property
    def label_map(self) -> Mapping[str, TokenT]:
        return self._label_map

    def token_for_id(self, token_id: int) -> TokenT:
        return self._id_map[token_id]

    def token_for_label(self, label: str) -> TokenT:
        return self._label_map[label]

    def has_id(self, token_id: int) -> bool:
        return token_id in self._id_map

    def __iter__(self) -> Iterator[TokenT]:
        return iter(self.tokens)

    def __len__(self) -> int:
        return len(self.tokens)


@dataclass(frozen=True)
class Vocabularies:
    """Shared token vocabularies for all current BeatState dimensions."""

    meters: TokenVocabulary[MeterToken]
    beat_positions: TokenVocabulary[BeatPositionToken]
    boundaries: TokenVocabulary[BoundaryToken]
    keys: TokenVocabulary[KeyToken]
    chords: TokenVocabulary[ChordToken]
    roles: TokenVocabulary[RoleToken]
    heads: TokenVocabulary[HeadToken]
    grooves: TokenVocabulary[GrooveToken]


def _build_meter_vocabulary(signatures: Sequence[str]) -> TokenVocabulary[MeterToken]:
    tokens = []
    for token_id, signature in enumerate(dict.fromkeys(signatures)):
        beats_per_bar, _ = _parse_meter_signature(signature)
        tokens.append(
            MeterToken(
                id=token_id,
                label=signature,
                beats_per_bar=beats_per_bar,
                strong_beats=_strong_beats_for_meter(beats_per_bar),
            )
        )
    return TokenVocabulary(name="meters", tokens=tuple(tokens))


def _build_beat_position_vocabulary(max_beats_per_bar: int) -> TokenVocabulary[BeatPositionToken]:
    if max_beats_per_bar <= 0:
        raise ValueError("max_beats_per_bar must be > 0.")
    tokens = tuple(
        BeatPositionToken(id=idx, label=f"beat_{idx + 1}", index=idx)
        for idx in range(max_beats_per_bar)
    )
    return TokenVocabulary(name="beat_positions", tokens=tokens)


def _build_boundary_vocabulary() -> TokenVocabulary[BoundaryToken]:
    tokens = (
        BoundaryToken(id=0, label="none", level=0),
        BoundaryToken(id=1, label="local", level=1),
        BoundaryToken(id=2, label="phrase", level=2),
        BoundaryToken(id=3, label="section", level=3),
    )
    return TokenVocabulary(name="boundaries", tokens=tokens)


def _build_key_vocabulary(cardinality: int) -> TokenVocabulary[KeyToken]:
    if cardinality <= 0:
        raise ValueError("key cardinality must be > 0.")
    labels = _pitch_class_labels(cardinality)
    tokens = tuple(KeyToken(id=idx, label=labels[idx], root_pc=idx) for idx in range(cardinality))
    return TokenVocabulary(name="keys", tokens=tokens)


def _build_chord_vocabulary(chord_vocabulary_size: int) -> TokenVocabulary[ChordToken]:
    if chord_vocabulary_size <= 0:
        raise ValueError("chord_vocabulary_size must be > 0.")
    quality_count = len(CORE_CHORD_QUALITIES)
    if chord_vocabulary_size % quality_count != 0:
        raise ValueError("chord_vocabulary_size must be a multiple of the core chord-quality count.")

    root_cardinality = chord_vocabulary_size // quality_count
    root_labels = _pitch_class_labels(root_cardinality)

    tokens = []
    token_id = 0
    for root_pc, root_label in enumerate(root_labels):
        for quality in CORE_CHORD_QUALITIES:
            tokens.append(
                ChordToken(
                    id=token_id,
                    label=f"{root_label}{quality}",
                    root_pc=root_pc,
                    quality=quality,
                )
            )
            token_id += 1

    return TokenVocabulary(name="chords", tokens=tuple(tokens))


def _build_role_vocabulary() -> TokenVocabulary[RoleToken]:
    tokens = (
        RoleToken(id=0, label="hold", description="Maintain the current harmonic function."),
        RoleToken(id=1, label="prep", description="Prepare an upcoming structural change."),
        RoleToken(id=2, label="change", description="Introduce a local structural departure."),
        RoleToken(id=3, label="cad", description="Drive toward a cadential arrival."),
    )
    return TokenVocabulary(name="roles", tokens=tokens)


def _build_head_vocabulary() -> TokenVocabulary[HeadToken]:
    tokens = (
        HeadToken(id=0, label="rest", description="No melodic head anchor on this beat."),
        HeadToken(id=1, label="root", description="Anchor the melodic head on the chord root."),
        HeadToken(id=2, label="third", description="Anchor the melodic head on the chord third."),
        HeadToken(id=3, label="fifth", description="Anchor the melodic head on the chord fifth."),
        HeadToken(id=4, label="seventh", description="Anchor the melodic head on the chord seventh."),
        HeadToken(id=5, label="extension", description="Anchor the head on a color tone or extension."),
        HeadToken(id=6, label="upper_approach", description="Approach the target head from above."),
        HeadToken(id=7, label="lower_approach", description="Approach the target head from below."),
    )
    return TokenVocabulary(name="heads", tokens=tokens)


def _build_groove_vocabulary(families: Sequence[str]) -> TokenVocabulary[GrooveToken]:
    tokens = []
    groove_shapes = {
        "straight": (8, 16),
        "syncopated": (8, 16),
        "swing": (8,),
    }

    token_id = 0
    for family in dict.fromkeys(families):
        _require_non_empty_str("groove family", family)
        subdivisions = groove_shapes.get(family, (8, 16))
        for subdivision in subdivisions:
            tokens.append(
                GrooveToken(
                    id=token_id,
                    label=f"{family}_{subdivision}ths",
                    family=family,
                    subdivision=subdivision,
                )
            )
            token_id += 1

    return TokenVocabulary(name="grooves", tokens=tuple(tokens))


def build_default_vocabularies(
    style_config: StyleConfig | None = None,
    *,
    edo: int | None = None,
) -> Vocabularies:
    """Build the shared first-pass structural vocabularies.

    Key and chord roots cover every pitch class in ``Z_edo``.  Style vocabulary
    sizes remain lower bounds, but can never truncate the EDO pitch-class space.
    """

    meter_signatures = (
        DEFAULT_METER_SIGNATURES if style_config is None else style_config.allowed_meters
    )
    groove_families = (
        DEFAULT_GROOVE_FAMILIES if style_config is None else style_config.groove_families
    )
    if edo is None:
        key_cardinality = 12 if style_config is None else style_config.key_vocabulary_size
        chord_vocabulary_size = 48 if style_config is None else style_config.chord_vocabulary_size
    else:
        if not isinstance(edo, int) or isinstance(edo, bool) or edo < 1:
            raise ValueError("edo must be a positive integer.")
        key_cardinality = edo
        chord_vocabulary_size = edo * len(CORE_CHORD_QUALITIES)
    meters = _build_meter_vocabulary(meter_signatures)
    beat_positions = _build_beat_position_vocabulary(
        max(token.beats_per_bar for token in meters)
    )
    return Vocabularies(
        meters=meters,
        beat_positions=beat_positions,
        boundaries=_build_boundary_vocabulary(),
        keys=_build_key_vocabulary(key_cardinality),
        chords=_build_chord_vocabulary(chord_vocabulary_size),
        roles=_build_role_vocabulary(),
        heads=_build_head_vocabulary(),
        grooves=_build_groove_vocabulary(groove_families),
    )


DEFAULT_VOCABULARIES = build_default_vocabularies()
