"""tonal.py — Tonal system definition, chord templates, tonal distances.

Implements concepts from Lerdahl's Tonal Pitch Space (TPS) in an
EDO-generic manner.  All pitch classes live in Z_N where N is the
number of equal divisions of the octave.

Design principles (per readme §2.1):
  - Frozen dataclasses for data objects.
  - Pure functions with no hidden global state.
  - EDO as a simple parameter N, not hard-coded to 12.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Dict, FrozenSet, Tuple

from edo import EDO


# ===================================================================
# 1. Interval helpers — EDO-generic, pure functions
# ===================================================================

def pc(x: int, edo: int = 12) -> int:
    """Pitch-class: x mod edo."""
    return x % edo


def get_fifth_steps(edo: int) -> int:
    """Number of EDO steps in a perfect fifth (≈ 7/12 of the octave)."""
    return round(7 * edo / 12)


def get_fourth_steps(edo: int) -> int:
    """Number of EDO steps in a perfect fourth (≈ 5/12 of the octave)."""
    return round(5 * edo / 12)


def get_major_third_steps(edo: int) -> int:
    """Number of EDO steps in a major third (≈ 4/12 of the octave)."""
    return round(4 * edo / 12)


def get_minor_third_steps(edo: int) -> int:
    """Number of EDO steps in a minor third (≈ 3/12 of the octave)."""
    return round(3 * edo / 12)


def get_major_second_steps(edo: int) -> int:
    """Number of EDO steps in a major second / whole tone (≈ 2/12)."""
    return round(2 * edo / 12)


def get_minor_second_steps(edo: int) -> int:
    """Number of EDO steps in a minor second / semitone (≈ 1/12)."""
    return round(1 * edo / 12)


def is_fifth_down(a: int, b: int, edo: int = 12) -> bool:
    """True when root motion a → b is a perfect fifth downward.

    This is the standard circle-of-fifths resolution direction
    (e.g. G → C in 12-EDO, or step-11 → step-0 in 19-EDO).
    """
    return pc(a - b, edo) == get_fifth_steps(edo)


def is_fifth_up(a: int, b: int, edo: int = 12) -> bool:
    """True when root motion a → b is a perfect fifth upward.

    This is the Plagal cadence direction (IV → I, e.g. F → C).
    """
    return pc(b - a, edo) == get_fifth_steps(edo)


def is_major_second_up(a: int, b: int, edo: int = 12) -> bool:
    """True when root motion a → b is a major second upward.

    This is the Deceptive cadence direction (V → vi, e.g. G → A).
    """
    return pc(b - a, edo) == get_major_second_steps(edo)


# ===================================================================
# 2. Chord templates — defined in 12-EDO semitones, scaled to any EDO
# ===================================================================

# Each value is a tuple of intervals measured in 12-EDO semitones
# from the root.  The function chord_pitch_classes() scales these
# to the target EDO via round(interval * edo / 12).
_CHORD_INTERVALS_12: Dict[str, Tuple[int, ...]] = {
    # --- Triads ---
    "maj":   (0, 4, 7),
    "min":   (0, 3, 7),
    "dim":   (0, 3, 6),
    "aug":   (0, 4, 8),
    # --- Seventh chords ---
    "7":     (0, 4, 7, 10),
    "maj7":  (0, 4, 7, 11),
    "m7":    (0, 3, 7, 10),
    "m7b5":  (0, 3, 6, 10),
    "dim7":  (0, 3, 6, 9),
    # --- Sixth chords ---
    "6":     (0, 4, 7, 9),
    "m6":    (0, 3, 7, 9),
    "69":    (0, 2, 4, 7, 9),
    "m69":   (0, 2, 3, 7, 9),
    # --- Extended chords ---
    "9":     (0, 2, 4, 7, 10),
    "maj9":  (0, 2, 4, 7, 11),
    "m9":    (0, 2, 3, 7, 10),
    "m11":   (0, 2, 3, 5, 7, 10),
    "13":    (0, 2, 4, 5, 7, 9, 10),
    # --- Altered dominants ---
    "7b9":   (0, 1, 4, 7, 10),
    "7#11":  (0, 4, 6, 7, 10),
    # --- Suspended ---
    "sus4":  (0, 5, 7),
    "sus2":  (0, 2, 7),
    "sus7":  (0, 5, 7, 10),
}

ALL_QUALITIES: Tuple[str, ...] = tuple(sorted(_CHORD_INTERVALS_12.keys()))


def _scale_interval(semitones_12: int, edo: int) -> int:
    """Convert a 12-EDO interval to its nearest EDO-step equivalent."""
    return round(semitones_12 * edo / 12)


def chord_pitch_classes(
    root_pc: int, quality: str, edo: int = 12
) -> FrozenSet[int]:
    """Expand a chord symbol into its constituent pitch classes.

    Args:
        root_pc: Root pitch class in Z_edo.
        quality: Chord quality string (e.g. "maj7", "m7", "7").
        edo:     Number of equal divisions of the octave.

    Returns:
        Frozenset of pitch classes belonging to the chord.

    Raises:
        ValueError: If *quality* is not a recognized template.
    """
    intervals = _CHORD_INTERVALS_12.get(quality)
    if intervals is None:
        raise ValueError(f"Unknown chord quality: {quality!r}")
    return frozenset(
        pc(root_pc + _scale_interval(iv, edo), edo) for iv in intervals
    )


# ===================================================================
# 3. Functional classification
# ===================================================================

_MINOR_QUALITIES = frozenset({"min", "m7", "m9", "m11", "m6", "m69", "m7b5"})
_DOMINANT_QUALITIES = frozenset({"7", "9", "13", "7b9", "7#11", "sus7"})
_TONIC_QUALITIES = frozenset({
    "maj", "maj7", "6", "69", "maj9", "min", "m6", "m69",
})
_SUBDOMINANT_QUALITIES = frozenset({
    "maj", "maj7", "min", "m7", "sus4", "sus2", "6",
})


def is_minorish(q: str) -> bool:
    """True if the quality belongs to the minor family."""
    return q in _MINOR_QUALITIES


def is_dominant(q: str) -> bool:
    """True if the quality belongs to the dominant-seventh family."""
    return q in _DOMINANT_QUALITIES


def is_tonic_family(q: str) -> bool:
    """True if the quality can function as a tonic chord."""
    return q in _TONIC_QUALITIES


def is_subdominant(q: str) -> bool:
    """True if the quality commonly appears on subdominant degrees.

    Note: Actual subdominant *function* also depends on scale degree;
    this classifier only checks the quality string.
    """
    return q in _SUBDOMINANT_QUALITIES


# ===================================================================
# 4. Tonal distance — Lerdahl's Tonal Pitch Space
# ===================================================================

@lru_cache(maxsize=4096)
def tonal_distance(a_root: int, b_root: int, edo: int = 12) -> float:
    """Circle-of-fifths distance between two roots (Lerdahl's *j*).

    Returns the shortest path on the circle of fifths, measured in
    fifth-steps.  Symmetric: d(a, b) == d(b, a).

    In 12-EDO, tonal_distance(0, 7) == 1  (C–G, one fifth apart).
    In 12-EDO, tonal_distance(0, 6) == 6  (C–F♯, tritone = maximally far).
    """
    f = get_fifth_steps(edo)

    def _cof_position(root: int) -> int:
        """Find the position of *root* on the circle of fifths."""
        for i in range(edo):
            if (i * f) % edo == root % edo:
                return i
        return 0  # fallback for degenerate EDOs

    pos_a = _cof_position(a_root)
    pos_b = _cof_position(b_root)
    diff = abs(pos_a - pos_b)
    return min(diff, edo - diff)


@lru_cache(maxsize=4096)
def basic_space_distance(
    a_root: int, a_quality: str,
    b_root: int, b_quality: str,
    edo: int = 12,
) -> float:
    """Lerdahl's TPS distance  δ(x, y) = j + k.

    *j* — Circle-of-fifths distance between roots.
    *k* — Non-common pitch classes across the basic-space hierarchy.

    The basic space has three levels (simplified from Lerdahl's five,
    since we operate without a global key context):

        Level a  –  {root}
        Level b  –  {root, fifth}
        Level c  –  all chord tones

    *k* is the sum of symmetric-difference sizes across all levels.
    Deeper levels (a, b) weigh heavier by being included in every
    higher level's comparison, exactly as in Lerdahl's formulation:
    a root change is more disruptive than a colour-tone change.

    Falls back to :func:`tonal_distance` alone when either quality
    string is unrecognised.
    """
    j = tonal_distance(a_root, b_root, edo)

    if (a_quality not in _CHORD_INTERVALS_12
            or b_quality not in _CHORD_INTERVALS_12):
        return float(j)

    fifth = get_fifth_steps(edo)

    # Build hierarchical basic-space levels for each chord.
    a_levels = (
        frozenset({pc(a_root, edo)}),                            # level a
        frozenset({pc(a_root, edo), pc(a_root + fifth, edo)}),   # level b
        chord_pitch_classes(a_root, a_quality, edo),             # level c
    )
    b_levels = (
        frozenset({pc(b_root, edo)}),                            # level b
        frozenset({pc(b_root, edo), pc(b_root + fifth, edo)}),   # level b
        chord_pitch_classes(b_root, b_quality, edo),             # level c
    )

    # k = total non-common tones across levels
    k = sum(
        len(al.symmetric_difference(bl))
        for al, bl in zip(a_levels, b_levels)
    )

    return float(j + k)


@lru_cache(maxsize=4096)
def nearest_roots(root_pc: int, edo: int = 12, limit: int = 3) -> Tuple[int, ...]:
    """Return the nearest pitch-class roots under the tonal metric.

    The result excludes ``root_pc`` itself and is ordered from most-related
    to less-related, using pitch-class order as a deterministic tiebreaker.
    """
    if limit <= 0:
        return ()

    candidates = []
    for other in range(edo):
        if other == root_pc % edo:
            continue
        candidates.append((tonal_distance(root_pc, other, edo), other))

    candidates.sort(key=lambda item: (item[0], item[1]))
    return tuple(root for _, root in candidates[:limit])


# ===================================================================
# 5. TonalSystem — high-level OO wrapper
# ===================================================================

class TonalSystem:
    """Defines a tonal system for a given EDO.

    Provides access to chord templates, functional classification,
    and tonal distance metrics.  Wraps the pure-function API above
    with an object that remembers the EDO context.
    """

    def __init__(self, edo: EDO):
        self.edo = edo
        self._n: int = edo.config.n

    # -- Chord templates ------------------------------------------------

    def available_qualities(self) -> Tuple[str, ...]:
        """Return all recognized chord-quality strings, sorted."""
        return ALL_QUALITIES

    def chord_pcs(self, root_pc: int, quality: str) -> FrozenSet[int]:
        """Expand a chord into its pitch classes under this EDO."""
        return chord_pitch_classes(root_pc, quality, self._n)

    # -- Distance metrics -----------------------------------------------

    def distance(
        self,
        a_root: int, a_quality: str,
        b_root: int, b_quality: str,
    ) -> float:
        """Lerdahl's TPS distance δ = j + k for this EDO."""
        return basic_space_distance(
            a_root, a_quality, b_root, b_quality, self._n
        )

    def fifths_distance(self, a_root: int, b_root: int) -> float:
        """Circle-of-fifths distance (j-component only)."""
        return tonal_distance(a_root, b_root, self._n)

    # -- Functional classification (static, quality-only) ---------------

    @staticmethod
    def classify(quality: str) -> str:
        """Classify a chord quality into a functional family.

        Returns one of: ``'dominant'``, ``'tonic'``, ``'diminished'``,
        ``'minor'``, ``'subdominant'``, or ``'unknown'``.

        Priority order matches GTTM's tension hierarchy:
        dominant > tonic > diminished > minor > subdominant.
        """
        if is_dominant(quality):
            return "dominant"
        if is_tonic_family(quality):
            return "tonic"
        if quality in {"dim", "dim7", "m7b5"}:
            return "diminished"
        if is_minorish(quality):
            return "minor"
        if is_subdominant(quality):
            return "subdominant"
        return "unknown"

    def __repr__(self) -> str:
        return f"TonalSystem(edo={self.edo})"
