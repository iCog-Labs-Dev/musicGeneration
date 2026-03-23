from dataclasses import dataclass
from typing import Optional, Sequence

W_II_V = 1.5
W_V_I = 2.0
W_II_V_I = 4.0
W_TRITONE_SUB = 1.5
W_BACKDOOR = 2.0
W_DOWNBEAT = 0.5
W_LONG_DURATION = 0.35


@dataclass(frozen=True)
class ChordState:
    root_pc: int
    quality: str
    bass_pc: Optional[int]
    duration: float
    bar_position: float


def pc(x: int) -> int:
    return x % 12


def is_minorish(q: str) -> bool:
    return q in {"min", "m7", "m9", "m11"}


def is_dominant(q: str) -> bool:
    return q in {"7", "9", "13", "7b9", "7#11", "sus7"}


def is_tonic_family(q: str) -> bool:
    return q in {"maj", "maj7", "6", "69", "maj9", "min", "m6", "m69"}


def is_fifth_down(a: int, b: int) -> bool:
    return pc(a - b) == 7


def duration_factor(c: ChordState) -> float:
    return 1.0 + W_LONG_DURATION * max(c.duration - 1.0, 0.0)


def downbeat_bonus(c: ChordState) -> float:
    return W_DOWNBEAT if 0.0 <= c.bar_position < 0.2 else 0.0


def transition_score(a: ChordState, b: ChordState) -> float:
    s = 0.0
    if (
        is_minorish(a.quality)
        and is_dominant(b.quality)
        and is_fifth_down(a.root_pc, b.root_pc)
    ):
        s += W_II_V
    if (
        is_dominant(a.quality)
        and is_tonic_family(b.quality)
        and is_fifth_down(a.root_pc, b.root_pc)
    ):
        s += W_V_I
    if (
        is_dominant(a.quality)
        and is_tonic_family(b.quality)
        and pc(a.root_pc - b.root_pc) == 1
    ):
        s += W_TRITONE_SUB
    if (
        is_dominant(a.quality)
        and is_tonic_family(b.quality)
        and pc(a.root_pc - b.root_pc) == 10
    ):
        s += W_BACKDOOR
    s += downbeat_bonus(b)
    return s * duration_factor(b)


def local_window_score(prev, curr, next_):
    if (
        prev
        and next_
        and is_minorish(prev.quality)
        and is_dominant(curr.quality)
        and is_tonic_family(next_.quality)
        and is_fifth_down(prev.root_pc, curr.root_pc)
        and is_fifth_down(curr.root_pc, next_.root_pc)
    ):
        return W_II_V_I * duration_factor(next_) + downbeat_bonus(next_)
    return 0.0


def sequence_score(seq):
    total = sum(transition_score(seq[i], seq[i + 1]) for i in range(len(seq) - 1))
    total += sum(
        local_window_score(seq[i - 1], seq[i], seq[i + 1])
        for i in range(1, len(seq) - 1)
    )
    return total
