from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence, Tuple, TypeVar


T = TypeVar("T")
_MASK_64 = (1 << 64) - 1


def _require_int(name: str, value: int, *, minimum: int = 0) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an int.")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}.")


def _mix64(value: int) -> int:
    """SplitMix64-style integer mixer for deterministic key derivation."""
    value &= _MASK_64
    value ^= value >> 30
    value = (value * 0xBF58476D1CE4E5B9) & _MASK_64
    value ^= value >> 27
    value = (value * 0x94D049BB133111EB) & _MASK_64
    value ^= value >> 31
    return value & _MASK_64


def _seed_for(seed: int, stream: int) -> int:
    """Derive a concrete Python RNG seed from the logical key."""
    return _mix64(seed ^ (stream * 0x9E3779B97F4A7C15))


@dataclass(frozen=True)
class RNGKey:
    """Immutable random key with explicit stream threading."""

    seed: int
    stream: int = 0

    def __post_init__(self) -> None:
        _require_int("seed", self.seed, minimum=0)
        _require_int("stream", self.stream, minimum=0)

    def next_key(self, steps: int = 1) -> "RNGKey":
        """Advance the key stream deterministically."""
        _require_int("steps", steps, minimum=1)
        return RNGKey(seed=self.seed, stream=self.stream + steps)

    def split(self, count: int = 2) -> Tuple["RNGKey", ...]:
        """Derive child keys without mutating the current key."""
        _require_int("count", count, minimum=1)
        base = self.stream + 1
        return tuple(RNGKey(seed=self.seed, stream=base + idx) for idx in range(count))

    def spawn(self, tag: int) -> "RNGKey":
        """Derive a deterministic child stream keyed by an explicit tag."""
        _require_int("tag", tag, minimum=0)
        derived_stream = _mix64(self.stream ^ (tag + 1)) & 0x7FFFFFFF
        return RNGKey(seed=self.seed, stream=derived_stream)

    def generator(self) -> random.Random:
        """Create an isolated Python RNG for the current key."""
        return random.Random(_seed_for(self.seed, self.stream))


def random_unit(key: RNGKey) -> tuple[float, RNGKey]:
    """Sample a float in [0, 1) and return the advanced key."""
    rng = key.generator()
    return rng.random(), key.next_key()


def randint(key: RNGKey, start: int, stop: int) -> tuple[int, RNGKey]:
    """Sample an integer in [start, stop) and return the advanced key."""
    _require_int("start", start)
    _require_int("stop", stop)
    if stop <= start:
        raise ValueError("stop must be > start.")
    rng = key.generator()
    return rng.randrange(start, stop), key.next_key()


def choice(key: RNGKey, values: Sequence[T]) -> tuple[T, RNGKey]:
    """Sample one element from a non-empty sequence and return the advanced key."""
    if not values:
        raise ValueError("values must not be empty.")
    index, next_key_value = randint(key, 0, len(values))
    return values[index], next_key_value


def shuffle(key: RNGKey, values: Sequence[T]) -> tuple[Tuple[T, ...], RNGKey]:
    """Return a shuffled tuple plus the advanced key."""
    rng = key.generator()
    shuffled = list(values)
    rng.shuffle(shuffled)
    return tuple(shuffled), key.next_key()

