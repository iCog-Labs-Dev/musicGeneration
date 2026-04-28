from typing import Tuple
from config import EDOConfig, MicrotonalRendering


class EDO:
    """
    Represents an Equal Division of the Octave (EDO) system
    and provides utilities for pitch math.
    """

    def __init__(self, config: EDOConfig):
        self.config = config

    def pitch_class(self, h: int) -> int:
        """
        Calculates the pitch class for a given pitch height.

        Args:
            h: The pitch height in EDO steps.

        Returns:
            The pitch class as an integer in Z_n.
        """
        return h % self.config.n

    def to_midi(self, h: int) -> Tuple[int, int]:
        """
        Converts an EDO pitch height to a MIDI note and pitch bend.

        Args:
            h: The pitch height in EDO steps.

        Returns:
            A tuple containing the MIDI note number and the pitch bend value.
            The pitch bend is an integer from -8192 to 8191.
        """
        if self.config.n == 12:
            # For 12-EDO, mapping is direct
            return (int(self.config.base_tuning + h), 0)

        if self.config.microtonal_rendering_method == MicrotonalRendering.MTS:
            # MTS would be handled by the renderer, not per-note.
            # For now, we can return a direct mapping and assume the
            # renderer is configured.
            # This is a simplification.
            midi_note = self.config.base_tuning + (h * 12 / self.config.n)
            return (round(midi_note), 0)

        # MPE rendering
        midi_note_float = self.config.base_tuning + h * (12 / self.config.n)
        nearest_midi_note = round(midi_note_float)
        
        pitch_diff_cents = (midi_note_float - nearest_midi_note) * 100
        
        # Calculate pitch bend value
        bend_range_cents = self.config.pitch_bend_range * 100
        bend_fraction = pitch_diff_cents / bend_range_cents
        
        # Clamp bend_fraction to [-0.5, 0.5] to stay within the nearest note
        bend_fraction = max(-0.5, min(0.5, bend_fraction))

        # Pitch bend is from -8192 to 8191
        pitch_bend = int(bend_fraction * 8191 * 2)

        return (int(nearest_midi_note), pitch_bend)

    def __repr__(self) -> str:
        return f"EDO(n={self.config.n})"
