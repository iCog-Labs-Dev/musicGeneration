from config import EDOConfig, MicrotonalRendering
from edo import EDO
from tonal import TonalSystem


def main():
    """Demonstrates the use of the EDO formalization."""

    # 1. Configure and create a 12-EDO system
    print("--- 12-EDO System ---")
    config_12_edo = EDOConfig(n=12)
    edo_12 = EDO(config_12_edo)
    tonal_system_12 = TonalSystem(edo_12)

    print(f"Configuration: {config_12_edo}")
    print(f"EDO System: {edo_12}")
    print(f"Tonal System: {tonal_system_12}")

    # Demonstrate pitch calculations for 12-EDO
    pitch_height_c4 = 0  # Middle C
    pitch_height_g4 = 7  # G above Middle C
    pitch_height_c5 = 12  # C one octave higher

    print(f"Pitch height {pitch_height_c4} -> Pitch Class: {edo_12.pitch_class(pitch_height_c4)}")
    print(f"Pitch height {pitch_height_g4} -> Pitch Class: {edo_12.pitch_class(pitch_height_g4)}")
    print(f"Pitch height {pitch_height_c5} -> Pitch Class: {edo_12.pitch_class(pitch_height_c5)}")

    print(f"Pitch height {pitch_height_c4} -> MIDI: {edo_12.to_midi(pitch_height_c4)}")
    print(f"Pitch height {pitch_height_g4} -> MIDI: {edo_12.to_midi(pitch_height_g4)}")
    print(f"Pitch height {pitch_height_c5} -> MIDI: {edo_12.to_midi(pitch_height_c5)}")

    print("\n" + "=" * 20 + "\n")

    # 2. Configure and create a 19-EDO system
    print("--- 19-EDO System ---")
    config_19_edo = EDOConfig(n=19, microtonal_rendering_method=MicrotonalRendering.MPE)
    edo_19 = EDO(config_19_edo)
    tonal_system_19 = TonalSystem(edo_19)

    print(f"Configuration: {config_19_edo}")
    print(f"EDO System: {edo_19}")
    print(f"Tonal System: {tonal_system_19}")

    # Demonstrate pitch calculations for 19-EDO
    # In 19-EDO, a "perfect fifth" is 11 steps.
    pitch_height_c4_19 = 0
    pitch_height_g4_19 = 11
    pitch_height_c5_19 = 19

    print(f"Pitch height {pitch_height_c4_19} -> Pitch Class: {edo_19.pitch_class(pitch_height_c4_19)}")
    print(f"Pitch height {pitch_height_g4_19} -> Pitch Class: {edo_19.pitch_class(pitch_height_g4_19)}")
    print(f"Pitch height {pitch_height_c5_19} -> Pitch Class: {edo_19.pitch_class(pitch_height_c5_19)}")

    # Note how non-12-EDO pitches are mapped to MIDI note + pitch bend
    print(f"Pitch height {pitch_height_c4_19} -> MIDI: {edo_19.to_midi(pitch_height_c4_19)}")
    print(f"Pitch height {pitch_height_g4_19} -> MIDI: {edo_19.to_midi(pitch_height_g4_19)}")
    print(f"Pitch height {pitch_height_c5_19} -> MIDI: {edo_19.to_midi(pitch_height_c5_19)}")


if __name__ == "__main__":
    main()
