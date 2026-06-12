import unittest

from aimusic.core.config import DecodeConfig
from aimusic.core.core_types import BeatState, NoteEvent
from aimusic.core.vocab import DEFAULT_VOCABULARIES
from aimusic.decode import (
    DEFAULT_TICKS_PER_BEAT,
    _cleanup_events,
    build_subbeat_grid,
    decode_path_to_score,
    generate_bass_events,
    generate_comping_events,
    generate_lead_events,
)
from aimusic.theory.tonal import chord_pitch_classes


VOCABS = DEFAULT_VOCABULARIES


def state(
    *,
    meter: str = "4/4",
    beat: int = 0,
    boundary: str = "none",
    key: str = "C",
    chord: str = "Cmaj",
    role: str = "hold",
    head: str = "root",
    groove: str = "straight_8ths",
) -> BeatState:
    return BeatState(
        meter_id=VOCABS.meters.token_for_label(meter).id,
        beat_in_bar=beat,
        boundary_lvl=VOCABS.boundaries.token_for_label(boundary).id,
        key_id=VOCABS.keys.token_for_label(key).id,
        chord_id=VOCABS.chords.token_for_label(chord).id,
        role_id=VOCABS.roles.token_for_label(role).id,
        head_id=VOCABS.heads.token_for_label(head).id,
        groove_id=VOCABS.grooves.token_for_label(groove).id,
    )


class TestDecodeGrid(unittest.TestCase):
    def test_build_subbeat_grid_matches_quantization(self):
        path = (state(beat=0), state(beat=1))
        grid = build_subbeat_grid(
            path,
            decode_config=DecodeConfig(subbeats_per_beat=4),
            include_terminal_state=True,
        )

        self.assertEqual(grid[0], (0, 120, 240, 360))
        self.assertEqual(grid[1], (480, 600, 720, 840))


class TestDecodeTracks(unittest.TestCase):
    def test_lead_head_anchor_stays_on_chord_tones(self):
        path = (
            state(chord="Cmaj", head="root"),
            state(beat=1, chord="Cmaj", head="third"),
            state(beat=2, chord="Cmaj", head="fifth"),
        )

        lead_events = generate_lead_events(
            path,
            decode_config=DecodeConfig(lead_density=1.0),
            vocabularies=VOCABS,
            include_terminal_state=True,
        )
        chord_pcs = chord_pitch_classes(0, "maj", 12)

        self.assertEqual(len(lead_events), 3)
        for event in lead_events:
            self.assertIn(event.h % 12, chord_pcs)

    def test_major_chord_seventh_head_uses_major_seventh(self):
        path = (state(chord="Cmaj", head="seventh"),)

        lead_events = generate_lead_events(
            path, vocabularies=VOCABS, include_terminal_state=True
        )

        self.assertEqual(len(lead_events), 1)
        self.assertEqual(lead_events[0].h % 12, 11)

    def test_decode_respects_register_limits(self):
        path = (
            state(chord="Cmaj", head="root"),
            state(beat=1, chord="G7", head="third"),
            state(beat=2, chord="Amin", head="fifth"),
            state(beat=3, chord="Fmaj", head="third"),
        )

        score = decode_path_to_score(path, vocabularies=VOCABS)

        for event in score.note_events:
            if event.track == "bass":
                self.assertTrue(28 <= event.h <= 52)
            elif event.track == "comping":
                self.assertTrue(45 <= event.h <= 72)
            elif event.track == "lead":
                self.assertTrue(60 <= event.h <= 88)

    def test_decode_defaults_to_excluding_terminal_endpoint_state(self):
        path = (
            state(chord="Cmaj", head="root"),
            state(beat=1, chord="G7", head="third"),
            state(beat=2, chord="Amin", head="fifth"),
        )

        score = decode_path_to_score(path, vocabularies=VOCABS)
        lead_events = [event for event in score.note_events if event.track == "lead"]

        self.assertEqual(len(lead_events), 2)

    def test_comping_voice_leading_stays_local(self):
        path = (
            state(chord="Cmaj"),
            state(beat=1, chord="G7"),
            state(beat=2, chord="Amin"),
            state(beat=3, chord="Fmaj"),
        )
        comping = generate_comping_events(
            path, vocabularies=VOCABS, include_terminal_state=True
        )

        by_onset = {}
        for event in comping:
            by_onset.setdefault(event.ton, []).append(event.h)
        ordered = [tuple(sorted(by_onset[ton])) for ton in sorted(by_onset)]
        for previous, current in zip(ordered, ordered[1:]):
            for next_pitch in current:
                self.assertLessEqual(
                    min(abs(next_pitch - prev_pitch) for prev_pitch in previous),
                    7,
                )
            for prev_pitch in previous:
                self.assertLessEqual(
                    min(abs(prev_pitch - next_pitch) for next_pitch in current),
                    7,
                )

    def test_density_controls_can_suppress_tracks(self):
        path = (
            state(chord="Cmaj", head="root"),
            state(beat=1, chord="Cmaj", head="third"),
        )
        decode_config = DecodeConfig(
            bass_density=0.0,
            comping_density=0.0,
            lead_density=0.0,
            drum_density=0.0,
        )

        score = decode_path_to_score(
            path, decode_config=decode_config, vocabularies=VOCABS
        )

        self.assertEqual(score.note_events, ())

    def test_density_changes_event_counts_for_lead(self):
        path = (
            state(chord="Cmaj", head="root"),
            state(beat=1, chord="Cmaj", head="third"),
            state(beat=2, chord="G7", head="fifth"),
            state(beat=3, chord="Amin", head="root"),
            state(beat=0, chord="Fmaj", head="third"),
        )

        sparse_score = decode_path_to_score(
            path,
            decode_config=DecodeConfig(lead_density=0.1),
            vocabularies=VOCABS,
            include_terminal_state=True,
        )
        dense_score = decode_path_to_score(
            path,
            decode_config=DecodeConfig(lead_density=1.0),
            vocabularies=VOCABS,
            include_terminal_state=True,
        )

        sparse_lead = [
            event for event in sparse_score.note_events if event.track == "lead"
        ]
        dense_lead = [
            event for event in dense_score.note_events if event.track == "lead"
        ]
        self.assertLess(len(sparse_lead), len(dense_lead))

    def test_tension_changes_velocity_expression_and_articulation_proxy(self):
        hold_path = (state(chord="Cmaj", role="hold", head="root"),)
        cad_path = (state(chord="Cmaj", role="cad", head="root"),)
        change_path = (state(chord="Cmaj", role="change", head="root"),)

        hold_bass = generate_bass_events(
            hold_path,
            decode_config=DecodeConfig(bass_density=1.0),
            vocabularies=VOCABS,
            include_terminal_state=True,
        )
        cad_bass = generate_bass_events(
            cad_path,
            decode_config=DecodeConfig(bass_density=1.0),
            vocabularies=VOCABS,
            include_terminal_state=True,
        )
        change_bass = generate_bass_events(
            change_path,
            decode_config=DecodeConfig(bass_density=1.0),
            vocabularies=VOCABS,
            include_terminal_state=True,
        )

        self.assertEqual(len(hold_bass), 1)
        self.assertEqual(len(cad_bass), 1)
        self.assertEqual(len(change_bass), 1)
        self.assertLess(hold_bass[0].v, cad_bass[0].v)
        self.assertLess(hold_bass[0].e[0], cad_bass[0].e[0])
        self.assertEqual(hold_bass[0].toff - hold_bass[0].ton, DEFAULT_TICKS_PER_BEAT)
        self.assertEqual(
            change_bass[0].toff - change_bass[0].ton, DEFAULT_TICKS_PER_BEAT // 2
        )

    def test_overlap_cleanup_truncates_first_same_track_same_pitch_event(self):
        overlapping = (
            NoteEvent(
                ton=0,
                toff=DEFAULT_TICKS_PER_BEAT,
                h=60,
                v=0.5,
                e=(0.2,),
                track="lead",
            ),
            NoteEvent(
                ton=DEFAULT_TICKS_PER_BEAT // 2,
                toff=(DEFAULT_TICKS_PER_BEAT // 2) + DEFAULT_TICKS_PER_BEAT,
                h=60,
                v=0.7,
                e=(0.4,),
                track="lead",
            ),
        )

        cleaned = _cleanup_events(overlapping)

        self.assertEqual(len(cleaned), 2)
        self.assertEqual(cleaned[0].ton, 0)
        self.assertEqual(cleaned[0].toff, DEFAULT_TICKS_PER_BEAT // 2)
        self.assertEqual(cleaned[1].ton, DEFAULT_TICKS_PER_BEAT // 2)
        self.assertEqual(
            cleaned[1].toff, (DEFAULT_TICKS_PER_BEAT // 2) + DEFAULT_TICKS_PER_BEAT
        )

    def test_bass_prep_role_uses_lower_neighbor_approach_into_next_root(self):
        path = (
            state(chord="Cmaj", role="hold", head="root", boundary="phrase"),
            state(beat=1, chord="Cmaj", role="prep", head="root"),
            state(beat=2, chord="Cmaj", role="cad", head="root"),
        )

        bass_events = generate_bass_events(
            path,
            decode_config=DecodeConfig(bass_density=1.0),
            vocabularies=VOCABS,
            include_terminal_state=True,
        )

        self.assertEqual(len(bass_events), 3)
        self.assertEqual(bass_events[1].h % 12, 11)
        self.assertEqual(bass_events[2].h % 12, 0)
        self.assertEqual((bass_events[2].h - bass_events[1].h) % 12, 1)

    def test_decode_generates_multi_track_score(self):
        path = (
            state(chord="Cmaj", head="root", boundary="phrase"),
            state(beat=1, chord="G7", role="prep", head="third"),
            state(beat=2, chord="Cmaj", role="cad", head="root"),
            state(beat=3, chord="Fmaj", role="change", head="fifth"),
        )

        score = decode_path_to_score(
            path,
            vocabularies=VOCABS,
            ticks_per_beat=DEFAULT_TICKS_PER_BEAT,
            include_terminal_state=True,
        )
        tracks = score.track_event_counts()

        self.assertGreater(tracks["bass"], 0)
        self.assertGreater(tracks["comping"], 0)
        self.assertGreater(tracks["lead"], 0)
        self.assertGreater(tracks["drums"], 0)


if __name__ == "__main__":
    unittest.main()
