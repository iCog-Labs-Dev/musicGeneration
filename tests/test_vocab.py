import unittest

from aimusic.core.config import StyleConfig
from aimusic.core.vocab import (
    CORE_CHORD_QUALITIES,
    DEFAULT_VOCABULARIES,
    KeyToken,
    TokenVocabulary,
    build_default_vocabularies,
)


class TestDefaultVocabularies(unittest.TestCase):
    def test_required_vocabularies_exist(self):
        vocabs = DEFAULT_VOCABULARIES

        self.assertGreater(len(vocabs.meters), 0)
        self.assertGreater(len(vocabs.beat_positions), 0)
        self.assertGreater(len(vocabs.boundaries), 0)
        self.assertGreater(len(vocabs.keys), 0)
        self.assertGreater(len(vocabs.chords), 0)
        self.assertGreater(len(vocabs.roles), 0)
        self.assertGreater(len(vocabs.heads), 0)
        self.assertGreater(len(vocabs.grooves), 0)

    def test_default_meter_vocab_is_lookupable_and_musically_named(self):
        meters = DEFAULT_VOCABULARIES.meters

        self.assertEqual(meters.token_for_id(0).label, "4/4")
        self.assertEqual(meters.token_for_label("3/4").beats_per_bar, 3)
        self.assertEqual(meters.token_for_label("5/4").strong_beats, (0, 3))

    def test_default_chord_vocab_matches_first_pass_12_edo_size(self):
        chords = DEFAULT_VOCABULARIES.chords

        self.assertEqual(len(chords), 12 * len(CORE_CHORD_QUALITIES))
        self.assertEqual(chords.token_for_id(0).label, "Cmaj")
        self.assertEqual(chords.token_for_id(1).label, "Cmin")
        self.assertEqual(chords.token_for_label("G7").root_pc, 7)

    def test_default_roles_use_structural_role_labels(self):
        roles = DEFAULT_VOCABULARIES.roles
        self.assertEqual(tuple(token.label for token in roles), ("hold", "prep", "change", "cad"))


class TestVocabularyIntegrity(unittest.TestCase):
    def test_token_vocabulary_rejects_duplicate_ids(self):
        with self.assertRaises(ValueError):
            TokenVocabulary(
                name="keys",
                tokens=(
                    KeyToken(id=0, label="C", root_pc=0),
                    KeyToken(id=0, label="D", root_pc=2),
                ),
            )

    def test_token_vocabulary_rejects_duplicate_labels(self):
        with self.assertRaises(ValueError):
            TokenVocabulary(
                name="keys",
                tokens=(
                    KeyToken(id=0, label="C", root_pc=0),
                    KeyToken(id=1, label="C", root_pc=1),
                ),
            )


class TestStyleDrivenVocabularyBuild(unittest.TestCase):
    def test_explicit_19_edo_covers_all_pitch_classes(self):
        vocabs = build_default_vocabularies(StyleConfig(), edo=19)
        self.assertEqual(tuple(token.root_pc for token in vocabs.keys), tuple(range(19)))
        self.assertEqual({token.root_pc for token in vocabs.chords}, set(range(19)))
        self.assertEqual(len(vocabs.chords), 19 * len(CORE_CHORD_QUALITIES))

    def test_build_vocabularies_uses_style_config_sizes_and_labels(self):
        style = StyleConfig(
            allowed_meters=("4/4", "9/8"),
            groove_families=("straight", "odd"),
            chord_vocabulary_size=76,
            key_vocabulary_size=19,
        )

        vocabs = build_default_vocabularies(style)

        self.assertEqual(len(vocabs.meters), 2)
        self.assertEqual(vocabs.meters.token_for_label("9/8").beats_per_bar, 9)
        self.assertEqual(len(vocabs.beat_positions), 9)
        self.assertEqual(len(vocabs.keys), 19)
        self.assertEqual(len(vocabs.chords), 76)
        self.assertTrue(vocabs.grooves.has_id(3))
        self.assertEqual(vocabs.grooves.token_for_label("odd_16ths").family, "odd")


if __name__ == "__main__":
    unittest.main()
