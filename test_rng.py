import unittest

from rng import RNGKey, choice, randint, random_unit, shuffle


class TestRNGKey(unittest.TestCase):
    def test_key_rejects_invalid_values(self):
        with self.assertRaises(ValueError):
            RNGKey(seed=-1)
        with self.assertRaises(TypeError):
            RNGKey(seed=True)  # type: ignore[arg-type]

    def test_next_key_advances_stream(self):
        key = RNGKey(seed=123, stream=4)
        self.assertEqual(key.next_key().stream, 5)
        self.assertEqual(key.next_key(3).stream, 7)

    def test_split_is_deterministic_and_ordered(self):
        key = RNGKey(seed=7, stream=2)
        split_a = key.split(3)
        split_b = key.split(3)

        self.assertEqual(split_a, split_b)
        self.assertEqual(tuple(child.stream for child in split_a), (3, 4, 5))

    def test_spawn_is_deterministic_by_tag(self):
        key = RNGKey(seed=99, stream=3)
        self.assertEqual(key.spawn(5), key.spawn(5))
        self.assertNotEqual(key.spawn(5), key.spawn(6))


class TestPureSamplingHelpers(unittest.TestCase):
    def test_random_unit_is_reproducible(self):
        key = RNGKey(seed=1234)

        value_a, next_key_a = random_unit(key)
        value_b, next_key_b = random_unit(key)

        self.assertEqual(value_a, value_b)
        self.assertEqual(next_key_a, next_key_b)
        self.assertEqual(next_key_a.stream, 1)

    def test_randint_choice_and_shuffle_thread_keys_explicitly(self):
        key = RNGKey(seed=42)

        randint_value, key_after_randint = randint(key, 10, 20)
        choice_value, key_after_choice = choice(key_after_randint, ("a", "b", "c"))
        shuffled, key_after_shuffle = shuffle(key_after_choice, (1, 2, 3, 4))

        self.assertTrue(10 <= randint_value < 20)
        self.assertIn(choice_value, ("a", "b", "c"))
        self.assertEqual(sorted(shuffled), [1, 2, 3, 4])
        self.assertEqual(key_after_randint.stream, 1)
        self.assertEqual(key_after_choice.stream, 2)
        self.assertEqual(key_after_shuffle.stream, 3)

    def test_choice_and_shuffle_reject_invalid_sequences(self):
        key = RNGKey(seed=1)
        with self.assertRaises(ValueError):
            choice(key, ())
        with self.assertRaises(ValueError):
            randint(key, 5, 5)


if __name__ == "__main__":
    unittest.main()
