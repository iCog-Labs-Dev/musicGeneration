import unittest
from dataclasses import FrozenInstanceError

from config import (
    DecodeConfig,
    EDOConfig,
    NeuralPriorConfig,
    PlanConfig,
    PlanMethod,
    PlaceholderPriorMode,
    PriorFactorization,
    PriorWeights,
    SBBackend,
    SBConfig,
    SectioningStrategy,
    StyleConfig,
)


class TestEDOConfig(unittest.TestCase):
    def test_edo_config_validates_positive_values(self):
        cfg = EDOConfig(n=19, base_tuning=57.0, pitch_bend_range=24)
        self.assertEqual(cfg.n, 19)
        self.assertEqual(cfg.base_tuning, 57.0)
        self.assertEqual(cfg.pitch_bend_range, 24)

    def test_edo_config_rejects_invalid_values(self):
        with self.assertRaises(ValueError):
            EDOConfig(n=0)
        with self.assertRaises(ValueError):
            EDOConfig(n=12, pitch_bend_range=0)
        with self.assertRaises(TypeError):
            EDOConfig(n=True)


class TestStyleConfig(unittest.TestCase):
    def test_style_config_coerces_sequences_to_tuples(self):
        cfg = StyleConfig(
            allowed_meters=["4/4", "7/4"],
            subdivision_patterns=[3, 4, 5],
            groove_families=["straight", "swing"],
            bass_register=[30, 54],
            comping_register=[45, 70],
            lead_register=[60, 86],
            typical_density_range=[0.2, 0.9],
        )

        self.assertEqual(cfg.allowed_meters, ("4/4", "7/4"))
        self.assertEqual(cfg.subdivision_patterns, (3, 4, 5))
        self.assertEqual(cfg.groove_families, ("straight", "swing"))
        self.assertEqual(cfg.bass_register, (30, 54))
        self.assertEqual(cfg.typical_density_range, (0.2, 0.9))

    def test_style_config_rejects_invalid_ranges(self):
        with self.assertRaises(ValueError):
            StyleConfig(groove_families=[])
        with self.assertRaises(ValueError):
            StyleConfig(bass_register=(52, 28))


class TestPriorWeights(unittest.TestCase):
    def test_prior_weights_accept_non_negative_values(self):
        weights = PriorWeights(lambda_data=1.2, lambda_gttm=0.8, harmonic=1.5)
        self.assertEqual(weights.lambda_data, 1.2)
        self.assertEqual(weights.harmonic, 1.5)

    def test_prior_weights_reject_fully_disabled_or_negative_weights(self):
        with self.assertRaises(ValueError):
            PriorWeights(lambda_data=0.0, lambda_gttm=0.0)
        with self.assertRaises(ValueError):
            PriorWeights(harmonic=-0.1)


class TestSBConfig(unittest.TestCase):
    def test_sb_config_defaults_to_numpy_backend(self):
        cfg = SBConfig()
        self.assertEqual(cfg.backend_selection, SBBackend.NUMPY)

    def test_sb_config_rejects_invalid_numerics(self):
        with self.assertRaises(ValueError):
            SBConfig(horizon_t=0)
        with self.assertRaises(ValueError):
            SBConfig(temperature=0.0)
        with self.assertRaises(ValueError):
            SBConfig(tolerance=0.0)
        with self.assertRaises(TypeError):
            SBConfig(backend_selection="numpy")  # type: ignore[arg-type]


class TestNeuralPriorConfig(unittest.TestCase):
    def test_neural_prior_config_validates_runtime_metadata(self):
        cfg = NeuralPriorConfig(
            model_family="flax_transformer",
            model_version="v0.1.0",
            factorization_mode=PriorFactorization.FACTORIZED,
            checkpoint_path="artifacts/prior.ckpt",
            tokenizer_path="artifacts/tokens.json",
            manifest_path="artifacts/prior_manifest.json",
            batch_size=64,
            placeholder_mode=PlaceholderPriorMode.STRUCTURED,
            default_logp=-0.25,
        )
        self.assertEqual(cfg.model_family, "flax_transformer")
        self.assertEqual(cfg.batch_size, 64)
        self.assertEqual(cfg.default_logp, -0.25)

    def test_neural_prior_config_rejects_invalid_values(self):
        with self.assertRaises(ValueError):
            NeuralPriorConfig(model_family="")
        with self.assertRaises(ValueError):
            NeuralPriorConfig(batch_size=0)
        with self.assertRaises(TypeError):
            NeuralPriorConfig(factorization_mode="factorized")  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            NeuralPriorConfig(placeholder_mode="structured")  # type: ignore[arg-type]


class TestDecodeConfig(unittest.TestCase):
    def test_decode_config_validates_ranges(self):
        cfg = DecodeConfig(
            drum_density=0.8,
            bass_density=0.6,
            comping_density=0.5,
            lead_density=0.4,
            tension_velocity_range=(0.6, 0.95),
        )
        self.assertEqual(cfg.tension_velocity_range, (0.6, 0.95))

    def test_decode_config_rejects_invalid_voice_counts_and_density(self):
        with self.assertRaises(ValueError):
            DecodeConfig(lead_density=1.1)
        with self.assertRaises(ValueError):
            DecodeConfig(min_comping_voices=5, max_comping_voices=4)


class TestPlanConfig(unittest.TestCase):
    def test_plan_config_validates_method_b_midpoint_and_sectioning(self):
        cfg = PlanConfig(
            method=PlanMethod.METHOD_B,
            loop_midpoint=16,
            sectioning_strategy=SectioningStrategy.SECTION_WISE,
            section_names=("intro", "return"),
        )
        self.assertEqual(cfg.loop_midpoint, 16)
        self.assertEqual(cfg.section_names, ("intro", "return"))

    def test_plan_config_rejects_invalid_combinations(self):
        with self.assertRaises(ValueError):
            PlanConfig(method=PlanMethod.METHOD_B)
        with self.assertRaises(ValueError):
            PlanConfig(loop_midpoint=8)
        with self.assertRaises(ValueError):
            PlanConfig(sectioning_strategy=SectioningStrategy.SECTION_WISE)
        with self.assertRaises(TypeError):
            PlanConfig(method="method_a")  # type: ignore[arg-type]


class TestConfigImmutability(unittest.TestCase):
    def test_configs_are_frozen(self):
        cfg = StyleConfig()
        with self.assertRaises(FrozenInstanceError):
            cfg.allowed_meters = ("3/4",)


if __name__ == "__main__":
    unittest.main()
