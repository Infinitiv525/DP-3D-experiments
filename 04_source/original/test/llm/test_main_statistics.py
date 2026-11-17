"""Synthetic-only statistics regression checks; no test dataset access."""
import unittest
import numpy as np
from main_statistics import primary_contrasts


class MainStatisticsTests(unittest.TestCase):
    def test_identical_methods_zero_differences(self):
        x = np.random.default_rng(1).uniform(size=(1, 4, 3, 5))
        result = primary_contrasts(np.repeat(x, 6, axis=0))
        self.assertTrue(all(c['lower'] == c['upper'] == 0 and not c['supports_lower_error'] for c in result['contrasts']))

    def test_constant_effect_and_repeat_pairing(self):
        x = np.ones((6, 3, 3, 5)); x[2:] -= .125
        result = primary_contrasts(x)
        self.assertTrue(all(c['lower'] == c['upper'] == -.125 and c['supports_lower_error']
                            and c['paired_repeat_effects'] == [-.125] * 3 for c in result['contrasts']))
        self.assertEqual(result['marginal_confidence'], .9875)

    def test_seed_reproducibility(self):
        x = np.random.default_rng(1).uniform(size=(6, 3, 3, 5))
        self.assertEqual(primary_contrasts(x), primary_contrasts(x))

    def test_incomplete_nonfinite_or_negative_rejected(self):
        for x in (np.ones((6, 3, 2, 5)), np.full((6, 3, 3, 5), np.nan), np.full((6, 3, 3, 5), -.1)):
            with self.assertRaises(ValueError): primary_contrasts(x)

    def test_settings_cannot_change_after_freeze(self):
        with self.assertRaises(ValueError): primary_contrasts(np.ones((6, 3, 3, 5)), replicates=9999)


if __name__ == '__main__': unittest.main(verbosity=2)
