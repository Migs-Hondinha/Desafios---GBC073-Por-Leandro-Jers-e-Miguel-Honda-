"""Verify scientific selection rules without running a full sweep."""
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parent


class ProtocolTests(unittest.TestCase):
    def get_module(self):
        self.assertTrue((ROOT / 'protocolo.py').exists(), 'Protocol runner is missing')
        import protocolo
        return protocolo

    def test_selection_excludes_divergence_and_breaks_ties_by_loss(self):
        module = self.get_module()
        rows = [dict(validation_accuracy='0.6', validation_loss='1.2', diverged='False'),
                dict(validation_accuracy='0.6', validation_loss='1.0', diverged='False'),
                dict(validation_accuracy='0.9', validation_loss='0.1', diverged='True')]
        self.assertIs(module.best(rows), rows[1])

    def test_grid_extends_when_best_rate_is_at_either_edge(self):
        module = self.get_module()
        self.assertAlmostEqual(module.next_rate([0.001, 0.01, 0.1], 0.001), 0.0001)
        self.assertAlmostEqual(module.next_rate([0.001, 0.01, 0.1], 0.1), 1.0)
        self.assertIsNone(module.next_rate([0.001, 0.01, 0.1], 0.01))

    def test_paired_gain_requires_all_three_seeds(self):
        module = self.get_module()
        result = module.gain_statistics([0.50, 0.51, 0.49], [0.55, 0.56, 0.54])
        self.assertAlmostEqual(result['gain_mean_pp'], 5.0)
        self.assertTrue(result['exceeds_variation'])
        uncertain = module.gain_statistics([0.5, 0.5, 0.5], [0.6, 0.4, 0.51])
        self.assertFalse(uncertain['exceeds_variation'])
        with self.assertRaises(ValueError):
            module.gain_statistics([0.5], [0.6])


if __name__ == '__main__':
    unittest.main()
