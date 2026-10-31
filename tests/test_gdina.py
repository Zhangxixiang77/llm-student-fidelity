import unittest

import numpy as np

from student_fidelity.baselines import GDINABaseline, make_baseline
from student_fidelity.data import make_synthetic_cdm_dataset


class GDINATests(unittest.TestCase):
    def test_fit_evaluate_and_sample(self):
        dataset = make_synthetic_cdm_dataset(
            n_students=50, n_items=6, n_attributes=3, seed=5
        )
        model = GDINABaseline(max_iter=8).fit(dataset)
        self.assertEqual(model.p_correct_.shape, (8, 6))
        self.assertTrue(np.all((model.p_correct_ > 0) & (model.p_correct_ < 1)))
        self.assertEqual(model.log_likelihood_by_student(dataset.responses).shape, (50,))
        self.assertTrue(np.allclose(model.posterior_patterns(dataset.responses).sum(axis=1), 1))
        self.assertGreaterEqual(model.srmsr(dataset.responses), 0)
        alpha = model.sample_patterns(8, seed=1)
        generated = model.sample(8, seed=2, alpha=alpha)
        self.assertEqual(generated.shape, (8, 6))
        self.assertTrue(np.isin(generated, [0, 1]).all())
        self.assertEqual(model.readback_ceiling(alpha, repeats=2)["n_repeats"], 2)
        self.assertIsInstance(make_baseline("gdina"), GDINABaseline)

    def test_sampling_uses_dina_pattern_bit_order(self):
        dataset = make_synthetic_cdm_dataset(
            n_students=20, n_items=4, n_attributes=2, seed=9
        )
        model = GDINABaseline(max_iter=2).fit(dataset)
        probabilities = np.array([0.05, 0.35, 0.65, 0.95])
        model.p_correct_ = np.repeat(probabilities[:, None], 4, axis=1)
        alpha = np.tile(
            np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.int8), (500, 1)
        )
        generated = model.sample(alpha.shape[0], seed=4, alpha=alpha)
        observed = generated.reshape(500, 4, 4).mean(axis=(0, 2))
        self.assertTrue(np.all(np.abs(observed - probabilities) < 0.04))
        residuals = model.response_residuals(np.ones_like(generated), alpha)
        self.assertAlmostEqual(residuals["gdina_mean_residual"], 0.5)


if __name__ == "__main__":
    unittest.main()
