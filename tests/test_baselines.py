import unittest

import numpy as np

from student_fidelity.baselines import (
    DINABaseline,
    GaussianCopulaBaseline,
    IndependentBernoulliBaseline,
    IRT2PLBaseline,
)
from student_fidelity.data import ResponseDataset, make_synthetic_cdm_dataset


class BaselineTests(unittest.TestCase):
    def test_baselines_generate_aligned_binary_responses(self):
        baselines = [
            IndependentBernoulliBaseline(),
            GaussianCopulaBaseline(),
            IRT2PLBaseline(n_quad=15, max_iter=4),
            DINABaseline(max_iter=8),
        ]
        for baseline in baselines:
            with self.subTest(baseline=type(baseline).__name__):
                dataset = make_synthetic_cdm_dataset(
                    n_students=50, n_items=6, n_attributes=3, seed=5
                )
                baseline.fit(dataset)
                generated = baseline.sample(12, seed=9)
                self.assertEqual(generated.shape, (12, 6))
                self.assertTrue(np.isin(generated, [0, 1]).all())

    def test_dina_assigned_alpha_and_ceiling(self):
        dataset = make_synthetic_cdm_dataset(n_students=60, n_items=8, n_attributes=3, seed=3)
        model = DINABaseline(max_iter=10).fit(dataset)
        alpha = model.sample_patterns(10, seed=1)
        generated = model.sample(10, seed=2, alpha=alpha)
        self.assertEqual(generated.shape, (10, 8))
        ceiling = model.readback_ceiling(alpha, repeats=3)
        self.assertEqual(
            set(ceiling),
            {
                "attribute_readback_agreement",
                "pattern_readback_agreement",
                "attribute_readback_by_attribute",
                "n_repeats",
            },
        )
        self.assertEqual(len(ceiling["attribute_readback_by_attribute"]), 3)
        self.assertEqual(ceiling["n_repeats"], 3)
        confidence = model.classification_confidence(dataset.responses)
        self.assertEqual(len(confidence["attribute_classification_confidence"]), 3)
        self.assertGreaterEqual(model.srmsr(dataset.responses), 0)

    def test_dina_response_residuals_use_conditional_predictions(self):
        model = DINABaseline()
        model.q_matrix_ = np.array([[1], [1]], dtype=np.int8)
        model.slip_ = np.array([0.2, 0.2])
        model.guess_ = np.array([0.1, 0.1])
        model.pattern_prior_ = np.array([0.5, 0.5])
        alpha = np.array([[0], [1]], dtype=np.int8)
        responses = np.array([[1, 0], [0, 1]], dtype=np.int8)
        residuals = model.response_residuals(responses, alpha)
        self.assertAlmostEqual(residuals["dina_mean_residual"], 0.05)
        self.assertAlmostEqual(residuals["dina_mean_master_shortfall"], 0.3)
        self.assertAlmostEqual(residuals["dina_mean_nonmaster_excess"], 0.4)
        np.testing.assert_allclose(
            residuals["dina_item_predicted_p_correct"], [0.45, 0.45]
        )
        np.testing.assert_allclose(
            residuals["dina_item_nonmaster_excess"], [0.9, -0.1]
        )

    def test_dina_em_uses_studentwise_nonmastery_probabilities(self):
        dataset = ResponseDataset(
            responses=np.array([[0], [1], [1]], dtype=np.int8),
            q_matrix=np.array([[1]], dtype=np.int8),
            item_texts=["one item"],
            answer_keys=["correct"],
        )
        model = DINABaseline(max_iter=1).fit(dataset)
        # With slip=guess=0.1, the first E step gives mastery probabilities
        # [0.1, 0.9, 0.9]. Nonmastery is 1 minus these values, not N minus them.
        self.assertAlmostEqual(model.guess_[0], 2 / 11)
        self.assertAlmostEqual(model.slip_[0], 1 / 19)

    def test_dina_multistart_is_reproducible_and_selects_best_likelihood(self):
        dataset = make_synthetic_cdm_dataset(
            n_students=60, n_items=8, n_attributes=3, seed=23
        )
        first = DINABaseline(max_iter=30, n_starts=4, seed=17).fit(dataset)
        second = DINABaseline(max_iter=30, n_starts=4, seed=17).fit(dataset)
        self.assertEqual(len(first.fit_history_), 4)
        self.assertIsNone(first.fit_history_[0]["start_seed"])
        self.assertEqual(first.best_start_, second.best_start_)
        self.assertAlmostEqual(
            first.training_log_likelihood_,
            max(start["training_log_likelihood"] for start in first.fit_history_),
        )
        np.testing.assert_allclose(first.slip_, second.slip_)
        np.testing.assert_allclose(first.guess_, second.guess_)
        np.testing.assert_allclose(first.pattern_prior_, second.pattern_prior_)
        np.testing.assert_allclose(first.posterior_.sum(axis=1), 1)
        with self.assertRaises(ValueError):
            DINABaseline(n_starts=0)


if __name__ == "__main__":
    unittest.main()
