import unittest

import numpy as np

from student_fidelity.metrics import (
    dependency_rmse,
    ideal_dina_responses,
    implicit_slip_guess,
    readback_agreement,
)
from student_fidelity.tetrachoric import tetrachoric_correlation


class MetricTests(unittest.TestCase):
    def test_tetrachoric_is_high_for_identical_items(self):
        x = np.array([0, 0, 1, 1] * 10)
        self.assertGreater(tetrachoric_correlation(x, x), 0.9)

    def test_tetrachoric_is_near_zero_for_independent_items(self):
        x = np.array([0] * 40 + [1] * 40)
        y = np.array([0] * 20 + [1] * 20 + [0] * 20 + [1] * 20)
        self.assertAlmostEqual(tetrachoric_correlation(x, y), 0.0, places=3)

    def test_dependency_rmse_zero_for_identical_matrices(self):
        rng = np.random.default_rng(4)
        responses = rng.binomial(1, 0.5, size=(40, 5))
        self.assertLess(dependency_rmse(responses, responses), 1e-10)

    def test_dina_ideal_response_and_controllability_metrics(self):
        alpha = np.array([[1, 0], [1, 1]])
        q_matrix = np.array([[1, 0], [0, 1]])
        np.testing.assert_array_equal(ideal_dina_responses(alpha, q_matrix), [[1, 0], [1, 1]])
        agreement = readback_agreement(alpha, np.array([[1, 0], [0, 1]]))
        self.assertEqual(agreement["attribute_readback_agreement"], 0.75)
        self.assertEqual(agreement["pattern_readback_agreement"], 0.5)

        alpha = np.array([[1, 0], [1, 1], [0, 0]])
        responses = np.array([[1, 0], [0, 1], [1, 0]])
        result = implicit_slip_guess(responses, alpha, q_matrix)
        self.assertEqual(result["item_slip"], [0.5, 0.0])
        self.assertEqual(result["item_guess"], [1.0, 0.0])


if __name__ == "__main__":
    unittest.main()
