import unittest

import numpy as np

from student_fidelity.baselines import (
    DINABaseline,
    GaussianCopulaBaseline,
    IndependentBernoulliBaseline,
    IRT2PLBaseline,
)
from student_fidelity.data import make_synthetic_cdm_dataset


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


if __name__ == "__main__":
    unittest.main()
