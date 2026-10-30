import tempfile
import unittest
from pathlib import Path

import numpy as np

from student_fidelity.data import ResponseDataset, make_synthetic_cdm_dataset, train_test_split


class DataTests(unittest.TestCase):
    def test_directory_roundtrip_and_split(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            dataset = make_synthetic_cdm_dataset(n_students=20, n_items=6, n_attributes=3, seed=1)
            dataset.to_directory(path)
            loaded = ResponseDataset.from_directory(path)
            np.testing.assert_array_equal(loaded.responses, dataset.responses)
            np.testing.assert_array_equal(loaded.q_matrix, dataset.q_matrix)
            self.assertEqual(loaded.item_texts, dataset.item_texts)

            fitting, holdout = train_test_split(loaded, seed=7)
            self.assertEqual(fitting.n_students, 10)
            self.assertEqual(holdout.n_students, 10)
            self.assertTrue(set(fitting.student_ids).isdisjoint(holdout.student_ids))

    def test_invalid_q_matrix_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least one attribute"):
            ResponseDataset(responses=np.ones((3, 1)), q_matrix=np.zeros((1, 2)))


if __name__ == "__main__":
    unittest.main()
