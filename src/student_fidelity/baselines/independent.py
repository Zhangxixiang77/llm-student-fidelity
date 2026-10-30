"""Independent Bernoulli baseline: preserves item marginals only."""

from __future__ import annotations

from typing import Self

import numpy as np
from numpy.typing import NDArray

from student_fidelity.data import ResponseDataset


class IndependentBernoulliBaseline:
    def fit(self, dataset: ResponseDataset) -> Self:
        self.pass_rates_ = np.clip(dataset.responses.mean(axis=0), 1e-6, 1 - 1e-6)
        return self

    def sample(self, n_students: int, seed: int) -> NDArray[np.int8]:
        self._check_fitted()
        rng = np.random.default_rng(seed)
        probabilities = np.broadcast_to(self.pass_rates_, (n_students, self.pass_rates_.size))
        return rng.binomial(1, probabilities).astype(np.int8)

    def _check_fitted(self) -> None:
        if not hasattr(self, "pass_rates_"):
            raise RuntimeError("fit the baseline before sampling")
