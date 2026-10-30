"""Gaussian copula baseline: preserves item marginals and pairwise dependence."""

from __future__ import annotations

from typing import Self

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

from student_fidelity.data import ResponseDataset
from student_fidelity.tetrachoric import nearest_psd_correlation, tetrachoric_matrix


class GaussianCopulaBaseline:
    def fit(self, dataset: ResponseDataset) -> Self:
        self.pass_rates_ = np.clip(dataset.responses.mean(axis=0), 1e-6, 1 - 1e-6)
        self.correlation_ = nearest_psd_correlation(tetrachoric_matrix(dataset.responses))
        self.thresholds_ = norm.ppf(self.pass_rates_)
        return self

    def sample(self, n_students: int, seed: int) -> NDArray[np.int8]:
        self._check_fitted()
        rng = np.random.default_rng(seed)
        latent = rng.multivariate_normal(
            mean=np.zeros(self.pass_rates_.size), cov=self.correlation_, size=n_students
        )
        return (latent <= self.thresholds_[None, :]).astype(np.int8)

    def _check_fitted(self) -> None:
        if not hasattr(self, "correlation_"):
            raise RuntimeError("fit the baseline before sampling")
