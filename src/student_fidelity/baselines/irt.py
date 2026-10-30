"""A compact marginal-maximum-likelihood 2PL IRT generative baseline."""

from __future__ import annotations

from typing import Self

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.special import expit, logsumexp

from student_fidelity.data import ResponseDataset


class IRT2PLBaseline:
    def __init__(self, n_quad: int = 31, max_iter: int = 30, tolerance: float = 1e-4):
        self.n_quad = n_quad
        self.max_iter = max_iter
        self.tolerance = tolerance

    def fit(self, dataset: ResponseDataset) -> Self:
        responses = dataset.responses.astype(float)
        n_students, n_items = responses.shape
        nodes, weights = np.polynomial.hermite.hermgauss(self.n_quad)
        self.theta_nodes_ = nodes * np.sqrt(2)
        log_weights = np.log(weights / np.sqrt(np.pi))
        pass_rates = np.clip(responses.mean(axis=0), 0.01, 0.99)
        discrimination = np.ones(n_items)
        difficulty = -np.log(pass_rates / (1 - pass_rates))
        posterior = np.full((n_students, self.n_quad), 1 / self.n_quad)

        for _ in range(self.max_iter):
            log_likelihood = np.zeros((n_students, self.n_quad))
            for item in range(n_items):
                probabilities = expit(
                    discrimination[item] * (self.theta_nodes_ - difficulty[item])
                )
                probabilities = np.clip(probabilities, 1e-8, 1 - 1e-8)
                log_likelihood += responses[:, item, None] * np.log(probabilities)[None, :]
                log_likelihood += (1 - responses[:, item, None]) * np.log1p(-probabilities)[None, :]
            log_posterior = log_likelihood + log_weights[None, :]
            posterior = np.exp(log_posterior - logsumexp(log_posterior, axis=1)[:, None])

            expected_n = posterior.sum(axis=0)
            updated_a = discrimination.copy()
            updated_b = difficulty.copy()
            for item in range(n_items):
                expected_r = (posterior * responses[:, item, None]).sum(axis=0)

                def objective(
                    params: NDArray[np.float64],
                    expected_r: NDArray[np.float64] = expected_r,
                    expected_n: NDArray[np.float64] = expected_n,
                ) -> tuple[float, list[float]]:
                    a, b = params
                    logits = a * (self.theta_nodes_ - b)
                    log_p = -np.logaddexp(0, -logits)
                    log_not_p = -np.logaddexp(0, logits)
                    value = -np.sum(expected_r * log_p + (expected_n - expected_r) * log_not_p)
                    residual = expected_r - expected_n * expit(logits)
                    gradient = [
                        -np.sum(residual * (self.theta_nodes_ - b)),
                        np.sum(residual * a),
                    ]
                    return float(value), gradient

                result = minimize(
                    objective,
                    x0=np.array([discrimination[item], difficulty[item]]),
                    jac=True,
                    method="L-BFGS-B",
                    bounds=[(0.05, 4.0), (-4.0, 4.0)],
                )
                updated_a[item], updated_b[item] = result.x

            change = max(
                float(np.max(np.abs(updated_a - discrimination))),
                float(np.max(np.abs(updated_b - difficulty))),
            )
            discrimination, difficulty = updated_a, updated_b
            if change < self.tolerance:
                break

        self.discrimination_ = discrimination
        self.difficulty_ = difficulty
        self.posterior_ = posterior
        self.n_training_students_ = n_students
        return self

    def sample_theta(self, n_students: int, seed: int) -> NDArray[np.float64]:
        """Sample from the empirical posterior ability distribution of the fitting half."""

        self._check_fitted()
        rng = np.random.default_rng(seed)
        students = rng.integers(0, self.n_training_students_, size=n_students)
        nodes = np.array(
            [rng.choice(self.n_quad, p=self.posterior_[student]) for student in students]
        )
        return self.theta_nodes_[nodes]

    def sample(self, n_students: int, seed: int) -> NDArray[np.int8]:
        theta = self.sample_theta(n_students, seed)
        rng = np.random.default_rng(seed + 1)
        logits = self.discrimination_[None, :] * (theta[:, None] - self.difficulty_[None, :])
        return rng.binomial(1, expit(logits)).astype(np.int8)

    def _check_fitted(self) -> None:
        if not hasattr(self, "posterior_"):
            raise RuntimeError("fit the baseline before sampling")
