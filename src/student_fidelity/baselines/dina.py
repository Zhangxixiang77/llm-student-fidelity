"""DINA EM baseline with a categorical prior over all mastery patterns."""

from __future__ import annotations

from typing import Self

import numpy as np
from numpy.typing import NDArray
from scipy.special import logsumexp

from student_fidelity.data import ResponseDataset
from student_fidelity.metrics import ideal_dina_responses


def _pair_correlation(
    item_probability: NDArray[np.float64], joint_probability: NDArray[np.float64]
) -> NDArray[np.float64]:
    variance = item_probability * (1 - item_probability)
    denominator = np.sqrt(variance[:, None] * variance[None, :])
    with np.errstate(divide="ignore", invalid="ignore"):
        residual = joint_probability - item_probability[:, None] * item_probability[None, :]
        return residual / denominator


class DINABaseline:
    def __init__(self, max_iter: int = 100, tolerance: float = 1e-4):
        self.max_iter = max_iter
        self.tolerance = tolerance

    @staticmethod
    def enumerate_patterns(n_attributes: int) -> NDArray[np.int8]:
        if not 1 <= n_attributes <= 16:
            raise ValueError("the exhaustive DINA implementation supports 1 to 16 attributes")
        integers = np.arange(2**n_attributes, dtype=np.uint32)
        return ((integers[:, None] >> np.arange(n_attributes)) & 1).astype(np.int8)

    def fit(self, dataset: ResponseDataset) -> Self:
        responses = dataset.responses.astype(float)
        self.q_matrix_ = dataset.q_matrix.copy()
        self.patterns_ = self.enumerate_patterns(dataset.n_attributes)
        self.eta_ = ideal_dina_responses(self.patterns_, self.q_matrix_)
        n_students = dataset.n_students
        n_patterns = self.patterns_.shape[0]
        pattern_prior = np.full(n_patterns, 1 / n_patterns)
        slip = np.full(dataset.n_items, 0.1)
        guess = np.full(dataset.n_items, 0.1)
        posterior = np.full((n_students, n_patterns), 1 / n_patterns)

        for _ in range(self.max_iter):
            log_likelihood = self._pattern_log_likelihood(responses, slip, guess)
            log_posterior = log_likelihood + np.log(np.clip(pattern_prior, 1e-12, 1))[None, :]
            posterior = np.exp(log_posterior - logsumexp(log_posterior, axis=1)[:, None])

            pattern_prior = np.clip(posterior.mean(axis=0), 1e-12, 1)
            pattern_prior /= pattern_prior.sum()
            expected_mastery = posterior @ self.eta_
            expected_nonmastery = n_students - expected_mastery
            updated_slip = np.divide(
                (expected_mastery * (1 - responses)).sum(axis=0),
                expected_mastery.sum(axis=0),
                out=np.zeros(dataset.n_items),
                where=expected_mastery.sum(axis=0) > 1e-10,
            )
            updated_guess = np.divide(
                (expected_nonmastery * responses).sum(axis=0),
                expected_nonmastery.sum(axis=0),
                out=np.zeros(dataset.n_items),
                where=expected_nonmastery.sum(axis=0) > 1e-10,
            )
            updated_slip = np.clip(updated_slip, 1e-4, 1 - 1e-4)
            updated_guess = np.clip(updated_guess, 1e-4, 1 - 1e-4)
            change = max(
                float(np.max(np.abs(updated_slip - slip))),
                float(np.max(np.abs(updated_guess - guess))),
            )
            slip, guess = updated_slip, updated_guess
            if change < self.tolerance:
                break

        self.slip_ = slip
        self.guess_ = guess
        self.pattern_prior_ = pattern_prior
        self.posterior_ = posterior
        return self

    def _pattern_log_likelihood(
        self, responses: NDArray[np.float64], slip: NDArray[np.float64], guess: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        result = np.empty((responses.shape[0], self.patterns_.shape[0]))
        for pattern in range(self.patterns_.shape[0]):
            p_correct = np.where(self.eta_[pattern] == 1, 1 - slip, guess)
            p_correct = np.clip(p_correct, 1e-8, 1 - 1e-8)
            result[:, pattern] = (
                responses * np.log(p_correct)[None, :]
                + (1 - responses) * np.log1p(-p_correct)[None, :]
            ).sum(axis=1)
        return result

    def posterior_patterns(self, responses: NDArray[np.int8]) -> NDArray[np.float64]:
        self._check_fitted()
        log_likelihood = self._pattern_log_likelihood(
            np.asarray(responses, dtype=float), self.slip_, self.guess_
        )
        log_posterior = log_likelihood + np.log(np.clip(self.pattern_prior_, 1e-12, 1))[None, :]
        return np.exp(log_posterior - logsumexp(log_posterior, axis=1)[:, None])

    def log_likelihood_by_student(
        self, responses: NDArray[np.int8]
    ) -> NDArray[np.float64]:
        """Held-out marginal log likelihood for each student under the fitted prior."""

        self._check_fitted()
        log_likelihood = self._pattern_log_likelihood(
            np.asarray(responses, dtype=float), self.slip_, self.guess_
        )
        log_posterior = log_likelihood + np.log(np.clip(self.pattern_prior_, 1e-12, 1))[None, :]
        return logsumexp(log_posterior, axis=1)

    def map_patterns(self, responses: NDArray[np.int8]) -> NDArray[np.int8]:
        posterior = self.posterior_patterns(responses)
        return self.patterns_[posterior.argmax(axis=1)]

    def sample_patterns(self, n_students: int, seed: int) -> NDArray[np.int8]:
        self._check_fitted()
        rng = np.random.default_rng(seed)
        indices = rng.choice(self.patterns_.shape[0], size=n_students, p=self.pattern_prior_)
        return self.patterns_[indices]

    def sample(
        self, n_students: int, seed: int, alpha: NDArray[np.int8] | None = None
    ) -> NDArray[np.int8]:
        self._check_fitted()
        if alpha is None:
            alpha = self.sample_patterns(n_students, seed)
        alpha = np.asarray(alpha, dtype=np.int8)
        if alpha.shape != (n_students, self.q_matrix_.shape[1]):
            raise ValueError("alpha has incompatible dimensions")
        eta = ideal_dina_responses(alpha, self.q_matrix_)
        p_correct = np.where(eta == 1, 1 - self.slip_[None, :], self.guess_[None, :])
        rng = np.random.default_rng(seed + 1)
        return rng.binomial(1, p_correct).astype(np.int8)

    def classification_confidence(
        self, responses: NDArray[np.int8]
    ) -> dict[str, object]:
        """Marginal posterior classification confidence, not external criterion validity."""

        posterior = self.posterior_patterns(responses)
        mastery_probability = posterior @ self.patterns_
        confidence = np.maximum(mastery_probability, 1 - mastery_probability).mean(axis=0)
        return {
            "attribute_classification_confidence": confidence.tolist(),
            "mean_classification_confidence": float(confidence.mean()),
        }

    def srmsr(self, responses: NDArray[np.int8]) -> float:
        """SRMSR for observed and DINA-predicted item correlations."""

        responses = np.asarray(responses, dtype=float)
        eta = self.eta_.astype(bool)
        p_correct = np.where(eta, 1 - self.slip_[None, :], self.guess_[None, :])
        predicted_p = self.pattern_prior_ @ p_correct
        predicted_p11 = np.einsum(
            "p,pi,pj->ij", self.pattern_prior_, p_correct, p_correct, optimize=True
        )
        observed_p = responses.mean(axis=0)
        observed_p11 = np.einsum("ni,nj->nij", responses, responses).mean(axis=0)
        upper = np.triu_indices(responses.shape[1], k=1)
        observed_correlation = _pair_correlation(observed_p, observed_p11)[upper]
        predicted_correlation = _pair_correlation(predicted_p, predicted_p11)[upper]
        residual = observed_correlation - predicted_correlation
        if not np.isfinite(residual).all():
            raise ValueError("SRMSR is undefined for zero-variance items")
        return float(np.sqrt(np.mean(residual**2)))

    def readback_ceiling(
        self, alpha: NDArray[np.int8], seed: int = 42, repeats: int = 1
    ) -> dict[str, object]:
        """Repeatedly generate and diagnose assigned patterns under fitted human randomness."""

        if repeats < 1:
            raise ValueError("repeats must be positive")
        agreements = []
        pattern_agreements = []
        for repeat in range(repeats):
            responses = self.sample(alpha.shape[0], seed=seed + 2 * repeat, alpha=alpha)
            estimated = self.map_patterns(responses)
            agreements.append((alpha == estimated).mean(axis=0))
            pattern_agreements.append(np.all(alpha == estimated, axis=1).mean())
        attribute_agreement = np.asarray(agreements).mean(axis=0)
        return {
            "attribute_readback_agreement": float(attribute_agreement.mean()),
            "pattern_readback_agreement": float(np.mean(pattern_agreements)),
            "attribute_readback_by_attribute": attribute_agreement.tolist(),
            "n_repeats": repeats,
        }

    def _check_fitted(self) -> None:
        if not hasattr(self, "pattern_prior_"):
            raise RuntimeError("fit the baseline before sampling")
