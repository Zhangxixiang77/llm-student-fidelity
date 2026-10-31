"""Logit-link G-DINA EM baseline for pilot model-comparison diagnostics."""

from __future__ import annotations

from itertools import combinations
from typing import Self

import numpy as np
from numpy.typing import NDArray
from scipy.special import expit, logsumexp

from student_fidelity.baselines.dina import DINABaseline, _pair_correlation
from student_fidelity.data import ResponseDataset
from student_fidelity.metrics import ideal_dina_responses


class GDINABaseline:
    """Saturated item response functions with a categorical mastery-pattern prior.

    This lightweight implementation uses weak ridge regularization for numerical
    stability. Formal analyses should cross-check parameters with R GDINA/CDM.
    """

    def __init__(
        self,
        max_iter: int = 100,
        tolerance: float = 1e-4,
        ridge: float = 1e-3,
        max_newton_iter: int = 25,
    ):
        if max_iter < 1 or max_newton_iter < 1 or tolerance <= 0 or ridge <= 0:
            raise ValueError("iterations, tolerance, and ridge must be positive")
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.ridge = ridge
        self.max_newton_iter = max_newton_iter

    def fit(self, dataset: ResponseDataset) -> Self:
        responses = dataset.responses.astype(float)
        self.q_matrix_ = dataset.q_matrix.copy()
        self.patterns_ = DINABaseline.enumerate_patterns(dataset.n_attributes)
        n_students = dataset.n_students
        n_patterns = self.patterns_.shape[0]
        self.designs_ = [
            _item_design(self.patterns_, required) for required in self.q_matrix_
        ]
        self.nominal_n_parameters_ = n_patterns - 1 + sum(
            design.shape[1] for design in self.designs_
        )
        pattern_prior = np.full(n_patterns, 1 / n_patterns)
        betas = []
        for design in self.designs_:
            beta = np.zeros(design.shape[1])
            beta[0] = _logit(0.2)
            beta[-1] = _logit(0.8) - beta[0]
            betas.append(beta)

        previous_log_likelihood = -np.inf
        converged = False
        posterior = np.full((n_students, n_patterns), 1 / n_patterns)
        for iteration in range(1, self.max_iter + 1):
            p_correct = self._probabilities(betas)
            log_likelihood = _pattern_log_likelihood(responses, p_correct)
            log_posterior = log_likelihood + np.log(
                np.clip(pattern_prior, 1e-12, 1)
            )[None, :]
            log_marginal = logsumexp(log_posterior, axis=1)
            current_log_likelihood = float(log_marginal.sum())
            posterior = np.exp(log_posterior - log_marginal[:, None])
            pattern_prior = np.clip(posterior.mean(axis=0), 1e-12, 1)
            pattern_prior /= pattern_prior.sum()

            for item, design in enumerate(self.designs_):
                positive_weight = posterior.T @ responses[:, item]
                total_weight = posterior.sum(axis=0)
                betas[item] = _weighted_logistic_newton(
                    design,
                    positive_weight,
                    total_weight,
                    betas[item],
                    ridge=self.ridge,
                    max_iter=self.max_newton_iter,
                )
            change = current_log_likelihood - previous_log_likelihood
            if iteration > 1 and abs(change) / n_students < self.tolerance:
                converged = True
                break
            previous_log_likelihood = current_log_likelihood

        self.betas_ = betas
        self.p_correct_ = self._probabilities(self.betas_)
        self.pattern_prior_ = pattern_prior
        final_log_likelihood = _pattern_log_likelihood(responses, self.p_correct_)
        final_log_posterior = final_log_likelihood + np.log(
            np.clip(self.pattern_prior_, 1e-12, 1)
        )[None, :]
        final_log_marginal = logsumexp(final_log_posterior, axis=1)
        self.posterior_ = np.exp(final_log_posterior - final_log_marginal[:, None])
        self.training_log_likelihood_ = float(final_log_marginal.sum())
        self.n_iter_ = iteration
        self.converged_ = converged
        return self

    def _probabilities(
        self, betas: list[NDArray[np.float64]]
    ) -> NDArray[np.float64]:
        return np.column_stack(
            [expit(design @ beta) for design, beta in zip(self.designs_, betas, strict=True)]
        )

    def _log_likelihood_by_student(
        self, responses: NDArray[np.int8]
    ) -> NDArray[np.float64]:
        responses = _validate_responses(responses, self.q_matrix_.shape[0])
        log_likelihood = _pattern_log_likelihood(responses, self.p_correct_)
        log_posterior = log_likelihood + np.log(
            np.clip(self.pattern_prior_, 1e-12, 1)
        )[None, :]
        return logsumexp(log_posterior, axis=1)

    def log_likelihood_by_student(
        self, responses: NDArray[np.int8]
    ) -> NDArray[np.float64]:
        self._check_fitted()
        return self._log_likelihood_by_student(responses)

    def posterior_patterns(self, responses: NDArray[np.int8]) -> NDArray[np.float64]:
        self._check_fitted()
        responses = _validate_responses(responses, self.q_matrix_.shape[0])
        log_likelihood = _pattern_log_likelihood(responses, self.p_correct_)
        log_posterior = log_likelihood + np.log(
            np.clip(self.pattern_prior_, 1e-12, 1)
        )[None, :]
        return np.exp(log_posterior - logsumexp(log_posterior, axis=1)[:, None])

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
        bit_weights = 1 << np.arange(alpha.shape[1])
        pattern_indices = (alpha.astype(np.int64) * bit_weights).sum(axis=1)
        p_correct = self.p_correct_[pattern_indices]
        rng = np.random.default_rng(seed + 1)
        return rng.binomial(1, p_correct).astype(np.int8)

    def response_residuals(
        self, responses: NDArray[np.int8], alpha: NDArray[np.int8]
    ) -> dict[str, object]:
        """Observed-minus-predicted correctness for assigned mastery patterns.

        Master shortfall is predicted-minus-observed correctness on DINA-ideal
        mastery items; nonmaster excess is observed-minus-predicted correctness
        on the remaining items. These are residuals, not raw DINA slip/guess.
        """

        self._check_fitted()
        responses = _validate_responses(responses, self.q_matrix_.shape[0])
        alpha = np.asarray(alpha, dtype=np.int8)
        if alpha.shape != (responses.shape[0], self.q_matrix_.shape[1]):
            raise ValueError("alpha has incompatible dimensions")
        bit_weights = 1 << np.arange(alpha.shape[1])
        pattern_indices = (alpha.astype(np.int64) * bit_weights).sum(axis=1)
        predicted = self.p_correct_[pattern_indices]
        observed = responses.astype(float)
        residual = observed - predicted
        masters = ideal_dina_responses(alpha, self.q_matrix_).astype(bool)
        master_shortfall = np.full(self.q_matrix_.shape[0], np.nan)
        nonmaster_excess = np.full(self.q_matrix_.shape[0], np.nan)
        for item in range(self.q_matrix_.shape[0]):
            if masters[:, item].any():
                master_shortfall[item] = -residual[masters[:, item], item].mean()
            if (~masters[:, item]).any():
                nonmaster_excess[item] = residual[~masters[:, item], item].mean()
        return {
            "gdina_item_predicted_p_correct": predicted.mean(axis=0).tolist(),
            "gdina_item_observed_p_correct": observed.mean(axis=0).tolist(),
            "gdina_item_mean_residual": residual.mean(axis=0).tolist(),
            "gdina_item_master_shortfall": master_shortfall.tolist(),
            "gdina_item_nonmaster_excess": nonmaster_excess.tolist(),
            "gdina_mean_residual": float(residual.mean()),
            "gdina_mean_master_shortfall": float(np.nanmean(master_shortfall)),
            "gdina_mean_nonmaster_excess": float(np.nanmean(nonmaster_excess)),
        }

    def classification_confidence(
        self, responses: NDArray[np.int8]
    ) -> dict[str, object]:
        """Marginal posterior classification confidence, not external accuracy."""

        posterior = self.posterior_patterns(responses)
        mastery_probability = posterior @ self.patterns_
        confidence = np.maximum(mastery_probability, 1 - mastery_probability).mean(axis=0)
        return {
            "attribute_classification_confidence": confidence.tolist(),
            "mean_classification_confidence": float(confidence.mean()),
        }

    def srmsr(self, responses: NDArray[np.int8]) -> float:
        """SRMSR for observed and G-DINA-predicted item correlations."""

        responses = _validate_responses(responses, self.q_matrix_.shape[0]).astype(float)
        predicted_p = self.pattern_prior_ @ self.p_correct_
        predicted_p11 = np.einsum(
            "p,pi,pj->ij", self.pattern_prior_, self.p_correct_, self.p_correct_, optimize=True
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
        """Repeatedly generate and diagnose assigned patterns under fitted randomness."""

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
            raise RuntimeError("fit the baseline before evaluation or sampling")


def _item_design(
    patterns: NDArray[np.int8], required: NDArray[np.int8]
) -> NDArray[np.float64]:
    required_indices = np.flatnonzero(required)
    mastery = patterns[:, required_indices].astype(float)
    columns = [np.ones(patterns.shape[0])]
    for order in range(1, len(required_indices) + 1):
        for subset in combinations(range(len(required_indices)), order):
            columns.append(mastery[:, subset].prod(axis=1))
    return np.column_stack(columns)


def _pattern_log_likelihood(
    responses: NDArray[np.float64], p_correct: NDArray[np.float64]
) -> NDArray[np.float64]:
    p_correct = np.clip(p_correct, 1e-8, 1 - 1e-8)
    return (
        responses[:, None, :] * np.log(p_correct)[None, :, :]
        + (1 - responses[:, None, :]) * np.log1p(-p_correct)[None, :, :]
    ).sum(axis=2)


def _weighted_logistic_newton(
    design: NDArray[np.float64],
    positive_weight: NDArray[np.float64],
    total_weight: NDArray[np.float64],
    initial_beta: NDArray[np.float64],
    ridge: float,
    max_iter: int,
) -> NDArray[np.float64]:
    beta = initial_beta.copy()
    penalty = ridge * np.eye(beta.shape[0])
    for _ in range(max_iter):
        eta = design @ beta
        probability = expit(eta)
        gradient = design.T @ (positive_weight - total_weight * probability) - ridge * beta
        variance = total_weight * probability * (1 - probability)
        information = (design.T * variance) @ design + penalty
        try:
            step = np.linalg.solve(information, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(information, gradient, rcond=None)[0]
        if np.max(np.abs(step)) < 1e-7:
            break
        objective = _penalized_logistic_objective(
            design, positive_weight, total_weight, beta, ridge
        )
        scale = 1.0
        for _ in range(20):
            candidate = beta + scale * step
            candidate_objective = _penalized_logistic_objective(
                design, positive_weight, total_weight, candidate, ridge
            )
            if candidate_objective >= objective - 1e-10:
                beta = candidate
                break
            scale *= 0.5
        else:
            break
        if scale * np.max(np.abs(step)) < 1e-6:
            break
    return beta


def _penalized_logistic_objective(
    design: NDArray[np.float64],
    positive_weight: NDArray[np.float64],
    total_weight: NDArray[np.float64],
    beta: NDArray[np.float64],
    ridge: float,
) -> float:
    eta = design @ beta
    return float(
        np.sum(positive_weight * eta - total_weight * np.logaddexp(0, eta))
        - 0.5 * ridge * (beta @ beta)
    )


def _validate_responses(
    responses: NDArray[np.int8], n_items: int
) -> NDArray[np.int8]:
    responses = np.asarray(responses, dtype=np.int8)
    if responses.ndim != 2 or responses.shape[1] != n_items:
        raise ValueError("responses have incompatible dimensions")
    if not np.isin(responses, [0, 1]).all():
        raise ValueError("only complete binary responses are supported")
    return responses


def _logit(probability: float) -> float:
    return float(np.log(probability / (1 - probability)))
