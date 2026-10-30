"""Structural-fidelity and controllability metrics."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import wasserstein_distance

from student_fidelity.data import ResponseDataset
from student_fidelity.tetrachoric import tetrachoric_matrix


def _validate_pair(real: NDArray[np.int_], generated: NDArray[np.int_]) -> None:
    if real.ndim != 2 or generated.ndim != 2 or real.shape[1] != generated.shape[1]:
        raise ValueError("real and generated responses must have the same number of items")


def item_marginal_metrics(real: NDArray[np.int_], generated: NDArray[np.int_]) -> dict[str, float]:
    _validate_pair(real, generated)
    real_p = real.mean(axis=0)
    generated_p = generated.mean(axis=0)
    if real_p.std() == 0 or generated_p.std() == 0:
        correlation = float("nan")
    else:
        correlation = float(np.corrcoef(real_p, generated_p)[0, 1])
    return {
        "item_pass_rate_pearson": correlation,
        "item_pass_rate_rmse": float(np.sqrt(np.mean((real_p - generated_p) ** 2))),
    }


def dependency_rmse(real: NDArray[np.int_], generated: NDArray[np.int_]) -> float:
    """RQ1 primary metric: off-diagonal tetrachoric-matrix RMSE."""

    _validate_pair(real, generated)
    real_corr = tetrachoric_matrix(real)
    generated_corr = tetrachoric_matrix(generated)
    upper = np.triu_indices(real.shape[1], k=1)
    return float(np.sqrt(np.mean((real_corr[upper] - generated_corr[upper]) ** 2)))


def full_pattern_total_variation(
    real: NDArray[np.int_], generated: NDArray[np.int_]
) -> float:
    """Descriptive total-variation distance between observed full-response patterns."""

    _validate_pair(real, generated)
    real_patterns, real_counts = np.unique(real, axis=0, return_counts=True)
    generated_patterns, generated_counts = np.unique(generated, axis=0, return_counts=True)
    real_lookup = {
        row.tobytes(): count / real.shape[0]
        for row, count in zip(real_patterns, real_counts, strict=True)
    }
    generated_lookup = {
        row.tobytes(): count / generated.shape[0]
        for row, count in zip(generated_patterns, generated_counts, strict=True)
    }
    keys = set(real_lookup) | set(generated_lookup)
    distance = sum(
        abs(real_lookup.get(key, 0.0) - generated_lookup.get(key, 0.0)) for key in keys
    )
    return float(0.5 * distance)


def evaluate_structure(real: NDArray[np.int_], generated: NDArray[np.int_]) -> dict[str, float]:
    """Return A1, A2, and descriptive A4 metrics against held-out real students."""

    result = item_marginal_metrics(real, generated)
    result["tetrachoric_matrix_rmse"] = dependency_rmse(real, generated)
    result["total_score_wasserstein"] = float(
        wasserstein_distance(real.sum(axis=1), generated.sum(axis=1))
    )
    result["full_pattern_total_variation"] = full_pattern_total_variation(real, generated)
    return result


def split_half_noise(
    dataset: ResponseDataset, n_repeats: int = 200, seed: int = 42
) -> NDArray[np.float64]:
    """Real-data split-half distances used to derive the preregistered noise bound Delta."""

    if n_repeats < 1 or dataset.n_students < 4:
        raise ValueError("noise estimation requires repeats and at least four students")
    rng = np.random.default_rng(seed)
    distances = []
    for _ in range(n_repeats):
        order = rng.permutation(dataset.n_students)
        n_half = dataset.n_students // 2
        left = dataset.responses[order[:n_half]]
        right = dataset.responses[order[n_half : 2 * n_half]]
        distances.append(dependency_rmse(left, right))
    return np.asarray(distances)


def bootstrap_dependency_ci(
    real: NDArray[np.int_],
    generated: NDArray[np.int_],
    n_boot: int = 200,
    confidence: float = 0.90,
    seed: int = 42,
) -> tuple[float, float]:
    """Student-resampling CI for a generated condition; the held-out reference stays fixed."""

    rng = np.random.default_rng(seed)
    estimates = np.empty(n_boot)
    for index in range(n_boot):
        rows = rng.integers(0, generated.shape[0], size=generated.shape[0])
        estimates[index] = dependency_rmse(real, generated[rows])
    alpha = (1 - confidence) / 2
    return float(np.quantile(estimates, alpha)), float(np.quantile(estimates, 1 - alpha))


def ideal_dina_responses(alpha: NDArray[np.int_], q_matrix: NDArray[np.int_]) -> NDArray[np.int_]:
    alpha = np.asarray(alpha, dtype=np.int8)
    q_matrix = np.asarray(q_matrix, dtype=np.int8)
    if alpha.ndim != 2 or q_matrix.ndim != 2 or alpha.shape[1] != q_matrix.shape[1]:
        raise ValueError("alpha and Q matrix dimensions are incompatible")
    return np.all(alpha[:, None, :] >= q_matrix[None, :, :], axis=2).astype(np.int8)


def readback_agreement(
    target_alpha: NDArray[np.int_], estimated_alpha: NDArray[np.int_]
) -> dict[str, float]:
    target_alpha = np.asarray(target_alpha)
    estimated_alpha = np.asarray(estimated_alpha)
    if target_alpha.shape != estimated_alpha.shape or target_alpha.ndim != 2:
        raise ValueError("target and estimated mastery patterns must have matching 2-D shapes")
    return {
        "attribute_readback_agreement": float((target_alpha == estimated_alpha).mean()),
        "pattern_readback_agreement": float(
            np.all(target_alpha == estimated_alpha, axis=1).mean()
        ),
    }


def implicit_slip_guess(
    responses: NDArray[np.int_], alpha: NDArray[np.int_], q_matrix: NDArray[np.int_]
) -> dict[str, object]:
    """Dimension-B slip/guess conditional on the assigned mastery pattern."""

    responses = np.asarray(responses)
    if responses.shape[0] != alpha.shape[0] or responses.shape[1] != q_matrix.shape[0]:
        raise ValueError("responses, alpha, and Q matrix are not aligned")
    eta = ideal_dina_responses(alpha, q_matrix).astype(bool)
    slip = np.zeros(q_matrix.shape[0])
    guess = np.zeros(q_matrix.shape[0])
    for item in range(q_matrix.shape[0]):
        masters = eta[:, item]
        nonmasters = ~masters
        slip[item] = 1 - responses[masters, item].mean() if masters.any() else np.nan
        guess[item] = responses[nonmasters, item].mean() if nonmasters.any() else np.nan
    return {
        "item_slip": slip.tolist(),
        "item_guess": guess.tolist(),
        "mean_slip": float(np.nanmean(slip)),
        "mean_guess": float(np.nanmean(guess)),
        "ability_leakage_rate": float(np.nanmean(guess)),
    }
