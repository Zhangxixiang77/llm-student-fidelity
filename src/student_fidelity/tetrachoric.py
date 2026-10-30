"""Smoothed pairwise maximum-likelihood tetrachoric correlations."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize_scalar
from scipy.stats import multivariate_normal, norm


def tetrachoric_correlation(x: NDArray[np.int_], y: NDArray[np.int_]) -> float:
    """Estimate one tetrachoric correlation using a 0.5 continuity correction."""

    x = np.asarray(x, dtype=int)
    y = np.asarray(y, dtype=int)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("x and y must be aligned one-dimensional vectors")
    if not np.isin(x, [0, 1]).all() or not np.isin(y, [0, 1]).all():
        raise ValueError("tetrachoric correlation requires binary inputs")

    a = np.sum((x == 0) & (y == 0)) + 0.5
    b = np.sum((x == 0) & (y == 1)) + 0.5
    c = np.sum((x == 1) & (y == 0)) + 0.5
    d = np.sum((x == 1) & (y == 1)) + 0.5
    total = a + b + c + d
    zx = norm.ppf((a + b) / total)
    zy = norm.ppf((a + c) / total)
    p_x0 = float(norm.cdf(zx))
    p_y0 = float(norm.cdf(zy))

    def negative_log_likelihood(rho: float) -> float:
        covariance = [[1.0, rho], [rho, 1.0]]
        p00 = float(
            multivariate_normal.cdf(
                [zx, zy],
                mean=[0.0, 0.0],
                cov=covariance,
                maxpts=2_000,
                abseps=1e-6,
                releps=1e-6,
                rng=0,
            )
        )
        probabilities = [p00, p_x0 - p00, p_y0 - p00, 1 - p_x0 - p_y0 + p00]
        return -sum(
            count * np.log(max(probability, 1e-12))
            for count, probability in zip([a, b, c, d], probabilities, strict=True)
        )

    result = minimize_scalar(
        negative_log_likelihood, bounds=(-0.95, 0.95), method="bounded", options={"xatol": 1e-4}
    )
    return float(np.clip(result.x, -0.95, 0.95))


def tetrachoric_matrix(responses: NDArray[np.int_]) -> NDArray[np.float64]:
    responses = np.asarray(responses, dtype=int)
    if responses.ndim != 2 or not np.isin(responses, [0, 1]).all():
        raise ValueError("responses must be a two-dimensional binary matrix")
    n_items = responses.shape[1]
    matrix = np.eye(n_items, dtype=float)
    for left in range(n_items):
        for right in range(left + 1, n_items):
            value = tetrachoric_correlation(responses[:, left], responses[:, right])
            matrix[left, right] = matrix[right, left] = value
    return matrix


def nearest_psd_correlation(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    """Project a pairwise correlation estimate to a positive-semidefinite correlation matrix."""

    matrix = np.asarray(matrix, dtype=float)
    symmetric = (matrix + matrix.T) / 2
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    psd = (eigenvectors * np.maximum(eigenvalues, 1e-6)) @ eigenvectors.T
    scale = np.sqrt(np.diag(psd))
    psd = psd / scale[:, None] / scale[None, :]
    np.fill_diagonal(psd, 1.0)
    return psd
