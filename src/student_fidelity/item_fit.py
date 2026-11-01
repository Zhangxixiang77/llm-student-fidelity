"""Posterior-simulation item-pair fit diagnostics for held-out binary responses.

The statistics follow GDINA::itemfit (Chen, de la Torre, & Zhang, 2013):
Fisher-z correlation residuals and log-odds-ratio residuals, with Holm
adjustment at test and item level. This is an independent lightweight
implementation; simulation-based calibration and optional external cross-checks
are documented in docs/estimator_validation.md.
"""

from __future__ import annotations

import math
from itertools import combinations
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

from student_fidelity.baselines import DINABaseline, GDINABaseline
from student_fidelity.metrics import ideal_dina_responses


def item_pair_fit(
    model: DINABaseline | GDINABaseline,
    responses: NDArray[np.int8],
    item_ids: list[str] | tuple[str, ...] | None = None,
    n_simulated_students: int | None = None,
    seed: int = 123456,
    small_expected_cell_threshold: float = 5.0,
) -> dict[str, Any]:
    """Compare observed item pairs with posterior-simulated model expectations.

    Simulation draws mastery patterns from the mean item-level posterior over
    students, equivalent to sampling the flattened posterior matrix used by
    GDINA::itemfit(person.sim="post"). The observed responses are never used
    to refit the model. Normal-approximation p-values are flagged when any
    model-expected 2x2 cell count, scaled to the observed sample size, is small.
    """

    responses = np.asarray(responses, dtype=np.int8)
    if responses.ndim != 2 or responses.shape[1] < 2 or responses.shape[0] < 4:
        raise ValueError("item-pair fit requires at least four students and two items")
    if not np.isin(responses, [0, 1]).all():
        raise ValueError("only complete binary responses are supported")
    n_students, n_items = responses.shape
    if np.any((responses.mean(axis=0) == 0) | (responses.mean(axis=0) == 1)):
        raise ValueError("item-pair correlations are undefined for zero-variance items")
    if item_ids is None:
        item_ids = [f"item_{index + 1:03d}" for index in range(n_items)]
    if len(item_ids) != n_items or len(set(item_ids)) != n_items:
        raise ValueError("item_ids must be unique and match the response columns")
    if n_simulated_students is None:
        n_simulated_students = max(100_000, 10 * n_students)
    if n_simulated_students < 1:
        raise ValueError("n_simulated_students must be positive")
    if small_expected_cell_threshold <= 0:
        raise ValueError("small_expected_cell_threshold must be positive")

    posterior = model.posterior_patterns(responses)
    pattern_probabilities = _pattern_probabilities(model)
    rng = np.random.default_rng(seed)
    pattern_indices = rng.choice(
        pattern_probabilities.shape[0],
        size=n_simulated_students,
        p=posterior.mean(axis=0),
    )
    simulated = rng.binomial(1, pattern_probabilities[pattern_indices]).astype(np.int8)
    if np.any((simulated.mean(axis=0) == 0) | (simulated.mean(axis=0) == 1)):
        raise ValueError("posterior simulation produced a zero-variance item")

    observed_correlation = np.corrcoef(responses.astype(float), rowvar=False)
    expected_correlation = np.corrcoef(simulated.astype(float), rowvar=False)
    pairs = []
    for item_1, item_2 in combinations(range(n_items), 2):
        observed_cells = _pair_cells(responses[:, item_1], responses[:, item_2])
        expected_cells = _pair_cells(simulated[:, item_1], simulated[:, item_2])
        corrected_observed, observed_corrections = _continuity_correction(observed_cells)
        corrected_expected, expected_corrections = _continuity_correction(expected_cells)
        observed_log_or = _log_odds_ratio(corrected_observed)
        expected_log_or = _log_odds_ratio(corrected_expected)
        replication = math.ceil(n_simulated_students / n_students)
        log_or_se = math.sqrt(
            float((1 / corrected_expected).sum()) * replication
        )
        log_or_difference = observed_log_or - expected_log_or
        log_or_z = abs(log_or_difference) / log_or_se

        observed_z = _fisher_z(observed_correlation[item_1, item_2])
        expected_z = _fisher_z(expected_correlation[item_1, item_2])
        correlation_se = math.sqrt(1 / (n_students - 3))
        correlation_difference = observed_z - expected_z
        correlation_z = abs(correlation_difference) / correlation_se
        expected_at_observed_n = expected_cells * (n_students / n_simulated_students)
        small_cells = np.flatnonzero(
            expected_at_observed_n < small_expected_cell_threshold
        ).tolist()
        pairs.append(
            {
                "item_1_index": item_1 + 1,
                "item_2_index": item_2 + 1,
                "item_1": item_ids[item_1],
                "item_2": item_ids[item_2],
                "observed_cells_n11_n10_n01_n00": observed_cells.astype(int).tolist(),
                "expected_cells_at_observed_n_n11_n10_n01_n00": (
                    expected_at_observed_n.tolist()
                ),
                "n_small_expected_cells": len(small_cells),
                "small_expected_cell_indices": small_cells,
                "observed_zero_cells_corrected": observed_corrections,
                "expected_zero_cells_corrected": expected_corrections,
                "observed_correlation": float(observed_correlation[item_1, item_2]),
                "expected_correlation": float(expected_correlation[item_1, item_2]),
                "correlation_fisher_z_difference": float(correlation_difference),
                "correlation_se": correlation_se,
                "correlation_z": float(correlation_z),
                "correlation_p": _two_sided_normal_p(correlation_z),
                "observed_log_odds_ratio": observed_log_or,
                "expected_log_odds_ratio": expected_log_or,
                "log_odds_ratio_difference": log_or_difference,
                "log_odds_ratio_se": log_or_se,
                "log_odds_ratio_z": log_or_z,
                "log_odds_ratio_p": _two_sided_normal_p(log_or_z),
            }
        )

    for statistic in ("correlation", "log_odds_ratio"):
        p_values = np.array([pair[f"{statistic}_p"] for pair in pairs])
        adjusted = holm_adjust(p_values)
        for pair, value in zip(pairs, adjusted, strict=True):
            pair[f"{statistic}_holm_test_p"] = value

    item_level = []
    for item in range(n_items):
        result: dict[str, Any] = {"item_index": item + 1, "item": item_ids[item]}
        for statistic in ("correlation", "log_odds_ratio"):
            locations = [
                index
                for index, pair in enumerate(pairs)
                if item + 1 in (pair["item_1_index"], pair["item_2_index"])
            ]
            p_values = np.array([pairs[index][f"{statistic}_p"] for index in locations])
            adjusted = holm_adjust(p_values)
            z_values = np.array([pairs[index][f"{statistic}_z"] for index in locations])
            worst = int(np.argmax(z_values))
            worst_pair = pairs[locations[worst]]
            result[f"{statistic}_max_z"] = float(z_values[worst])
            result[f"{statistic}_unadjusted_p"] = float(p_values[worst])
            result[f"{statistic}_holm_item_p"] = float(adjusted[worst])
            result[f"{statistic}_worst_partner"] = (
                worst_pair["item_2"]
                if worst_pair["item_1_index"] == item + 1
                else worst_pair["item_1"]
            )
        item_level.append(result)

    return {
        "method": "GDINA-style posterior-simulation item-pair fit",
        "person_simulation": "posterior",
        "n_observed_students": n_students,
        "n_simulated_students": n_simulated_students,
        "seed": seed,
        "test_level_holm_family_size": len(pairs),
        "item_level_holm_family_size": n_items - 1,
        "zero_cell_correction": "Replace each zero 2x2 cell count by 0.5 for log-OR only",
        "small_expected_cell_threshold_at_observed_n": small_expected_cell_threshold,
        "n_pairs_with_small_expected_cells": int(
            sum(pair["n_small_expected_cells"] > 0 for pair in pairs)
        ),
        "n_pairs_with_zero_cell_correction": int(
            sum(
                bool(
                    pair["observed_zero_cells_corrected"]
                    or pair["expected_zero_cells_corrected"]
                )
                for pair in pairs
            )
        ),
        "n_test_significant_holm_0_05": {
            statistic: int(
                sum(pair[f"{statistic}_holm_test_p"] <= 0.05 for pair in pairs)
            )
            for statistic in ("correlation", "log_odds_ratio")
        },
        "n_item_significant_holm_0_05": {
            statistic: int(
                sum(item[f"{statistic}_holm_item_p"] <= 0.05 for item in item_level)
            )
            for statistic in ("correlation", "log_odds_ratio")
        },
        "item_pairs": pairs,
        "item_level": item_level,
        "interpretation": (
            "Test-level Holm uses all J(J-1)/2 pairs separately for each statistic; "
            "item-level Holm uses the J-1 partners of each item. Sparse-cell flags "
            "qualify the normal approximations and are not themselves misfit tests."
        ),
    }


def holm_adjust(p_values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Holm step-down adjusted p-values, preserving the original order."""

    p_values = np.asarray(p_values, dtype=float)
    if p_values.ndim != 1 or not np.isfinite(p_values).all():
        raise ValueError("p_values must be a finite one-dimensional array")
    if np.any((p_values < 0) | (p_values > 1)):
        raise ValueError("p_values must lie between zero and one")
    if p_values.size == 0:
        return p_values.copy()
    order = np.argsort(p_values, kind="stable")
    sorted_p = p_values[order]
    multipliers = np.arange(p_values.size, 0, -1)
    adjusted_sorted = np.maximum.accumulate(multipliers * sorted_p).clip(max=1.0)
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = adjusted_sorted
    return adjusted


def _pattern_probabilities(
    model: DINABaseline | GDINABaseline,
) -> NDArray[np.float64]:
    if isinstance(model, GDINABaseline):
        return model.p_correct_
    eta = ideal_dina_responses(model.patterns_, model.q_matrix_).astype(bool)
    return np.where(eta, 1 - model.slip_[None, :], model.guess_[None, :])


def _pair_cells(
    response_1: NDArray[np.int8], response_2: NDArray[np.int8]
) -> NDArray[np.float64]:
    response_1 = response_1.astype(bool)
    response_2 = response_2.astype(bool)
    return np.array(
        [
            np.sum(response_1 & response_2),
            np.sum(response_1 & ~response_2),
            np.sum(~response_1 & response_2),
            np.sum(~response_1 & ~response_2),
        ],
        dtype=float,
    )


def _continuity_correction(
    cells: NDArray[np.float64],
) -> tuple[NDArray[np.float64], list[int]]:
    corrected = cells.copy()
    locations = np.flatnonzero(corrected == 0)
    corrected[locations] = 0.5
    return corrected, locations.astype(int).tolist()


def _log_odds_ratio(cells: NDArray[np.float64]) -> float:
    n11, n10, n01, n00 = cells
    return float(np.log((n11 * n00) / (n10 * n01)))


def _fisher_z(correlation: float) -> float:
    return float(np.arctanh(np.clip(correlation, -1 + 1e-12, 1 - 1e-12)))


def _two_sided_normal_p(z_value: float) -> float:
    return float(2 * norm.sf(z_value))
