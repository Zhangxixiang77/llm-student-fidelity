"""Same-split DINA/G-DINA comparisons for the human-data fit gate."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from student_fidelity.baselines import DINABaseline, GDINABaseline
from student_fidelity.data import ResponseDataset


def compare_cdm_fits(
    dina: DINABaseline,
    gdina: GDINABaseline,
    fitting: ResponseDataset,
    holdout: ResponseDataset,
    n_boot: int = 500,
    seed: int = 42,
) -> dict[str, Any]:
    """Compare held-out likelihood with a paired student bootstrap.

    Positive differences favor G-DINA. One split remains a pilot diagnostic,
    not a substitute for repeated splits, item-pair tests, or external accuracy.
    """

    if n_boot < 1:
        raise ValueError("n_boot must be positive")
    dina_holdout = dina.log_likelihood_by_student(holdout.responses)
    gdina_holdout = gdina.log_likelihood_by_student(holdout.responses)
    difference = gdina_holdout - dina_holdout
    lower, upper = _paired_bootstrap_ci(difference, n_boot=n_boot, seed=seed)
    return {
        "dina": _fit_statistics(
            dina,
            fitting.responses,
            holdout.responses,
            dina_holdout,
            nominal_n_parameters=2 * fitting.n_items + dina.patterns_.shape[0] - 1,
        ),
        "gdina": _fit_statistics(
            gdina,
            fitting.responses,
            holdout.responses,
            gdina_holdout,
            nominal_n_parameters=gdina.nominal_n_parameters_,
            convergence={
                "converged": gdina.converged_,
                "n_iter": gdina.n_iter_,
                "max_iter": gdina.max_iter,
                "mean_student_log_likelihood_tolerance": gdina.tolerance,
                "ridge": gdina.ridge,
            },
        ),
        "gdina_minus_dina_mean_holdout_log_likelihood": float(difference.mean()),
        "paired_student_bootstrap_95_ci": [lower, upper],
        "n_bootstrap": n_boot,
        "bootstrap_seed": seed,
        "provisional_dimension_b_model": "gdina" if lower > 0 else "dina",
        "interpretation": (
            "The provisional choice follows only the same-split held-out likelihood rule. "
            "It does not certify gate 1: SRMSR, multiple-comparison item-pair tests, "
            "external classification accuracy, convergence, and repeated splits remain separate."
        ),
    }


def _fit_statistics(
    model: DINABaseline | GDINABaseline,
    fitting_responses: NDArray[np.int8],
    holdout_responses: NDArray[np.int8],
    holdout_log_likelihood: NDArray[np.float64],
    nominal_n_parameters: int,
    convergence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result = {
        "training_log_likelihood": float(
            model.log_likelihood_by_student(fitting_responses).sum()
        ),
        "holdout_log_likelihood": float(holdout_log_likelihood.sum()),
        "mean_holdout_log_likelihood": float(holdout_log_likelihood.mean()),
        "holdout_srmsr": model.srmsr(holdout_responses),
        "nominal_n_parameters": nominal_n_parameters,
        **model.classification_confidence(holdout_responses),
    }
    if convergence is not None:
        result["convergence"] = convergence
    return result


def _paired_bootstrap_ci(
    difference: NDArray[np.float64], n_boot: int, seed: int
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for index in range(n_boot):
        rows = rng.integers(0, difference.shape[0], size=difference.shape[0])
        means[index] = difference[rows].mean()
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))
