"""Compare full-sample FrcSub DINA estimates with Sorrel et al. (2023), Table 1.

The reference used GDINA's MMLE-EM on the same 536-by-20 frac20 data and Q matrix.
This is an external numerical cross-check, not a simulation or a redistribution of data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from student_fidelity.baselines import DINABaseline
from student_fidelity.data import ResponseDataset

PUBLISHED_GUESS = np.array(
    [0.030, 0.016, 0.000, 0.224, 0.301, 0.095, 0.025, 0.444, 0.288, 0.029,
     0.066, 0.127, 0.013, 0.062, 0.031, 0.109, 0.038, 0.119, 0.022, 0.013]
)
PUBLISHED_SLIP = np.array(
    [0.089, 0.041, 0.134, 0.110, 0.172, 0.044, 0.197, 0.182, 0.247, 0.214,
     0.082, 0.041, 0.335, 0.061, 0.105, 0.111, 0.138, 0.138, 0.240, 0.157]
)
PUBLISHED_AIC = 9394.606
PUBLISHED_BIC = 10658.430
REFERENCE = {
    "authors": "Sorrel, Escudero, Najera, Kreitchmann, and Vazquez-Lira",
    "year": 2023,
    "title": "Exploring Approaches for Estimating Parameters in Cognitive Diagnosis Models with Small Sample Sizes",
    "journal": "Psych, 5, 336-349",
    "table": "Table 1, MMLE-EM columns",
    "doi": "10.3390/psych5020023",
    "url": "https://www.mdpi.com/2624-8611/5/2/23",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dina-starts", type=int, default=30)
    parser.add_argument("--dina-max-iter", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20261031)
    args = parser.parse_args()

    dataset = ResponseDataset.from_directory(args.data_dir)
    if (dataset.n_students, dataset.n_items, dataset.n_attributes) != (536, 20, 8):
        parser.error("the published cross-check requires full FrcSub dimensions 536 x 20 x 8")
    model = DINABaseline(
        max_iter=args.dina_max_iter, n_starts=args.dina_starts, seed=args.seed
    ).fit(dataset)
    n_parameters = 2 * dataset.n_items + 2**dataset.n_attributes - 1
    aic = -2 * model.training_log_likelihood_ + 2 * n_parameters
    bic = -2 * model.training_log_likelihood_ + np.log(dataset.n_students) * n_parameters

    comparison = pd.DataFrame(
        {
            "item_id": dataset.item_ids,
            "python_guess": model.guess_,
            "published_guess": PUBLISHED_GUESS,
            "guess_difference": model.guess_ - PUBLISHED_GUESS,
            "python_slip": model.slip_,
            "published_slip": PUBLISHED_SLIP,
            "slip_difference": model.slip_ - PUBLISHED_SLIP,
        }
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(args.output_dir / "item_comparison.csv", index=False)
    summary = {
        "reference": REFERENCE,
        "n_students": dataset.n_students,
        "n_items": dataset.n_items,
        "n_attributes": dataset.n_attributes,
        "n_parameters_for_information_criteria": n_parameters,
        "dina_starts": args.dina_starts,
        "dina_max_iter": args.dina_max_iter,
        "best_start": model.best_start_,
        "n_converged_starts": sum(start["converged"] for start in model.fit_history_),
        "training_log_likelihood": model.training_log_likelihood_,
        "python_aic": float(aic),
        "published_aic": PUBLISHED_AIC,
        "aic_difference": float(aic - PUBLISHED_AIC),
        "python_bic": float(bic),
        "published_bic": PUBLISHED_BIC,
        "bic_difference": float(bic - PUBLISHED_BIC),
        "fit_history": model.fit_history_,
    }
    for name, estimate, published in (
        ("guess", model.guess_, PUBLISHED_GUESS),
        ("slip", model.slip_, PUBLISHED_SLIP),
    ):
        difference = estimate - published
        summary[f"{name}_mae_from_rounded_table"] = float(np.abs(difference).mean())
        summary[f"{name}_rmse_from_rounded_table"] = float(np.sqrt(np.mean(difference**2)))
        summary[f"{name}_max_absolute_difference"] = float(np.abs(difference).max())
        summary[f"{name}_correlation"] = float(np.corrcoef(estimate, published)[0, 1])
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "fit_history"}, indent=2))


if __name__ == "__main__":
    main()
