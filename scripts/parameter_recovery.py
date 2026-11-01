"""Run resumable DINA/G-DINA parameter-recovery simulations on prepared FrcSub data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from student_fidelity.data import ResponseDataset
from student_fidelity.parameter_recovery import (
    dina_item_recovery,
    fit_generating_model,
    recovery_replication,
    summarize_recovery,
)


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _read_checkpoints(directory: Path) -> list[dict]:
    if not directory.exists():
        return []
    return [json.loads(path.read_text(encoding="utf-8")) for path in sorted(directory.glob("rep_*.json"))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--models", nargs="+", choices=["dina", "gdina"], default=["dina", "gdina"])
    parser.add_argument("--sample-sizes", nargs="+", type=int, default=[268, 536])
    parser.add_argument("--replications", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261031)
    parser.add_argument("--dina-starts", type=int, default=10)
    parser.add_argument("--generating-dina-starts", type=int, default=30)
    parser.add_argument("--dina-max-iter", type=int, default=300)
    parser.add_argument("--gdina-max-iter", type=int, default=300)
    parser.add_argument("--itemfit-simulations", type=int, default=100_000)
    parser.add_argument("--skip-itemfit", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.replications < 1:
        parser.error("--replications must be positive")

    dataset = ResponseDataset.from_directory(args.data_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    design = {
        "data_shape": [dataset.n_students, dataset.n_items],
        "models": args.models,
        "sample_sizes": args.sample_sizes,
        "replications": args.replications,
        "seed": args.seed,
        "dina_starts": args.dina_starts,
        "generating_dina_starts": args.generating_dina_starts,
        "dina_max_iter": args.dina_max_iter,
        "gdina_max_iter": args.gdina_max_iter,
        "itemfit_simulations": None if args.skip_itemfit else args.itemfit_simulations,
        "correctly_specified_parametric_bootstrap": True,
        "note": "Simulation validates estimation and test calibration, not real-data model adequacy.",
    }
    design_path = args.output_dir / "design.json"
    if args.resume and design_path.exists():
        previous_design = json.loads(design_path.read_text(encoding="utf-8"))
        if previous_design != design:
            parser.error("resume design differs from the existing output directory")
    _write_json(design_path, design)
    all_summaries = []

    for model_index, model_name in enumerate(args.models):
        print(f"Fitting {model_name} generating model", flush=True)
        generating = fit_generating_model(
            dataset,
            model_name,
            dina_starts=args.generating_dina_starts,
            dina_max_iter=args.dina_max_iter,
            gdina_max_iter=args.gdina_max_iter,
            seed=args.seed + 1000 * model_index,
        )
        generating_record = {
            "model": model_name,
            "training_log_likelihood": generating.training_log_likelihood_,
            "converged": generating.converged_,
            "n_iter": generating.n_iter_,
            "pattern_prior": generating.pattern_prior_.tolist(),
        }
        if model_name == "dina":
            generating_record.update(
                {
                    "guess": generating.guess_.tolist(),
                    "slip": generating.slip_.tolist(),
                    "best_start": generating.best_start_,
                    "fit_history": generating.fit_history_,
                }
            )
        else:
            generating_record["p_correct"] = generating.p_correct_.tolist()
        _write_json(args.output_dir / f"generating_{model_name}.json", generating_record)

        for size_index, n_students in enumerate(args.sample_sizes):
            checkpoint_dir = args.output_dir / f"replications_{model_name}_N{n_students}"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            records = _read_checkpoints(checkpoint_dir) if args.resume else []
            if not args.resume:
                for old_checkpoint in checkpoint_dir.glob("rep_*.json"):
                    old_checkpoint.unlink()
            completed = {record["replication"] for record in records}
            for replication in range(1, args.replications + 1):
                if replication in completed:
                    continue
                seed = (
                    args.seed
                    + 1_000_000 * model_index
                    + 100_000 * size_index
                    + 1000 * replication
                )
                record = recovery_replication(
                    generating,
                    dataset,
                    model_name,
                    n_students,
                    replication,
                    seed=seed,
                    dina_starts=args.dina_starts,
                    dina_max_iter=args.dina_max_iter,
                    gdina_max_iter=args.gdina_max_iter,
                    itemfit_simulations=(
                        None if args.skip_itemfit else args.itemfit_simulations
                    ),
                )
                _write_json(checkpoint_dir / f"rep_{replication:04d}.json", record)
                records.append(record)
                print(
                    f"{model_name} N={n_students} replication={replication}/{args.replications} "
                    f"SRMSR={record['srmsr']:.4f} converged={record['converged']}",
                    flush=True,
                )
            records.sort(key=lambda record: record["replication"])
            summary = summarize_recovery(records)
            _write_json(args.output_dir / f"summary_{model_name}_N{n_students}.json", summary)
            all_summaries.append(summary)
            if model_name == "dina":
                pd.DataFrame(dina_item_recovery(records, generating, dataset.item_ids)).to_csv(
                    args.output_dir / f"item_recovery_dina_N{n_students}.csv", index=False
                )
    _write_json(args.output_dir / "summary.json", all_summaries)


if __name__ == "__main__":
    main()
