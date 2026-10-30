"""Command-line entry points for offline demos and configured pilot cells."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from student_fidelity.data import ResponseDataset, make_synthetic_cdm_dataset
from student_fidelity.experiment import execute_pilot


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="student-fidelity", description="Cognitive structural-fidelity pilot toolkit"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser("demo", help="run an entirely offline synthetic-data smoke test")
    demo.add_argument("--output-dir", default="runs/demo")
    demo.add_argument("--students", type=int, default=40, help="virtual students per condition")
    demo.add_argument("--noise-repeats", type=int, default=10)
    demo.add_argument("--seed", type=int, default=42)

    pilot = subparsers.add_parser("pilot", help="run one configured experimental cell")
    pilot.add_argument("--config", required=True, help="YAML pilot configuration")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "demo":
        dataset = make_synthetic_cdm_dataset(
            n_students=max(2 * args.students, 40), n_items=20, n_attributes=8, seed=args.seed
        )
        summary = execute_pilot(
            dataset,
            args.output_dir,
            simulator_config={"type": "dina_mock"},
            n_students=args.students,
            noise_repeats=args.noise_repeats,
            seed=args.seed,
        )
    else:
        config = _load_config(args.config)
        dataset_path = Path(config["dataset"]["path"])
        dataset = ResponseDataset.from_directory(dataset_path)
        summary = execute_pilot(
            dataset=dataset,
            output_dir=config.get("output_dir", "runs/pilot"),
            simulator_config=config.get("simulator", {"type": "dina_mock"}),
            baseline_names=config.get("baselines", ["independent", "copula", "irt", "dina"]),
            persona_kind=config.get("persona", "attribute"),
            n_students=config.get("n_students"),
            state=config.get("state", "stateless"),
            temperature=float(config.get("temperature", 0.0)),
            noise_repeats=int(config.get("noise_repeats", 200)),
            seed=int(config.get("seed", 42)),
            resume=bool(config.get("resume", False)),
        )
    output_dir = (
        args.output_dir if args.command == "demo" else config.get("output_dir", "runs/pilot")
    )
    print(f"Pilot complete: {summary['design']}")
    print(f"Metrics: {Path(output_dir) / 'metrics.csv'}")


def _load_config(path: str) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as source:
        config = yaml.safe_load(source)
    if not isinstance(config, dict) or "dataset" not in config:
        raise ValueError("configuration must be a mapping with a dataset section")
    return config


if __name__ == "__main__":
    main()
