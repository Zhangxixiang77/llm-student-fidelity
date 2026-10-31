"""Run repeated human-data split diagnostics without making model calls."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from student_fidelity.data import ResponseDataset
from student_fidelity.repeated_splits import repeated_split_diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--splits", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--comparison-bootstraps", type=int, default=200)
    parser.add_argument("--itemfit-simulations", type=int, default=100_000)
    parser.add_argument("--gdina-max-iter", type=int, default=300)
    parser.add_argument("--focus-item-id", default="item_008")
    parser.add_argument("--focus-pair", nargs=2, default=["item_004", "item_008"])
    args = parser.parse_args()

    dataset = ResponseDataset.from_directory(args.dataset)
    result = repeated_split_diagnostics(
        dataset,
        n_splits=args.splits,
        seed=args.seed,
        comparison_bootstraps=args.comparison_bootstraps,
        itemfit_simulations=args.itemfit_simulations,
        gdina_max_iter=args.gdina_max_iter,
        focus_item_id=args.focus_item_id,
        focus_pair=tuple(args.focus_pair),
    )
    result["created_at_utc"] = datetime.now(UTC).isoformat()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Repeated-split diagnostics written to {args.output}")


if __name__ == "__main__":
    main()
