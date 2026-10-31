"""Run the literature-motivated item_008 exclusion sensitivity offline."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from student_fidelity.data import ResponseDataset
from student_fidelity.item_sensitivity import item_exclusion_sensitivity


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--exclude-items", nargs="+", default=["item_008"])
    parser.add_argument("--splits", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--itemfit-simulations", type=int, default=100_000)
    parser.add_argument("--gdina-max-iter", type=int, default=300)
    args = parser.parse_args()

    dataset = ResponseDataset.from_directory(args.dataset)
    result = item_exclusion_sensitivity(
        dataset,
        excluded_item_ids=args.exclude_items,
        n_splits=args.splits,
        seed=args.seed,
        itemfit_simulations=args.itemfit_simulations,
        gdina_max_iter=args.gdina_max_iter,
    )
    result["created_at_utc"] = datetime.now(UTC).isoformat()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Item-exclusion sensitivity written to {args.output}")


if __name__ == "__main__":
    main()
