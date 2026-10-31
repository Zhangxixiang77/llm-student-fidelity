"""Audit a completed local pilot without making model calls."""

from __future__ import annotations

import argparse
from pathlib import Path

from student_fidelity.audit import write_audit
from student_fidelity.data import ResponseDataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--ceiling-repeats", type=int, default=100)
    args = parser.parse_args()
    dataset = ResponseDataset.from_directory(args.dataset)
    output = write_audit(dataset, args.run_dir, ceiling_repeats=args.ceiling_repeats)
    print(f"Audit written to {output}")


if __name__ == "__main__":
    main()
