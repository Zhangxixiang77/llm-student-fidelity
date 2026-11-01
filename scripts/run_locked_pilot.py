#!/usr/bin/env python3
"""Validate or execute the locked four-cell pilot."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from student_fidelity.pilot_runner import run_locked_pilot


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/pilot_frcsub_gpt4omini_v1.yaml"
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="perform paid preflight and student requests; omission is a dry run",
    )
    args = parser.parse_args()
    plan = run_locked_pilot(args.config, execute=args.execute)
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    if not args.execute:
        print("Dry run only: no API key, dataset, or model request was used.")


if __name__ == "__main__":
    main()
