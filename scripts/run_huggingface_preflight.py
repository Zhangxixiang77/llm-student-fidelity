#!/usr/bin/env python3
"""Resolve, download, or preflight local Hugging Face candidates explicitly."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from student_fidelity.huggingface_runner import (
    download_huggingface_models,
    load_huggingface_design,
    resolve_huggingface_metadata,
    run_huggingface_preflight,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/huggingface_candidates_v1.yaml")
    parser.add_argument("--metadata", help="resolved model metadata JSON")
    action = parser.add_mutually_exclusive_group()
    action.add_argument(
        "--resolve-metadata",
        action="store_true",
        help="query model metadata and immutable revisions; no weight download",
    )
    action.add_argument(
        "--download", action="store_true", help="download resolved model snapshots"
    )
    action.add_argument(
        "--execute",
        action="store_true",
        help="run local preflight only; never starts student generation",
    )
    args = parser.parse_args()
    config = load_huggingface_design(args.config)
    root = Path(config["output_dir"])
    metadata = Path(args.metadata) if args.metadata else root / "resolved_models.json"
    if args.resolve_metadata:
        result = resolve_huggingface_metadata(args.config, metadata)
    elif args.download:
        result = download_huggingface_models(
            args.config, metadata, root / "download_manifest.json"
        )
    else:
        result = run_huggingface_preflight(
            args.config, execute=args.execute, metadata_path=metadata if args.metadata else None
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not (args.resolve_metadata or args.download or args.execute):
        print("Dry run only: no metadata, weights, dataset, credentials, or model were used.")


if __name__ == "__main__":
    main()
