"""Convert GDINA's frac20 RDA into the repository's local CSV format.

The source archive is supplied by the user; this script does not redistribute the data.
Item expressions come from DeCarlo (2011), Table 12, arXiv:1303.0426.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import tarfile
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import rdata

ATTRIBUTES = [
    "convert_whole_to_fraction",
    "separate_whole_from_fraction",
    "simplify_before_subtracting",
    "find_common_denominator",
    "borrow_from_whole_part",
    "column_borrow",
    "subtract_numerators",
    "reduce_to_simplest_form",
]

# Verbatim mathematical content from DeCarlo (2011), Table 12, rendered in LaTeX.
ITEM_TEXTS = [
    r"\(5/3-3/4\)",
    r"\(3/4-3/8\)",
    r"\(5/6-1/9\)",
    r"\(3\frac{1}{2}-2\frac{3}{2}\)",
    r"\(4\frac{3}{5}-3\frac{4}{10}\)",
    r"\(6/7-4/7\)",
    r"\(3-2\frac{1}{5}\)",
    r"\(2/3-2/3\)",
    r"\(3\frac{7}{8}-2\)",
    r"\(4\frac{4}{12}-2\frac{7}{12}\)",
    r"\(4\frac{1}{3}-2\frac{4}{3}\)",
    r"\(11/8-1/8\)",
    r"\(3\frac{3}{8}-2\frac{5}{6}\)",
    r"\(3\frac{4}{5}-3\frac{2}{5}\)",
    r"\(2-1/3\)",
    r"\(4\frac{5}{7}-1\frac{4}{7}\)",
    r"\(7\frac{3}{5}-4/5\)",
    r"\(4\frac{1}{10}-2\frac{8}{10}\)",
    r"\(4-1\frac{4}{3}\)",
    r"\(4\frac{1}{3}-1\frac{5}{3}\)",
]

# Canonical rational answers; the scorer also accepts equivalent mixed numbers and decimals.
ANSWER_KEYS = [
    "11/12",
    "3/8",
    "13/18",
    "0",
    "6/5",
    "2/7",
    "4/5",
    "0",
    "15/8",
    "7/4",
    "1",
    "5/4",
    "13/24",
    "2/5",
    "5/3",
    "22/7",
    "34/5",
    "13/10",
    "5/3",
    "5/3",
]

# DeCarlo (2011), Table 13; used as an independent check against GDINA's Q matrix.
EXPECTED_Q = np.array(
    [
        [0, 0, 0, 1, 0, 1, 1, 0],
        [0, 0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1, 0],
        [0, 1, 1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [1, 1, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 1, 0, 1, 1],
        [0, 1, 0, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 1, 1],
        [0, 1, 0, 1, 1, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 1, 0],
        [1, 0, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 1, 0, 1, 0],
        [0, 1, 0, 0, 1, 1, 1, 0],
        [1, 1, 1, 0, 1, 0, 1, 0],
        [0, 1, 1, 0, 1, 0, 1, 0],
    ],
    dtype=np.int8,
)


def prepare(archive_path: Path, output_dir: Path) -> None:
    archive_bytes = archive_path.read_bytes()
    with tempfile.TemporaryDirectory() as directory:
        rda_path = Path(directory) / "frac20.rda"
        with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as archive:
            member = archive.getmember("GDINA/data/frac20.rda")
            source = archive.extractfile(member)
            if source is None:
                raise ValueError("could not extract GDINA/data/frac20.rda")
            rda_path.write_bytes(source.read())
        obj = rdata.read_rda(rda_path)["frac20"]

    responses = obj["dat"].to_numpy(dtype=np.int8)
    q_matrix = obj["Q"].to_numpy(dtype=np.int8)
    if responses.shape != (536, 20) or q_matrix.shape != (20, 8):
        raise ValueError("unexpected frac20 dimensions")
    if not np.isin(responses, [0, 1]).all():
        raise ValueError("responses are not binary")
    if not np.array_equal(q_matrix, EXPECTED_Q):
        raise ValueError("GDINA Q matrix differs from DeCarlo (2011), Table 13")

    item_ids = [f"item_{index:03d}" for index in range(1, 21)]
    student_ids = [f"student_{index:03d}" for index in range(1, 537)]
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        responses, index=pd.Index(student_ids, name="student_id"), columns=item_ids
    ).to_csv(output_dir / "responses.csv")
    pd.DataFrame(
        q_matrix, index=pd.Index(item_ids, name="item_id"), columns=ATTRIBUTES
    ).to_csv(output_dir / "q_matrix.csv")
    pd.DataFrame(
        {"item_id": item_ids, "text": ITEM_TEXTS, "answer_key": ANSWER_KEYS}
    ).set_index("item_id").to_csv(output_dir / "items.csv")

    manifest = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source_archive": archive_path.name,
        "source_archive_sha256": hashlib.sha256(archive_bytes).hexdigest(),
        "response_and_q_source": "GDINA 2.9.4, GDINA/data/frac20.rda",
        "response_and_q_url": (
            "https://cran.r-project.org/src/contrib/Archive/GDINA/GDINA_2.9.4.tar.gz"
        ),
        "item_and_q_reference": "DeCarlo (2011), Tables 12-13, arXiv:1303.0426",
        "item_url": "https://arxiv.org/pdf/1303.0426",
        "license_note": (
            "No separate redistribution permission was verified. Keep generated data local "
            "until the data and item licenses are confirmed."
        ),
    }
    (output_dir / "source_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Prepared {responses.shape[0]} students x {responses.shape[1]} items in {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path, help="GDINA_2.9.4.tar.gz")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    prepare(args.archive, args.output_dir)


if __name__ == "__main__":
    main()
