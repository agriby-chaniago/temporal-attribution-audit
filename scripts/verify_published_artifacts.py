"""Verify the curated Git publication; never run a scientific estimator or model."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    ledger = root / "docs/PUBLISHED_FILE_HASHES.csv"
    failures: list[str] = []
    seen: set[str] = set()
    with ledger.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        relative = row["path"]
        path = (root / relative).resolve()
        if relative in seen or not path.is_relative_to(root):
            failures.append(f"Invalid or duplicate path: {relative}")
            continue
        seen.add(relative)
        if not path.is_file():
            failures.append(f"Missing published file: {relative}")
            continue
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        if digest != row["sha256"] or path.stat().st_size != int(row["bytes"]):
            failures.append(f"Published bytes differ: {relative}")
    for failure in failures:
        print(failure)
    print(f"Checked {len(rows)} published files; failures: {len(failures)}")
    print("This verifies stored bytes only, not scientific validity or model replay.")
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
