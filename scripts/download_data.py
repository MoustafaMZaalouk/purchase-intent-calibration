#!/usr/bin/env python3
"""Download the Online Shoppers Purchasing Intention dataset.

Fetches the CSV from the UCI Machine Learning Repository (dataset 468,
CC BY 4.0) into data/, validates its schema, and prints the SHA-256 so
the exact file can be pinned in reports.
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from purchase_intent.data import DATA_URL, load_dataset, sha256_file

OUT = Path(__file__).resolve().parent.parent / "data" / "online_shoppers_intention.csv"


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {DATA_URL}")
    urllib.request.urlretrieve(DATA_URL, OUT)
    digest = sha256_file(OUT)
    df = load_dataset(OUT)
    print(f"Saved {OUT} ({len(df)} rows)")
    print(f"SHA-256: {digest}")
    print(f"Positive rate (Revenue=True): {df['Revenue'].mean():.4f}")


if __name__ == "__main__":
    main()
