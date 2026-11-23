#!/usr/bin/env python3
"""Regenerate every figure in results/ from a fresh experiment run.

Usage:
    python scripts/make_plots.py [--include-page-values] [--out results]

This is the ONLY way the committed figures are produced. If a number or
plot in the README is ever in doubt, rerun this script and
run_experiment.py and compare.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from purchase_intent.data import load_dataset
from purchase_intent.plots import plot_pr_curves, plot_reliability
from purchase_intent.train import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(ROOT / "data" / "online_shoppers_intention.csv"))
    parser.add_argument("--out", default=str(ROOT / "results"))
    parser.add_argument("--include-page-values", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = load_dataset(args.data)
    results = run_experiment(
        df, include_page_values=args.include_page_values, seed=args.seed
    )
    scores = results["_scores"]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_pagevalues" if args.include_page_values else ""
    pr_path = out_dir / f"pr_curves{suffix}.png"
    rel_path = out_dir / f"reliability{suffix}.png"
    plot_pr_curves(scores, str(pr_path))
    plot_reliability(scores, str(rel_path))
    print(f"Wrote {pr_path}")
    print(f"Wrote {rel_path}")


if __name__ == "__main__":
    main()
