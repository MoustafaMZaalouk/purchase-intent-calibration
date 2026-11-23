#!/usr/bin/env python3
"""Run the full experiment: train, tune on validation, evaluate on test.

Usage:
    python scripts/run_experiment.py [--include-page-values] \
        [--data data/online_shoppers_intention.csv] [--out results]

Writes <out>/metrics.json (or metrics_pagevalues.json) and prints a
summary table. Every number in the README comes from these files.
Deterministic for a fixed seed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from purchase_intent.data import load_dataset, sha256_file
from purchase_intent.train import public_report, run_experiment


def format_table(report: dict) -> str:
    lines = []
    header = (
        f"{'model':<22} {'split':<6} {'PR-AUC':>7} {'ROC-AUC':>8} "
        f"{'Brier':>7} {'ECE':>7} {'thr':>7} {'P':>6} {'R':>6} {'F1':>6}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for name, entry in report["models"].items():
        for split in ("validation", "test"):
            m = entry[split]
            at = m["at_threshold"]
            lines.append(
                f"{name:<22} {split:<6} {m['pr_auc']:>7.4f} "
                f"{m['roc_auc']:>8.4f} {m['brier']:>7.4f} {m['ece']:>7.4f} "
                f"{at['threshold']:>7.3f} {at['precision']:>6.3f} "
                f"{at['recall']:>6.3f} {at['f1']:>6.3f}"
            )
        for fixed, tm in entry["test"]["at_fixed_thresholds"].items():
            lines.append(
                f"{name:<22} {'test':<6} {'':>7} {'':>8} {'':>7} {'':>7} "
                f"{tm['threshold']:>7.3f} {tm['precision']:>6.3f} "
                f"{tm['recall']:>6.3f} {tm['f1']:>6.3f}   (fixed {fixed})"
            )
        cal = entry["calibration_on_test"]
        lines.append(
            f"  calibration on test - raw ECE {cal['raw']['ece']:.4f} "
            f"Brier {cal['raw']['brier']:.4f} | platt ECE "
            f"{cal['platt']['ece']:.4f} Brier {cal['platt']['brier']:.4f} "
            f"| isotonic ECE {cal['isotonic']['ece']:.4f} Brier "
            f"{cal['isotonic']['brier']:.4f}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(ROOT / "data" / "online_shoppers_intention.csv"))
    parser.add_argument("--out", default=str(ROOT / "results"))
    parser.add_argument("--include-page-values", action="store_true",
                        help="add PageValues as a feature (comparison run; "
                             "excluded by default as borderline leakage)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = load_dataset(args.data)
    results = run_experiment(
        df, include_page_values=args.include_page_values, seed=args.seed
    )
    report = public_report(results)
    report["data"] = {
        "path": str(Path(args.data).name),
        "sha256": sha256_file(args.data),
        "rows": len(df),
        "positive_rate": float(df["Revenue"].mean()),
    }

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_pagevalues" if args.include_page_values else ""
    out_file = out_dir / f"metrics{suffix}.json"
    out_file.write_text(json.dumps(report, indent=2) + "\n")

    print(f"PageValues included: {args.include_page_values}")
    print(format_table(report))
    print(f"\nWrote {out_file}")


if __name__ == "__main__":
    main()
