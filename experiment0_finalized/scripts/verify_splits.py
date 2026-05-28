"""
scripts/verify_splits.py
------------------------
Standalone script to verify patient-disjoint splits on any CSV.
Run this independently at any time to confirm no leakage.

Usage:
    python scripts/verify_splits.py --csv outputs/toy_dataset.csv
    python scripts/verify_splits.py --csv path/to/slice3d.csv
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from src.data.splitter import verify_splits


def main():
    parser = argparse.ArgumentParser(
        description="Verify patient-disjoint splits in a dataset CSV"
    )
    parser.add_argument("--csv", required=True,
                        help="Path to CSV with patient_id and split columns")
    parser.add_argument("--split-col", default="split",
                        help="Name of the split column (default: 'split')")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    required_cols = ["patient_id", args.split_col]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"ERROR: Missing columns: {missing}")
        print(f"Available columns: {df.columns.tolist()}")
        sys.exit(1)

    print(f"\nLoaded {len(df)} rows from {args.csv}")
    verify_splits(df, split_col=args.split_col)


if __name__ == "__main__":
    main()
