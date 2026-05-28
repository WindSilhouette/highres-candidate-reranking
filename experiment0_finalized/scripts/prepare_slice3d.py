"""
scripts/prepare_slice3d.py
--------------------------
Adapts a SLICE-3D metadata CSV to the format expected by experiment0.

SLICE-3D CSV columns (typical):
    isic_id, patient_id, target (0/1), image_path,
    diagnosis, anatom_site_general, age_approx, sex, ...

This script:
  1. Renames columns to match experiment0 schema
  2. Validates required fields
  3. Reports missingness and patient counts
  4. Writes cleaned CSV ready for experiment0

Usage:
    python scripts/prepare_slice3d.py \
        --input /data/slice3d/train-metadata.csv \
        --image-root /data/slice3d/train-image/ \
        --output outputs/slice3d_ready.csv
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd


# Column name mapping: SLICE-3D → experiment0
COLUMN_MAP = {
    "isic_id":              "lesion_id",
    "patient_id":           "patient_id",
    "target":               "malignant",
    "image_path":           "image_path",
    "diagnosis":            "diagnosis",
    "anatom_site_general":  "anatomical_site",
    "age_approx":           "age",
    "sex":                  "sex",
}

REQUIRED_OUT = ["patient_id", "lesion_id", "malignant", "image_path"]


def prepare_slice3d(input_csv: str, image_root: str, output_csv: str):
    print(f"\nLoading: {input_csv}")
    df = pd.read_csv(input_csv)
    print(f"  Raw shape: {df.shape}")
    print(f"  Columns: {df.columns.tolist()}")

    # Rename known columns
    rename = {k: v for k, v in COLUMN_MAP.items() if k in df.columns}
    df = df.rename(columns=rename)
    print(f"\nRenamed: {rename}")

    # Build image_path from isic_id if not already present
    if "image_path" not in df.columns and "lesion_id" in df.columns:
        root = Path(image_root)
        df["image_path"] = df["lesion_id"].apply(
            lambda x: str(root / f"{x}.jpg")
        )
        print("  image_path built from lesion_id + image_root")

    # Add label column (copy of malignant for compatibility)
    if "malignant" in df.columns:
        df["label"] = df["malignant"]

    # Validate required
    missing = [c for c in REQUIRED_OUT if c not in df.columns]
    if missing:
        print(f"\nERROR: Missing required columns after renaming: {missing}")
        print("Please update COLUMN_MAP in this script to match your CSV.")
        sys.exit(1)

    # Add empty split column
    df["split"] = ""

    # Compute n_lesions per patient
    n_les = df.groupby("patient_id")["lesion_id"].transform("count")
    df["n_lesions_patient"] = n_les

    # Add embedding_index placeholder (filled after extraction)
    df["embedding_index"] = range(len(df))

    # Report
    print(f"\n[Summary]")
    print(f"  Patients  : {df['patient_id'].nunique()}")
    print(f"  Lesions   : {len(df)}")
    if "malignant" in df.columns:
        print(f"  Malignant : {df['malignant'].sum()} "
              f"({100*df['malignant'].mean():.1f}%)")
    missing_vals = df[REQUIRED_OUT].isnull().sum()
    print(f"\n  Missing values in required columns:")
    print(missing_vals.to_string())

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"\nCleaned CSV saved to: {output_csv}")
    print("Next: run python run_experiment0.py --csv", output_csv)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",      required=True)
    parser.add_argument("--image-root", default="")
    parser.add_argument("--output",     default="outputs/slice3d_ready.csv")
    args = parser.parse_args()
    prepare_slice3d(args.input, args.image_root, args.output)


if __name__ == "__main__":
    main()
