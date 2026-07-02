"""
src/io_utils.py
===============
Small tabular I/O wrapper. FABLE prefers parquet, but local sandboxes and some
fresh Colab runtimes may not have pyarrow/fastparquet yet. These helpers keep the
pipeline runnable by falling back to pickle while preserving the requested path.
"""

from __future__ import annotations

import os
import pandas as pd


_PARQUET_HINTS = ("pyarrow", "fastparquet", "parquet", "Unable to find a usable engine")


def _looks_like_missing_parquet_engine(exc: Exception) -> bool:
    msg = str(exc)
    return any(h.lower() in msg.lower() for h in _PARQUET_HINTS)


def write_table(df: pd.DataFrame, path: str, index: bool = False) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    try:
        df.to_parquet(path, index=index)
        with open(path + ".format", "w") as f:
            f.write("parquet\n")
    except Exception as exc:
        if not _looks_like_missing_parquet_engine(exc):
            raise
        df.to_pickle(path)
        with open(path + ".format", "w") as f:
            f.write("pickle_fallback\n")
        print(f">> parquet engine unavailable; wrote pickle fallback at {path}")


def read_table(path: str) -> pd.DataFrame:
    try:
        return pd.read_parquet(path)
    except Exception as first_exc:
        try:
            return pd.read_pickle(path)
        except Exception:
            raise first_exc
