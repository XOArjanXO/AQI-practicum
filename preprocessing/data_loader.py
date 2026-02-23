"""
preprocessing/data_loader.py
────────────────────────────
Step 4 — Load raw AirNet CSV data and return a clean DataFrame.

Pipeline
--------
1. Read the CSV.
2. Parse / set a datetime index (if a date/time column exists).
3. Validate that all expected feature columns are present.
4. Handle missing values:
   - Forward-fill  (propagate last valid observation forward)
   - Backward-fill (fill any leading NaNs)
   - Drop rows that are still NaN after both fills.
5. Remove duplicate timestamps (keep first occurrence).
6. Return the clean DataFrame, sorted by time.
"""

import sys
import pandas as pd
from pathlib import Path

# ── Allow running as a standalone script from any cwd ─────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import RAW_DATA_PATH, FEATURES, TARGET


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def load_raw_data(csv_path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Load and clean the raw AirNet CSV file.

    Parameters
    ----------
    csv_path : str
        Path to the raw CSV file (default from config.RAW_DATA_PATH).

    Returns
    -------
    pd.DataFrame
        Clean DataFrame with a DatetimeIndex, sorted chronologically.

    Raises
    ------
    FileNotFoundError
        If the CSV does not exist at ``csv_path``.
    ValueError
        If required columns are missing from the CSV.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Raw data file not found: {csv_path}")

    # ── 1. Read CSV ──────────────────────────────────────────────────────────
    df = pd.read_csv(csv_path)
    print(f"[data_loader] Loaded {len(df):,} rows x {df.shape[1]} cols from '{csv_path}'")

    # ── 2. Parse datetime ────────────────────────────────────────────────────
    df = _parse_datetime_index(df)

    # ── 3. Validate columns ──────────────────────────────────────────────────
    required_cols = [TARGET] + [
        f for f in FEATURES if f != 'PM10_lag'  # PM10_lag created later
    ]
    _validate_columns(df, required_cols)

    # ── 4. Handle missing values ─────────────────────────────────────────────
    before = len(df)
    df = df.sort_index()                    # sort chronologically first
    df = df[~df.index.duplicated(keep='first')]   # remove duplicate timestamps
    df = df.ffill()                         # forward-fill
    df = df.bfill()                         # backward-fill (catches leading NaNs)

    still_nan = df.isnull().sum().sum()
    if still_nan > 0:
        print(f"[data_loader] WARNING: {still_nan} NaN values remain -- dropping those rows.")
        df = df.dropna()

    after = len(df)
    dropped = before - after
    if dropped:
        print(f"[data_loader] Dropped {dropped} rows during cleaning.")

    print(f"[data_loader] Clean DataFrame: {len(df):,} rows, columns: {list(df.columns)}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """Detect and parse a datetime column, setting it as the index."""
    datetime_candidates = ['datetime', 'date', 'timestamp', 'time', 'Date', 'DateTime']
    for col in datetime_candidates:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], infer_datetime_format=True, errors='coerce')
            df = df.set_index(col)
            print(f"[data_loader] Parsed datetime index from column '{col}'.")
            return df

    # Fallback: if the index already looks like dates, parse it
    try:
        df.index = pd.to_datetime(df.index, infer_datetime_format=True)
        print("[data_loader] Parsed existing index as datetime.")
    except Exception:
        print("[data_loader] WARNING: No datetime column found -- using default integer index.")
    return df


def _validate_columns(df: pd.DataFrame, required: list) -> None:
    """Raise ValueError if any required column is missing."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"[data_loader] Missing columns in CSV: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )
    print(f"[data_loader] All required columns present: {required}")


# ─────────────────────────────────────────────────────────────────────────────
# Script entry-point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    df = load_raw_data()
    print("\n── Head ──")
    print(df.head())
    print("\n── Info ──")
    df.info()
    print("\n── Null counts ──")
    print(df.isnull().sum())
