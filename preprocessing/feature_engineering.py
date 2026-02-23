"""
preprocessing/feature_engineering.py
──────────────────────────────────────
Step 5 — Create lag features and normalise the dataset.

Pipeline
--------
1. Create PM10_lag (= PM10 shifted by 1 step) + optional extra lags/windows.
2. Drop any rows that became NaN after shifting.
3. Fit a MinMaxScaler on the training split ONLY (avoids data leakage).
4. Scale both train and validation splits with that scaler.
5. Persist the fitted scaler to disk (data/scalers/feature_scaler.pkl).
6. Return scaled train and val DataFrames.
"""

import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

# ── Allow running as a standalone script from any cwd ─────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import FEATURES, TARGET, VAL_SPLIT, SCALER_PATH


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add engineered features (lag, rolling statistics) to the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Clean DataFrame produced by data_loader.load_raw_data().

    Returns
    -------
    pd.DataFrame
        DataFrame with new feature columns appended. Rows with NaN
        (caused by shifting) are dropped.
    """
    df = df.copy()

    # ── Core lag feature (mandatory) ─────────────────────────────────────────
    df['PM10_lag'] = df[TARGET].shift(1)

    # ── Optional enrichment ──────────────────────────────────────────────────
    # Additional lags
    df['PM10_lag2'] = df[TARGET].shift(2)
    df['PM10_lag3'] = df[TARGET].shift(3)

    # Rolling statistics (24-step window → 24 hour window for hourly data)
    df['PM10_rolling_mean_24'] = df[TARGET].rolling(window=24, min_periods=1).mean()
    df['PM10_rolling_std_24']  = df[TARGET].rolling(window=24, min_periods=1).std().fillna(0)

    # Hour-of-day & day-of-week cyclic encoding (if DatetimeIndex)
    if isinstance(df.index, pd.DatetimeIndex):
        hour = df.index.hour
        dow  = df.index.dayofweek
        df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * hour / 24)
        df['dow_sin']  = np.sin(2 * np.pi * dow  / 7)
        df['dow_cos']  = np.cos(2 * np.pi * dow  / 7)

    # Drop NaN rows introduced by shifting
    before = len(df)
    df = df.dropna()
    print(f"[feature_engineering] Added lag/rolling features. "
          f"Dropped {before - len(df)} NaN rows -> {len(df):,} rows remain.")

    return df


def normalize_data(
    df: pd.DataFrame,
    scaler_path: str = SCALER_PATH,
    val_split: float = VAL_SPLIT,
) -> tuple[pd.DataFrame, pd.DataFrame, MinMaxScaler]:
    """
    Split into train / val, fit MinMaxScaler on train, scale both splits.

    Parameters
    ----------
    df          : Engineered DataFrame (output of build_features).
    scaler_path : Where to save the fitted scaler (.pkl).
    val_split   : Fraction of data to reserve for validation.

    Returns
    -------
    (train_df, val_df, scaler)
        Both DataFrames are scaled; scaler is the fitted MinMaxScaler.
    """
    # ── Train / Val split (chronological — NO shuffle) ───────────────────────
    split_idx = int(len(df) * (1 - val_split))
    train_df  = df.iloc[:split_idx].copy()
    val_df    = df.iloc[split_idx:].copy()
    print(f"[feature_engineering] Train: {len(train_df):,} rows | "
          f"Val: {len(val_df):,} rows  (split @ idx {split_idx})")

    # ── Identify columns to scale (all numeric) ──────────────────────────────
    scale_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()

    # ── Fit scaler on TRAIN only ─────────────────────────────────────────────
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_df[scale_cols] = scaler.fit_transform(train_df[scale_cols])
    val_df[scale_cols]   = scaler.transform(val_df[scale_cols])

    # ── Persist scaler ────────────────────────────────────────────────────────
    scaler_path = Path(scaler_path)
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"[feature_engineering] Scaler saved -> '{scaler_path}'")

    return train_df, val_df, scaler


def load_scaler(scaler_path: str = SCALER_PATH) -> MinMaxScaler:
    """Load a previously saved MinMaxScaler from disk."""
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    print(f"[feature_engineering] Scaler loaded from '{scaler_path}'")
    return scaler


# ─────────────────────────────────────────────────────────────────────────────
# Script entry-point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    from data_loader import load_raw_data

    df           = load_raw_data()
    df_eng       = build_features(df)
    train, val, _ = normalize_data(df_eng)

    print("\n── Train head (scaled) ──")
    print(train.head())
    print("\n── Val head (scaled) ──")
    print(val.head())
