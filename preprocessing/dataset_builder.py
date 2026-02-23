"""
preprocessing/dataset_builder.py
──────────────────────────────────
Step 6 — Convert engineered DataFrames into sliding-window sequences
          and save as NumPy arrays.

Sequence Shape
--------------
  X : (num_samples, TIME_STEPS, num_features)   ← input window
  y : (num_samples,)                             ← next-step PM10 target

Output Files
------------
  data/processed/X_train.npy
  data/processed/y_train.npy
  data/processed/X_val.npy
  data/processed/y_val.npy
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# ── Allow running as a standalone script from any cwd ─────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    TIME_STEPS,
    FEATURES,
    TARGET,
    PROCESSED_DIR,
    RAW_DATA_PATH,
    RANDOM_SEED,
)


# ─────────────────────────────────────────────────────────────────────────────
# Core helpers
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_feature_cols(df: pd.DataFrame) -> list[str]:
    """
    Return the feature columns that are actually present in *df*.

    Config FEATURES may include 'PM10_lag' (created in feature_engineering).
    Any config column that is missing is skipped with a warning
    so the pipeline stays robust.
    """
    available = df.columns.tolist()
    resolved  = [f for f in FEATURES if f in available]
    extra     = [c for c in available
                 if c not in FEATURES
                 and c != TARGET
                 and c not in resolved]

    # Include engineered extras (rolling stats, cyclic time, extra lags)
    resolved += extra

    # Filter to only numeric columns to avoid "could not convert string to float"
    resolved = [c for c in resolved if pd.api.types.is_numeric_dtype(df[c])]

    missing = [f for f in FEATURES if f not in available]
    if missing:
        print(f"[dataset_builder] WARNING: config FEATURES not found in df "
              f"(will be skipped): {missing}")
    print(f"[dataset_builder] Using {len(resolved)} feature columns: {resolved}")
    return resolved


def create_sequences(
    df: pd.DataFrame,
    time_steps: int = TIME_STEPS,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build overlapping sliding-window sequences from a scaled DataFrame.

    Parameters
    ----------
    df         : Scaled DataFrame (train or val split).
    time_steps : Length of each input window.

    Returns
    -------
    (X, y)
        X : np.ndarray, shape (N, time_steps, num_features), dtype float32
        y : np.ndarray, shape (N,),                         dtype float32
    """
    feature_cols = _resolve_feature_cols(df)

    if TARGET not in df.columns:
        raise ValueError(f"[dataset_builder] Target column '{TARGET}' not found in DataFrame.")

    data   = df[feature_cols].values.astype(np.float32)
    target = df[TARGET].values.astype(np.float32)

    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i : i + time_steps])        # window of length time_steps
        y.append(target[i + time_steps])           # value immediately after window

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    print(f"[dataset_builder] Sequences built -> X: {X.shape}, y: {y.shape}")
    return X, y


def save_arrays(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val:   np.ndarray,
    y_val:   np.ndarray,
    save_dir: str = PROCESSED_DIR,
) -> None:
    """
    Persist the four arrays to *save_dir* as .npy files.

    Parameters
    ----------
    X_train, y_train, X_val, y_val : Numpy arrays from create_sequences().
    save_dir                        : Directory path (created if absent).
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        'X_train.npy': X_train,
        'y_train.npy': y_train,
        'X_val.npy'  : X_val,
        'y_val.npy'  : y_val,
    }
    for filename, array in paths.items():
        out_path = save_dir / filename
        np.save(out_path, array)
        print(f"[dataset_builder] Saved {filename:15s} -> shape {array.shape}  "
              f"@ '{out_path}'")


def load_arrays(
    save_dir: str = PROCESSED_DIR,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load the four .npy arrays back from disk.

    Returns
    -------
    (X_train, y_train, X_val, y_val)
    """
    save_dir = Path(save_dir)
    X_train = np.load(save_dir / 'X_train.npy')
    y_train = np.load(save_dir / 'y_train.npy')
    X_val   = np.load(save_dir / 'X_val.npy')
    y_val   = np.load(save_dir / 'y_val.npy')
    print(f"[dataset_builder] Loaded -- "
          f"X_train {X_train.shape}, y_train {y_train.shape}, "
          f"X_val {X_val.shape}, y_val {y_val.shape}")
    return X_train, y_train, X_val, y_val


# ─────────────────────────────────────────────────────────────────────────────
# Full pipeline convenience function
# ─────────────────────────────────────────────────────────────────────────────

def build_and_save_dataset(
    raw_csv_path: str = RAW_DATA_PATH,
    time_steps:   int = TIME_STEPS,
    save_dir:     str = PROCESSED_DIR,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    End-to-end helper: load → engineer → normalise → sequence → save.

    Returns
    -------
    (X_train, y_train, X_val, y_val)
    """
    # Import here to avoid circular imports when modules run standalone
    from data_loader import load_raw_data
    from feature_engineering import build_features, normalize_data

    print("=" * 60)
    print("Building dataset ...")
    print("=" * 60)

    np.random.seed(RANDOM_SEED)

    df              = load_raw_data(raw_csv_path)
    df_eng          = build_features(df)
    train_df, val_df, _ = normalize_data(df_eng)

    X_train, y_train = create_sequences(train_df, time_steps)
    X_val,   y_val   = create_sequences(val_df,   time_steps)

    save_arrays(X_train, y_train, X_val, y_val, save_dir)

    print("=" * 60)
    print("Dataset build complete [OK]")
    print(f"  X_train : {X_train.shape}")
    print(f"  y_train : {y_train.shape}")
    print(f"  X_val   : {X_val.shape}")
    print(f"  y_val   : {y_val.shape}")
    print("=" * 60)

    return X_train, y_train, X_val, y_val


# ─────────────────────────────────────────────────────────────────────────────
# Script entry-point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    build_and_save_dataset()
