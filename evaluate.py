"""
evaluate.py
------------
Step 10 -- Evaluate the trained CNN-LSTM-Attention model on the
           validation set and persist all metrics to metrics.json.

Workflow
--------
1.  Load X_val.npy / y_val.npy from data/processed/
2.  Load the fitted MinMaxScaler  (data/scalers/feature_scaler.pkl)
3.  Load the best model checkpoint (models/model_checkpoint.pth)
4.  Run inference on the full validation set
5.  Inverse-transform predictions and targets back to original PM10 scale
6.  Compute:
        RMSE  -- Root Mean Squared Error
        MAE   -- Mean Absolute Error
        MAPE  -- Mean Absolute Percentage Error
        R2    -- Coefficient of Determination
7.  Save metrics.json  (results/metrics.json)
8.  Print a formatted evaluation report
"""

import sys
import json
import math
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

# -- Project root on sys.path --------------------------------------------------
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config import (
    PROCESSED_DIR,
    SCALER_PATH,
    CHECKPOINT_PATH,
    METRICS_PATH,
    RANDOM_SEED,
)
from models.cnn_lstm_attention import build_model


# -----------------------------------------------------------------------------
# Metric functions
# -----------------------------------------------------------------------------

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(math.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    """Mean Absolute Percentage Error (%)."""
    return float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100)


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of Determination (R^2)."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / (ss_tot + 1e-8))


# -----------------------------------------------------------------------------
# Data helpers
# -----------------------------------------------------------------------------

def load_val_data(processed_dir: str = PROCESSED_DIR):
    """Load X_val and y_val numpy arrays from disk."""
    d = Path(processed_dir)
    X_val = np.load(d / "X_val.npy")
    y_val = np.load(d / "y_val.npy")
    print(f"[evaluate] Loaded X_val {X_val.shape}  y_val {y_val.shape}")
    return X_val, y_val


def load_model(checkpoint_path: str, num_features: int) -> nn.Module:
    """
    Build the architecture and load saved weights.

    Parameters
    ----------
    checkpoint_path : path to model_checkpoint.pth
    num_features    : number of input feature channels (inferred from X_val)

    Returns
    -------
    model in eval mode on CPU.
    """
    model = build_model(num_features=num_features)
    state_dict = torch.load(checkpoint_path, map_location="cpu",
                            weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"[evaluate] Checkpoint loaded from '{checkpoint_path}'")
    return model


# -----------------------------------------------------------------------------
# Inverse-transform helpers
# -----------------------------------------------------------------------------

def inverse_transform_target(
    values: np.ndarray,
    scaler,
    num_total_features: int,
    target_col_index: int = 0,
) -> np.ndarray:
    """
    Inverse-scale a 1-D array of target values.

    MinMaxScaler was fit on ALL columns jointly, so we must reconstruct a
    dummy array of the same width, place the target values in the correct
    column, inverse-transform, then extract that column back.

    Parameters
    ----------
    values             : 1-D array of scaled target predictions / ground truth.
    scaler             : the fitted MinMaxScaler.
    num_total_features : total number of columns the scaler was fit on.
    target_col_index   : column index of PM10 (target) in the scaler's matrix.
                         Defaults to 0 (PM10 is typically the first column).
    """
    dummy = np.zeros((len(values), num_total_features), dtype=np.float32)
    dummy[:, target_col_index] = values
    inv = scaler.inverse_transform(dummy)
    return inv[:, target_col_index]


# -----------------------------------------------------------------------------
# Inference
# -----------------------------------------------------------------------------

def predict(model: nn.Module, X_val: np.ndarray,
            batch_size: int = 256) -> np.ndarray:
    """
    Run batch inference and return predictions as numpy array.

    Parameters
    ----------
    model      : trained model in eval mode.
    X_val      : (N, T, F) float32 numpy array.
    batch_size : inference batch size (no gradient needed; limited by RAM).

    Returns
    -------
    preds : (N,) numpy array of raw (scaled) predictions.
    """
    device = next(model.parameters()).device
    preds  = []

    with torch.no_grad():
        for i in range(0, len(X_val), batch_size):
            x_batch = torch.tensor(
                X_val[i : i + batch_size], dtype=torch.float32
            ).to(device)
            out, _ = model(x_batch)          # (B, 1)
            preds.append(out.squeeze(1).cpu().numpy())

    return np.concatenate(preds, axis=0)


# -----------------------------------------------------------------------------
# Main evaluation routine
# -----------------------------------------------------------------------------

def evaluate(
    processed_dir   : str = PROCESSED_DIR,
    checkpoint_path : str = CHECKPOINT_PATH,
    scaler_path     : str = SCALER_PATH,
    metrics_path    : str = METRICS_PATH,
) -> dict:
    """
    Full evaluation pipeline.

    Returns
    -------
    metrics : dict with RMSE, MAE, MAPE, R2 (on original PM10 scale).
    """
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # -- 1. Load validation data -----------------------------------------------
    X_val, y_val_scaled = load_val_data(processed_dir)
    num_features = X_val.shape[2]

    # -- 2. Load scaler --------------------------------------------------------
    import pickle
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    num_scaler_features = scaler.n_features_in_
    print(f"[evaluate] Scaler loaded  (n_features={num_scaler_features})")

    # -- 3. Load model ---------------------------------------------------------
    model = load_model(checkpoint_path, num_features)

    # -- 4. Predict (scaled space) --------------------------------------------
    preds_scaled = predict(model, X_val)
    print(f"[evaluate] Predictions shape: {preds_scaled.shape}")

    # -- 5. Inverse-transform both predictions and ground truth ----------------
    # Determine the column index of PM10 in the scaler's feature matrix.
    # The scaler was fit on all numeric columns; PM10 is almost always index 0.
    # If the column order is available in the scaler (pandas-aware scalers
    # set feature_names_in_), we detect it dynamically.
    target_col_idx = 0
    if hasattr(scaler, "feature_names_in_"):
        names = list(scaler.feature_names_in_)
        if "PM10" in names:
            target_col_idx = names.index("PM10")

    preds_orig = inverse_transform_target(
        preds_scaled, scaler, num_scaler_features, target_col_idx
    )
    y_true_orig = inverse_transform_target(
        y_val_scaled.astype(np.float32), scaler,
        num_scaler_features, target_col_idx
    )

    # -- 6. Compute metrics ---------------------------------------------------
    val_rmse = rmse(y_true_orig, preds_orig)
    val_mae  = mae(y_true_orig,  preds_orig)
    val_mape = mape(y_true_orig, preds_orig)
    val_r2   = r2_score(y_true_orig, preds_orig)

    metrics = {
        "RMSE" : round(val_rmse, 6),
        "MAE"  : round(val_mae,  6),
        "MAPE" : round(val_mape, 6),
        "R2"   : round(val_r2,   6),
    }

    # -- 7. Save metrics.json -------------------------------------------------
    Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)

    # Merge with existing metrics.json (keeps training history if present)
    existing = {}
    if Path(metrics_path).exists():
        with open(metrics_path, "r") as f:
            try:
                existing = json.load(f)
            except json.JSONDecodeError:
                existing = {}

    existing.update({"evaluation": metrics})

    with open(metrics_path, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"[evaluate] Metrics saved -> '{metrics_path}'")

    # -- 8. Report ------------------------------------------------------------
    print()
    print("=" * 50)
    print("  Evaluation Results  (original PM10 scale)")
    print("=" * 50)
    print(f"  RMSE : {val_rmse:.4f}")
    print(f"  MAE  : {val_mae:.4f}")
    print(f"  MAPE : {val_mape:.4f} %")
    print(f"  R2   : {val_r2:.4f}")
    print("=" * 50)

    return metrics


# -----------------------------------------------------------------------------
# Script entry-point
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    evaluate()
