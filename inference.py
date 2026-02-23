"""
inference.py
-------------
Phase 6 -- Real-world inference with the trained CNN-LSTM-Attention model.

Simulates production usage:
  1. Load the saved model checkpoint
  2. Load the fitted MinMaxScaler
  3. Accept a NEW raw input window  (TIME_STEPS rows x num_features cols)
  4. Scale the input with the saved scaler
  5. Run the model forward pass
  6. Inverse-transform the prediction back to original PM10 units
  7. Return / print the predicted PM10 value

Usage (command line)
--------------------
    # Predict from a CSV file containing exactly TIME_STEPS rows
    python inference.py --input path/to/window.csv

    # Predict from a pre-built .npy file  (TIME_STEPS, num_features)
    python inference.py --input path/to/window.npy

    # Run a self-contained demo on random data (no real CSV needed)
    python inference.py --demo
"""

import sys
import argparse
import pickle
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

# -- Project root on sys.path --------------------------------------------------
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config import (
    CHECKPOINT_PATH,
    SCALER_PATH,
    TIME_STEPS,
    FEATURES,
    RANDOM_SEED,
)
from models.cnn_lstm_attention import build_model


# -----------------------------------------------------------------------------
# Loading helpers
# -----------------------------------------------------------------------------

def load_inference_model(
    checkpoint_path: str = CHECKPOINT_PATH,
    num_features: int = len(FEATURES),
) -> nn.Module:
    """
    Build the model architecture and restore saved weights.

    Parameters
    ----------
    checkpoint_path : path to model_checkpoint.pth
    num_features    : number of input feature channels

    Returns
    -------
    model in eval mode, on CPU.
    """
    model = build_model(num_features=num_features)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    print(f"[inference] Model loaded from '{checkpoint_path}'")
    return model


def load_scaler(scaler_path: str = SCALER_PATH):
    """Load the fitted MinMaxScaler from disk."""
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    print(f"[inference] Scaler loaded from '{scaler_path}'  "
          f"(n_features={scaler.n_features_in_})")
    return scaler


# -----------------------------------------------------------------------------
# Input helpers
# -----------------------------------------------------------------------------

def load_input_window(input_path: str) -> np.ndarray:
    """
    Load a raw (un-scaled) input window from a .csv or .npy file.

    Expected shape: (TIME_STEPS, num_features)

    CSV format  -- one row per time-step, columns matching the feature order
                   used during training (header row is optional).
    NPY format  -- saved numpy array of shape (TIME_STEPS, num_features).

    Returns
    -------
    window : np.ndarray, shape (TIME_STEPS, num_features), dtype float32
    """
    p = Path(input_path)
    if not p.exists():
        raise FileNotFoundError(f"[inference] Input file not found: {p}")

    if p.suffix.lower() == ".npy":
        window = np.load(p).astype(np.float32)
    elif p.suffix.lower() == ".csv":
        import pandas as pd
        df = pd.read_csv(p)
        # Drop non-numeric columns (e.g. datetime index)
        df = df.select_dtypes(include=[np.number])
        window = df.values.astype(np.float32)
    else:
        raise ValueError(f"[inference] Unsupported file type: {p.suffix}. "
                         f"Use .csv or .npy")

    print(f"[inference] Input window loaded: shape={window.shape}")
    return window


def validate_window(window: np.ndarray, time_steps: int = TIME_STEPS) -> None:
    """Check that the window has the expected number of time-steps."""
    if window.ndim != 2:
        raise ValueError(f"[inference] Expected 2-D array (T, F), got shape {window.shape}")
    if window.shape[0] != time_steps:
        raise ValueError(
            f"[inference] Expected {time_steps} time-steps, got {window.shape[0]}. "
            f"Adjust TIME_STEPS in config.py or provide a correctly sized window."
        )


# -----------------------------------------------------------------------------
# Scale helpers
# -----------------------------------------------------------------------------

def scale_window(window: np.ndarray, scaler) -> np.ndarray:
    """
    Scale the raw input window using the loaded MinMaxScaler.

    The scaler was fit on a DataFrame with (potentially) more columns than
    just FEATURES (e.g. engineered lag/rolling columns).  We need to pass
    a matching number of columns.

    Strategy:
      - If window already has the same number of columns as the scaler, scale directly.
      - Otherwise, pad with zeros so inverse_transform is consistent.
    """
    n_scaler_features = scaler.n_features_in_
    n_window_features = window.shape[1]

    if n_window_features == n_scaler_features:
        return scaler.transform(window).astype(np.float32)

    # Pad with zeros for the extra engineered columns
    padded = np.zeros((len(window), n_scaler_features), dtype=np.float32)
    cols = min(n_window_features, n_scaler_features)
    padded[:, :cols] = window[:, :cols]
    scaled_padded = scaler.transform(padded).astype(np.float32)
    return scaled_padded[:, :n_window_features]   # return only feature columns


def inverse_transform_prediction(
    pred_scaled: float,
    scaler,
    target_col_index: int = 0,
) -> float:
    """
    Inverse-transform a single scaled prediction back to original PM10 units.

    Reconstructs a full dummy row so the MinMaxScaler can invert correctly.
    """
    dummy = np.zeros((1, scaler.n_features_in_), dtype=np.float32)
    dummy[0, target_col_index] = pred_scaled
    inv = scaler.inverse_transform(dummy)
    return float(inv[0, target_col_index])


# -----------------------------------------------------------------------------
# Core prediction function
# -----------------------------------------------------------------------------

def predict_pm10(
    window: np.ndarray,
    model: nn.Module,
    scaler,
    target_col_index: int = 0,
) -> dict:
    """
    Given a raw (un-scaled) input window, return the predicted PM10 value.

    Parameters
    ----------
    window           : (TIME_STEPS, num_features) raw numpy array.
    model            : trained model in eval mode.
    scaler           : fitted MinMaxScaler.
    target_col_index : column position of PM10 in the scaler's feature matrix.

    Returns
    -------
    result : dict with keys:
        'prediction_pm10'  -- predicted PM10 in original units
        'prediction_scaled'-- raw model output (scaled space)
        'attention_weights'-- (TIME_STEPS,) array for XAI visualisation
    """
    validate_window(window)

    # Scale input
    scaled = scale_window(window, scaler)          # (T, F)

    # Add batch dimension -> (1, T, F)
    x = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0)

    # Forward pass
    with torch.no_grad():
        pred_tensor, attn_tensor = model(x)        # (1,1), (1,T)

    pred_scaled    = float(pred_tensor.squeeze().item())
    attn_weights   = attn_tensor.squeeze(0).cpu().numpy()   # (T,)

    # Inverse-transform
    pred_pm10 = inverse_transform_prediction(pred_scaled, scaler, target_col_index)

    return {
        "prediction_pm10"   : round(pred_pm10, 4),
        "prediction_scaled" : round(pred_scaled, 6),
        "attention_weights" : attn_weights.tolist(),
    }


# -----------------------------------------------------------------------------
# CLI & demo
# -----------------------------------------------------------------------------

def _demo(checkpoint_path: str, scaler_path: str) -> None:
    """Self-contained demo using a random synthetic window."""
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    print("\n[inference] --- DEMO MODE (random synthetic input) ---")

    # Build + save a temporary model & scaler so the demo is always runnable
    num_features = len(FEATURES)
    model  = load_inference_model(checkpoint_path, num_features)
    scaler = load_scaler(scaler_path)

    # Synthetic raw window in a plausible PM10 feature range
    window = np.random.uniform(low=0.0, high=1.0,
                               size=(TIME_STEPS, num_features)).astype(np.float32)
    print(f"[inference] Demo input window shape: {window.shape}")

    result = predict_pm10(window, model, scaler)

    print()
    print("=" * 50)
    print("  Inference Result")
    print("=" * 50)
    print(f"  Predicted PM10   : {result['prediction_pm10']} ug/m3")
    print(f"  Scaled output    : {result['prediction_scaled']}")
    print(f"  Attention weights: [{', '.join(f'{w:.3f}' for w in result['attention_weights'][:6])} ...]")
    print("=" * 50)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="XAI Air Quality -- inference script"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--input", "-i", type=str,
        help="Path to a .csv or .npy file containing ONE input window "
             f"of shape ({TIME_STEPS}, num_features). Columns must match "
             "the feature order used during training."
    )
    group.add_argument(
        "--demo", action="store_true",
        help="Run a self-contained demo on random synthetic data."
    )
    parser.add_argument(
        "--checkpoint", type=str, default=CHECKPOINT_PATH,
        help=f"Path to model checkpoint (default: {CHECKPOINT_PATH})"
    )
    parser.add_argument(
        "--scaler", type=str, default=SCALER_PATH,
        help=f"Path to scaler pickle (default: {SCALER_PATH})"
    )
    parser.add_argument(
        "--target-col", type=int, default=0,
        help="Column index of PM10 in the scaler feature matrix (default: 0)"
    )
    args = parser.parse_args()

    if args.demo:
        _demo(args.checkpoint, args.scaler)
        return

    # -- Real-input path -------------------------------------------------------
    model  = load_inference_model(args.checkpoint)
    scaler = load_scaler(args.scaler)

    window = load_input_window(args.input)
    result = predict_pm10(window, model, scaler,
                          target_col_index=args.target_col)

    print()
    print("=" * 50)
    print("  Inference Result")
    print("=" * 50)
    print(f"  Predicted PM10   : {result['prediction_pm10']} ug/m3")
    print(f"  Scaled output    : {result['prediction_scaled']}")
    print(f"  Attention weights: [{', '.join(f'{w:.3f}' for w in result['attention_weights'][:6])} ...]")
    print("=" * 50)


if __name__ == "__main__":
    main()
