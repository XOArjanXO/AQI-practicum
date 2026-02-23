"""
train.py
─────────
Step 9 — Main training entry-point.

Workflow
--------
1.  Load .npy arrays from data/processed/
2.  Build PyTorch DataLoaders
3.  Initialise CNNLSTMAttention model
4.  Run training loop (training_utils.train)
5.  Save final model state dict
    → models/model_checkpoint.pth
6.  Persist training metrics
    → results/metrics.json
"""

import sys
import json
import numpy as np
import torch
from pathlib import Path

# ── Project root on sys.path ───────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config import (
    PROCESSED_DIR,
    CHECKPOINT_PATH,
    METRICS_PATH,
    EPOCHS,
    BATCH_SIZE,
    RANDOM_SEED,
)
from models.cnn_lstm_attention import build_model
from models.training_utils import (
    build_dataloaders,
    train,
    get_device,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_processed_data(
    processed_dir: str = PROCESSED_DIR,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load pre-built .npy arrays from disk.

    Returns
    -------
    (X_train, y_train, X_val, y_val)  -- float32 numpy arrays.
    """
    d = Path(processed_dir)
    X_train = np.load(d / 'X_train.npy')
    y_train = np.load(d / 'y_train.npy')
    X_val   = np.load(d / 'X_val.npy')
    y_val   = np.load(d / 'y_val.npy')

    print("[train] Loaded processed arrays:")
    print(f"  X_train : {X_train.shape}  y_train : {y_train.shape}")
    print(f"  X_val   : {X_val.shape}  y_val   : {y_val.shape}")

    return X_train, y_train, X_val, y_val


def save_metrics(history: dict, metrics_path: str = METRICS_PATH) -> None:
    """Persist training history as JSON for later evaluation / plotting."""
    Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'epochs_trained' : len(history['train_loss']),
        'best_val_loss'  : float(min(history['val_loss'])),
        'best_val_rmse'  : float(min(history['val_rmse'])),
        'train_loss'     : history['train_loss'],
        'val_loss'       : history['val_loss'],
        'val_rmse'       : history['val_rmse'],
    }
    with open(metrics_path, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f"[train] Metrics saved -> '{metrics_path}'")


def save_final_model(model: torch.nn.Module,
                     checkpoint_path: str = CHECKPOINT_PATH) -> None:
    """
    Save the model's state_dict as specified in the framework.

    Note: training_utils.train() already saves the *best* checkpoint during
    training; this call saves the model state at the very end of training
    (last epoch or early-stop epoch) as an additional artefact.
    """
    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
    print(f"[train] Final model state_dict saved -> '{checkpoint_path}'")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print("=" * 65)
    print("  XAI Air Quality — Training Script")
    print("=" * 65)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    X_train, y_train, X_val, y_val = load_processed_data()

    # Infer actual number of features from loaded data
    num_features = X_train.shape[2]       # (samples, time_steps, features)
    print(f"[train] num_features inferred from data: {num_features}")

    # ── 2. DataLoaders ────────────────────────────────────────────────────────
    train_loader, val_loader = build_dataloaders(
        X_train, y_train, X_val, y_val,
        batch_size=BATCH_SIZE,
    )

    # ── 3. Model ──────────────────────────────────────────────────────────────
    model = build_model(num_features=num_features)

    # ── 4. Train ──────────────────────────────────────────────────────────────
    history = train(
        model           = model,
        train_loader    = train_loader,
        val_loader      = val_loader,
        epochs          = EPOCHS,
        checkpoint_path = CHECKPOINT_PATH,   # best val model saved here
        patience        = 10,
    )

    # ── 5. Save final state_dict (framework requirement) ─────────────────────
    torch.save(model.state_dict(), CHECKPOINT_PATH)
    print(f"[train] model.state_dict() saved -> '{CHECKPOINT_PATH}'")

    # ── 6. Persist metrics ────────────────────────────────────────────────────
    save_metrics(history)

    print("\n  Done. Artefacts written:")
    print(f"    Checkpoint : {CHECKPOINT_PATH}")
    print(f"    Metrics    : {METRICS_PATH}")
    print(f"    Train log  : results/training_log.txt")


if __name__ == '__main__':
    main()
