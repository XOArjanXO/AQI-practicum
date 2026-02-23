"""
models/training_utils.py
-------------------------
Step 8 — Training loop, optimizer, loss function, and LR scheduler.

Responsibilities
----------------
•  build_optimizer(model)       → AdamW optimiser
•  build_scheduler(optimizer)   → CosineAnnealingLR
•  train_one_epoch(...)         → one full pass over training batches
•  evaluate(...)                → one full pass over validation batches
•  train(...)                   → full training loop with early-stopping
•  log_metrics(...)             → append line to training_log.txt
"""

import sys
import os
import time
import json
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from typing import Optional

# ── Allow running as a standalone script from any cwd ─────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    LR,
    EPOCHS,
    BATCH_SIZE,
    CHECKPOINT_PATH,
    TRAINING_LOG_PATH,
    METRICS_PATH,
    RANDOM_SEED,
)


# ─────────────────────────────────────────────────────────────────────────────
# Setup helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    """Return CUDA if available, else CPU."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[training] Using device: {device}")
    return device


def build_optimizer(model: nn.Module, lr: float = LR) -> torch.optim.Optimizer:
    """
    AdamW with weight-decay regularisation.

    Parameters
    ----------
    model : nn.Module — the model whose parameters will be optimised.
    lr    : float     — initial learning rate (default from config).

    Returns
    -------
    torch.optim.AdamW
    """
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = lr,
        weight_decay = 1e-4,
    )
    print(f"[training] Optimizer: AdamW  |  lr={lr}  |  weight_decay=1e-4")
    return optimizer


def build_scheduler(
    optimizer : torch.optim.Optimizer,
    epochs    : int = EPOCHS,
) -> torch.optim.lr_scheduler.CosineAnnealingLR:
    """
    CosineAnnealingLR — smoothly decays LR to eta_min over *epochs* steps.

    Parameters
    ----------
    optimizer : the AdamW optimiser.
    epochs    : total training epochs (= T_max for cosine schedule).

    Returns
    -------
    torch.optim.lr_scheduler.CosineAnnealingLR
    """
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max   = epochs,
        eta_min = LR * 1e-2,        # decay down to 1 % of initial LR
    )
    print(f"[training] Scheduler: CosineAnnealingLR  |  T_max={epochs}  |  eta_min={LR*1e-2:.2e}")
    return scheduler


def build_dataloaders(
    X_train : 'torch.Tensor',
    y_train : 'torch.Tensor',
    X_val   : 'torch.Tensor',
    y_val   : 'torch.Tensor',
    batch_size : int = BATCH_SIZE,
) -> tuple[DataLoader, DataLoader]:
    """
    Wrap numpy arrays (or tensors) into PyTorch DataLoaders.

    Returns
    -------
    (train_loader, val_loader)
    """
    def _to_tensor(a):
        if not isinstance(a, torch.Tensor):
            return torch.tensor(a, dtype=torch.float32)
        return a.float()

    train_ds = TensorDataset(_to_tensor(X_train), _to_tensor(y_train))
    val_ds   = TensorDataset(_to_tensor(X_val),   _to_tensor(y_val))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=False)

    print(f"[training] DataLoaders ready  |  "
          f"train batches={len(train_loader)}  val batches={len(val_loader)}")
    return train_loader, val_loader


# ─────────────────────────────────────────────────────────────────────────────
# Per-epoch helpers
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model       : nn.Module,
    loader      : DataLoader,
    optimizer   : torch.optim.Optimizer,
    criterion   : nn.Module,
    device      : torch.device,
    grad_clip   : float = 1.0,
) -> float:
    """
    One full training pass.

    Parameters
    ----------
    grad_clip   : max gradient norm (applied via clip_grad_norm_).

    Returns
    -------
    mean training loss over all batches.
    """
    model.train()
    total_loss = 0.0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)              # (B, T, F)
        y_batch = y_batch.to(device).unsqueeze(1) # (B, 1)

        optimizer.zero_grad()
        preds, _ = model(X_batch)                 # (B, 1), ignore attn here
        loss     = criterion(preds, y_batch)
        loss.backward()

        # Gradient clipping — prevents exploding gradients in LSTM
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(
    model     : nn.Module,
    loader    : DataLoader,
    criterion : nn.Module,
    device    : torch.device,
) -> tuple[float, float]:
    """
    One full validation pass.

    Returns
    -------
    (val_loss_mse, val_rmse)
    """
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device).unsqueeze(1)
            preds, _ = model(X_batch)
            loss     = criterion(preds, y_batch)
            total_loss += loss.item()

    val_mse  = total_loss / len(loader)
    val_rmse = math.sqrt(val_mse)
    return val_mse, val_rmse


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

def log_metrics(
    epoch      : int,
    train_loss : float,
    val_loss   : float,
    val_rmse   : float,
    lr         : float,
    elapsed    : float,
    log_path   : str = TRAINING_LOG_PATH,
) -> None:
    """Append one epoch's metrics to the training log file."""
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    line = (
        f"Epoch {epoch:04d} | "
        f"Train Loss: {train_loss:.6f} | "
        f"Val Loss: {val_loss:.6f} | "
        f"Val RMSE: {val_rmse:.6f} | "
        f"LR: {lr:.2e} | "
        f"Time: {elapsed:.1f}s\n"
    )
    with open(log_path, 'a') as f:
        f.write(line)


# ─────────────────────────────────────────────────────────────────────────────
# Full training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(
    model            : nn.Module,
    train_loader     : DataLoader,
    val_loader       : DataLoader,
    epochs           : int   = EPOCHS,
    checkpoint_path  : str   = CHECKPOINT_PATH,
    patience         : int   = 10,       # early-stopping patience
) -> dict:
    """
    Full training loop with early-stopping and model checkpointing.

    Parameters
    ----------
    model           : CNNLSTMAttention (or any nn.Module compatible model).
    train_loader    : training DataLoader.
    val_loader      : validation DataLoader.
    epochs          : maximum number of epochs.
    checkpoint_path : where to save the best model weights.
    patience        : early-stopping — stop if val_loss doesn't improve
                      for *patience* consecutive epochs.

    Returns
    -------
    history : dict containing lists of train_losses, val_losses, val_rmses.
    """
    torch.manual_seed(RANDOM_SEED)

    device    = get_device()
    model     = model.to(device)

    optimizer = build_optimizer(model)
    scheduler = build_scheduler(optimizer, epochs)
    criterion = nn.MSELoss()

    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)

    history = {
        'train_loss' : [],
        'val_loss'   : [],
        'val_rmse'   : [],
    }

    best_val_loss   = float('inf')
    epochs_no_improve = 0

    print("\n" + "=" * 65)
    print(f"  Starting training  |  max_epochs={epochs}  |  patience={patience}")
    print("=" * 65)

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_loss           = train_one_epoch(model, train_loader, optimizer,
                                               criterion, device)
        val_loss, val_rmse   = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        cur_lr  = scheduler.get_last_lr()[0]

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_rmse'].append(val_rmse)

        log_metrics(epoch, train_loss, val_loss, val_rmse, cur_lr, elapsed)

        print(f"  Epoch {epoch:03d}/{epochs} | "
              f"Train {train_loss:.5f} | "
              f"Val {val_loss:.5f} | "
              f"RMSE {val_rmse:.5f} | "
              f"LR {cur_lr:.2e} | "
              f"{elapsed:.1f}s")

        # ── Checkpointing ─────────────────────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  ** Best model saved -> '{checkpoint_path}' (val_loss={best_val_loss:.6f})")
        else:
            epochs_no_improve += 1

        # ── Early stopping ────────────────────────────────────────────────────
        if epochs_no_improve >= patience:
            print(f"\n  Early stopping triggered at epoch {epoch} "
                  f"(no improvement for {patience} epochs).")
            break

    print("=" * 65)
    print(f"  Training complete. Best val_loss = {best_val_loss:.6f}")
    print("=" * 65 + "\n")

    return history
