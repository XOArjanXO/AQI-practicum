"""
models/cnn_lstm_attention.py
─────────────────────────────
Step 7 — CNN → LSTM → Attention → Regression head

Architecture
────────────
Input  : (batch, TIME_STEPS, num_features)   e.g. (32, 24, 7)
         │
  ┌──────▼──────┐
  │  Conv1d     │  Extract local temporal patterns across the feature axis
  │  BatchNorm  │
  │  GELU       │
  │  Dropout    │
  └──────┬──────┘
         │ (batch, TIME_STEPS, CNN_OUT_CHANNELS)
  ┌──────▼──────┐
  │  LSTM       │  Capture long-range sequential dependencies
  │  (stacked)  │
  └──────┬──────┘
         │ (batch, TIME_STEPS, LSTM_HIDDEN_SIZE)
  ┌──────▼──────────────┐
  │  Scaled Dot-Product │  Learn which time-steps matter most
  │  Self-Attention     │
  └──────┬──────────────┘
         │ context vector  (batch, LSTM_HIDDEN_SIZE)
  ┌──────▼──────┐
  │  FC head    │  Map attended representation → scalar PM10 prediction
  └──────┬──────┘
         ▼
  output : (batch, 1)
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

# ── Allow running as a standalone script from any cwd ─────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    TIME_STEPS,
    FEATURES,
    CNN_OUT_CHANNELS,
    CNN_KERNEL_SIZE,
    LSTM_HIDDEN_SIZE,
    LSTM_NUM_LAYERS,
    DROPOUT,
)


# ─────────────────────────────────────────────────────────────────────────────
# Scaled Dot-Product Self-Attention
# ─────────────────────────────────────────────────────────────────────────────

class ScaledDotProductAttention(nn.Module):
    """
    Single-head scaled dot-product self-attention.

    Parameters
    ----------
    hidden_size : int
        Dimensionality of the LSTM output (= query / key / value dim).
    dropout     : float
        Dropout applied to attention weights.
    """

    def __init__(self, hidden_size: int, dropout: float = 0.0):
        super().__init__()
        self.scale   = hidden_size ** 0.5
        self.dropout = nn.Dropout(dropout)

        # Learnable projections for Q, K, V
        self.W_q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_v = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(
        self,
        x: torch.Tensor,                       # (B, T, H)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : Tensor of shape (batch, seq_len, hidden_size)

        Returns
        -------
        context       : Tensor (batch, hidden_size) — weighted sum over time
        attn_weights  : Tensor (batch, seq_len)     — softmax scores
        """
        Q = self.W_q(x)                        # (B, T, H)
        K = self.W_k(x)                        # (B, T, H)
        V = self.W_v(x)                        # (B, T, H)

        # Energy: (B, T, T) via batch matrix multiply
        energy  = torch.bmm(Q, K.transpose(1, 2)) / self.scale
        attn    = F.softmax(energy, dim=-1)    # (B, T, T)
        attn    = self.dropout(attn)

        # Weighted sum of values → (B, T, H)
        attended = torch.bmm(attn, V)          # (B, T, H)

        # Collapse time dimension: mean pooling over attended values
        context  = attended.mean(dim=1)        # (B, H)

        # Return per-timestep weights (mean over query positions)
        attn_weights = attn.mean(dim=1)        # (B, T)

        return context, attn_weights


# ─────────────────────────────────────────────────────────────────────────────
# Main Model
# ─────────────────────────────────────────────────────────────────────────────

class CNNLSTMAttention(nn.Module):
    """
    CNN → LSTM → Self-Attention → FC regression model.

    Parameters
    ----------
    num_features    : int   — number of input feature channels (default: len(FEATURES))
    time_steps      : int   — input sequence length            (default: TIME_STEPS)
    cnn_out_ch      : int   — Conv1d output channels           (default: CNN_OUT_CHANNELS)
    cnn_kernel      : int   — Conv1d kernel size               (default: CNN_KERNEL_SIZE)
    lstm_hidden     : int   — LSTM hidden units                (default: LSTM_HIDDEN_SIZE)
    lstm_layers     : int   — number of stacked LSTM layers    (default: LSTM_NUM_LAYERS)
    dropout         : float — dropout rate                     (default: DROPOUT)
    """

    def __init__(
        self,
        num_features : int   = len(FEATURES),
        time_steps   : int   = TIME_STEPS,
        cnn_out_ch   : int   = CNN_OUT_CHANNELS,
        cnn_kernel   : int   = CNN_KERNEL_SIZE,
        lstm_hidden  : int   = LSTM_HIDDEN_SIZE,
        lstm_layers  : int   = LSTM_NUM_LAYERS,
        dropout      : float = DROPOUT,
    ):
        super().__init__()
        self.time_steps  = time_steps
        self.lstm_hidden = lstm_hidden

        # ── 1. CNN Block ──────────────────────────────────────────────────────
        # Conv1d expects (batch, channels, length); we treat features as channels.
        padding = cnn_kernel // 2               # 'same' padding to preserve length
        self.cnn_block = nn.Sequential(
            nn.Conv1d(
                in_channels  = num_features,
                out_channels = cnn_out_ch,
                kernel_size  = cnn_kernel,
                padding      = padding,
            ),
            nn.BatchNorm1d(cnn_out_ch),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ── 2. LSTM Block ─────────────────────────────────────────────────────
        self.lstm = nn.LSTM(
            input_size   = cnn_out_ch,
            hidden_size  = lstm_hidden,
            num_layers   = lstm_layers,
            batch_first  = True,
            dropout      = dropout if lstm_layers > 1 else 0.0,
        )
        self.lstm_dropout = nn.Dropout(dropout)

        # ── 3. Attention Block ────────────────────────────────────────────────
        self.attention = ScaledDotProductAttention(lstm_hidden, dropout)

        # ── 4. Fully-Connected Regression Head ────────────────────────────────
        self.fc_head = nn.Sequential(
            nn.LayerNorm(lstm_hidden),
            nn.Linear(lstm_hidden, lstm_hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden // 2, 1),
        )

    # ── Forward pass ──────────────────────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,                       # (B, T, F)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : Tensor  (batch_size, time_steps, num_features)

        Returns
        -------
        out          : Tensor (batch_size, 1)    — PM10 prediction
        attn_weights : Tensor (batch_size, T)    — attention scores (for XAI)
        """
        B, T, F = x.shape

        # ── CNN: expects (B, F, T) → output (B, C, T) → back to (B, T, C) ──
        cnn_out = self.cnn_block(x.permute(0, 2, 1))  # (B, C, T)
        cnn_out = cnn_out.permute(0, 2, 1)             # (B, T, C)

        # ── LSTM ──────────────────────────────────────────────────────────────
        lstm_out, _ = self.lstm(cnn_out)               # (B, T, H)
        lstm_out    = self.lstm_dropout(lstm_out)

        # ── Attention ─────────────────────────────────────────────────────────
        context, attn_weights = self.attention(lstm_out)  # (B, H), (B, T)

        # ── FC Head ───────────────────────────────────────────────────────────
        out = self.fc_head(context)                    # (B, 1)

        return out, attn_weights

    # ── Convenience ───────────────────────────────────────────────────────────

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return only predictions (no attention weights). Useful at inference."""
        out, _ = self.forward(x)
        return out

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# Factory helper
# ─────────────────────────────────────────────────────────────────────────────

def build_model(num_features: int = len(FEATURES), **kwargs) -> CNNLSTMAttention:
    """
    Instantiate and return a CNNLSTMAttention model.

    Parameters
    ----------
    num_features : actual number of features in the dataset
                   (may be > len(FEATURES) if extra engineered cols were added).
    **kwargs     : any CNNLSTMAttention constructor arguments to override config defaults.
    """
    model = CNNLSTMAttention(num_features=num_features, **kwargs)
    print(f"[model] CNNLSTMAttention built — "
          f"{model.count_parameters():,} trainable parameters.")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Script entry-point — shape verification with dummy input
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    torch.manual_seed(42)

    # ── Dummy input as specified in the spec ─────────────────────────────────
    x = torch.randn(32, 24, 7)              # (batch=32, time_steps=24, features=7)

    model = build_model(num_features=7)
    model.eval()

    with torch.no_grad():
        out, attn = model(x)

    print("\n── Shape Verification ───────────────────────────────────")
    print(f"  Input          : {tuple(x.shape)}")
    print(f"  Output         : {tuple(out.shape)}   ← expected (32, 1)")
    print(f"  Attention map  : {tuple(attn.shape)}  ← expected (32, 24)")
    print("─────────────────────────────────────────────────────────")

    assert out.shape  == (32, 1),  f"Output shape mismatch: {out.shape}"
    assert attn.shape == (32, 24), f"Attention shape mismatch: {attn.shape}"
    print("  All shape assertions PASSED ✓")

    print(f"\n  Trainable parameters : {model.count_parameters():,}")
    print("\n── Model Summary ────────────────────────────────────────")
    print(model)
