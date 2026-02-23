"""
xai/attention_visualizer.py
-----------------------------
Step 12 -- Visualise which time steps the model attends to most.

Input  : one sample window  (TIME_STEPS, num_features)
Output : bar chart saved to  xai/plots/attention_visualization.png
"""

import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import (
    TIME_STEPS,
    FEATURES,
    CHECKPOINT_PATH,
    SCALER_PATH,
    XAI_PLOTS_DIR,
)
from models.cnn_lstm_attention import build_model
from inference import load_inference_model, load_scaler, scale_window, validate_window


# -----------------------------------------------------------------------------
# Core: extract attention weights
# -----------------------------------------------------------------------------

def get_attention_weights(
    window: np.ndarray,
    model: torch.nn.Module,
    scaler,
) -> np.ndarray:
    """
    Forward-pass one window through the model and return its attention weights.

    Parameters
    ----------
    window : (TIME_STEPS, num_features) raw (un-scaled) numpy array.
    model  : trained model in eval mode.
    scaler : fitted MinMaxScaler.

    Returns
    -------
    attn : (TIME_STEPS,) float32 array -- softmax attention scores per step.
    """
    validate_window(window)
    scaled = scale_window(window, scaler)                    # (T, F)
    x      = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0)  # (1, T, F)

    model.eval()
    with torch.no_grad():
        _, attn_tensor = model(x)                            # (1, T)

    return attn_tensor.squeeze(0).cpu().numpy()              # (T,)


# -----------------------------------------------------------------------------
# Visualisation
# -----------------------------------------------------------------------------

def plot_attention(
    attn_weights : np.ndarray,
    window       : np.ndarray = None,
    save_path    : str        = None,
    title        : str        = "Attention Weights — Which Time Steps Matter",
    show         : bool       = False,
) -> str:
    """
    Create a two-panel figure:
      Top    : bar chart of attention scores per time step
      Bottom : raw feature lines overlay (if window is provided)

    Parameters
    ----------
    attn_weights : (TIME_STEPS,) attention scores.
    window       : optional (TIME_STEPS, num_features) raw window for overlay.
    save_path    : where to save the PNG (defaults to xai/plots/).
    title        : figure title.
    show         : whether to call plt.show().

    Returns
    -------
    str : absolute path to the saved PNG.
    """
    if save_path is None:
        plots_dir = ROOT / XAI_PLOTS_DIR
        plots_dir.mkdir(parents=True, exist_ok=True)
        save_path = str(plots_dir / "attention_visualization.png")

    T     = len(attn_weights)
    steps = np.arange(T)

    # Normalise so weights sum to 1 (they should already, but just in case)
    weights = attn_weights / (attn_weights.sum() + 1e-9)

    # Colour map: darker = higher attention
    colours = plt.cm.YlOrRd(weights / (weights.max() + 1e-9))

    has_window = window is not None and window.ndim == 2

    fig, axes = plt.subplots(
        2 if has_window else 1, 1,
        figsize=(14, 8 if has_window else 4),
        facecolor="#0f1117",
    )
    if not has_window:
        axes = [axes]

    # ---- Panel 1: Attention bar chart ----------------------------------------
    ax1 = axes[0]
    ax1.set_facecolor("#1a1d2e")
    bars = ax1.bar(steps, weights, color=colours, edgecolor="none", width=0.85)

    # Annotate top-5 most-attended steps
    top5 = np.argsort(weights)[-5:]
    for idx in top5:
        ax1.text(
            idx, weights[idx] + 0.001,
            f"t-{T - idx - 1}",
            ha="center", va="bottom",
            fontsize=7, color="white", fontweight="bold",
        )

    ax1.set_xlabel("Time Step", color="white")
    ax1.set_ylabel("Attention Score", color="white")
    ax1.set_title(title, color="white", fontsize=13, pad=10)
    ax1.tick_params(colors="white")
    for spine in ax1.spines.values():
        spine.set_edgecolor("#444")
    ax1.set_xlim(-0.5, T - 0.5)
    ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))

    # ---- Panel 2: Raw feature lines (optional) --------------------------------
    if has_window:
        ax2 = axes[1]
        ax2.set_facecolor("#1a1d2e")
        feature_labels = FEATURES[:window.shape[1]]
        for fi, label in enumerate(feature_labels):
            ax2.plot(steps, window[:, fi], alpha=0.7, label=label, linewidth=1.3)

        # Shade top-attended regions
        for idx in top5:
            ax2.axvspan(idx - 0.4, idx + 0.4, alpha=0.15,
                        color="yellow", linewidth=0)

        ax2.set_xlabel("Time Step", color="white")
        ax2.set_ylabel("Raw Feature Value", color="white")
        ax2.set_title("Input Features with High-Attention Steps Highlighted",
                      color="white", fontsize=10)
        ax2.tick_params(colors="white")
        for spine in ax2.spines.values():
            spine.set_edgecolor("#444")
        ax2.legend(
            facecolor="#1a1d2e", edgecolor="#444",
            labelcolor="white", fontsize=7, ncol=4,
        )

    plt.tight_layout(pad=2.0)
    plt.savefig(save_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[attention_visualizer] Saved -> '{save_path}'")
    if show:
        plt.show()
    return save_path


# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------

def explain_sample(
    window          : np.ndarray,
    checkpoint_path : str = CHECKPOINT_PATH,
    scaler_path     : str = SCALER_PATH,
    save_path       : str = None,
    show            : bool = False,
) -> dict:
    """
    End-to-end attention explanation for one input window.

    Parameters
    ----------
    window : (TIME_STEPS, num_features) raw numpy array.

    Returns
    -------
    dict with keys:
        'attention_weights'  : (TIME_STEPS,) numpy array
        'top_timesteps'      : list of (step_index, score) for top-5 steps
        'plot_path'          : path to saved PNG
    """
    num_features = window.shape[1]
    model  = load_inference_model(checkpoint_path, num_features)
    scaler = load_scaler(scaler_path)

    attn   = get_attention_weights(window, model, scaler)
    top5   = sorted(
        [(int(i), float(attn[i])) for i in np.argsort(attn)[-5:]],
        key=lambda x: x[1], reverse=True,
    )

    path = plot_attention(attn, window=window, save_path=save_path, show=show)

    print("\n[attention_visualizer] Top-5 attended time steps:")
    for rank, (idx, score) in enumerate(top5, 1):
        print(f"  {rank}. step t-{len(attn) - idx - 1:02d}  "
              f"(index {idx:2d})  score={score:.5f}")

    return {
        "attention_weights" : attn,
        "top_timesteps"     : top5,
        "plot_path"         : path,
    }


# -----------------------------------------------------------------------------
# Script entry-point
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import numpy as np
    from pathlib import Path
    from config import PROCESSED_DIR
    np.random.seed(42)
    # Use actual feature count from training data
    val_path = ROOT / PROCESSED_DIR / 'X_val.npy'
    if val_path.exists():
        _X_val = np.load(val_path)
        num_features = _X_val.shape[2]
        # Use a real sample window (first sample from val set)
        window = _X_val[0]  # (TIME_STEPS, num_features)
        print(f"[attention_visualizer] Using real val window shape={window.shape}")
    else:
        num_features = len(FEATURES)
        window = np.random.uniform(0, 1, (TIME_STEPS, num_features)).astype('float32')
    result = explain_sample(window)
    print(f"\nPlot saved to: {result['plot_path']}")
