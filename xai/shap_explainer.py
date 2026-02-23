"""
xai/shap_explainer.py
-----------------------
Step 13 -- SHAP-based global & temporal explanations.

Outputs
-------
1. xai/plots/shap_global_importance.png  -- mean |SHAP| per feature (bar)
2. xai/plots/shap_summary_plot.png       -- beeswarm summary plot
3. xai/plots/shap_temporal_heatmap.png   -- feature x time-step SHAP heatmap
"""

import sys
import pickle
import warnings
import numpy as np
import torch
import shap
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import (
    TIME_STEPS,
    FEATURES,
    CHECKPOINT_PATH,
    SCALER_PATH,
    PROCESSED_DIR,
    XAI_PLOTS_DIR,
)
from models.cnn_lstm_attention import build_model
from inference import load_inference_model, load_scaler, scale_window

warnings.filterwarnings("ignore")

PLOTS_DIR = ROOT / XAI_PLOTS_DIR


# -----------------------------------------------------------------------------
# SHAP-compatible wrapper
# -----------------------------------------------------------------------------

class FlatModelWrapper:
    """
    Wraps the CNN-LSTM-Attention model so SHAP sees it as:
        f(X_flat) -> prediction

    Input  : (N, TIME_STEPS * num_features)   flat 2-D numpy array
    Output : (N,)                             scalar PM10 predictions
    """

    def __init__(self, model: torch.nn.Module, time_steps: int, num_features: int):
        self.model       = model
        self.time_steps  = time_steps
        self.num_features = num_features

    def predict(self, X_flat: np.ndarray) -> np.ndarray:
        """Called by shap.KernelExplainer."""
        X_3d = X_flat.reshape(-1, self.time_steps, self.num_features).astype(np.float32)
        x    = torch.tensor(X_3d, dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            preds, _ = self.model(x)               # (N, 1)
        return preds.squeeze(1).cpu().numpy()      # (N,)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _get_feature_time_labels(features, time_steps: int):
    """Return flat column names like 'PM10_lag@t0', 'Temperature@t1', ..."""
    return [f"{f}@t{t}" for t in range(time_steps) for f in features]


def _load_background(
    X_val   : np.ndarray,
    scaler,
    n_bg    : int = 50,
) -> np.ndarray:
    """
    Sample *n_bg* rows from X_val as SHAP background dataset.
    Returns flat 2-D array (n_bg, T*F).
    """
    idx = np.random.choice(len(X_val), size=min(n_bg, len(X_val)), replace=False)
    bg  = X_val[idx]                                    # (n_bg, T, F)
    return bg.reshape(len(bg), -1)                      # (n_bg, T*F)


# -----------------------------------------------------------------------------
# Plot 1: Global feature importance (mean |SHAP| per feature)
# -----------------------------------------------------------------------------

def plot_global_importance(
    shap_values  : np.ndarray,
    feature_names: list,
    num_features : int,
    time_steps   : int,
    save_path    : str = None,
) -> str:
    """
    Collapse time dimension: mean |SHAP| per feature across all time steps.

    Parameters
    ----------
    shap_values   : (N_explain, T*F) SHAP value array.
    feature_names : list of base feature names (length F).
    """
    if save_path is None:
        save_path = str(PLOTS_DIR / "shap_global_importance.png")

    # Reshape and average over time steps
    sv = shap_values.reshape(-1, time_steps, num_features)  # (N, T, F)
    mean_abs = np.abs(sv).mean(axis=(0, 1))                # (F,)

    order   = np.argsort(mean_abs)[::-1]
    colours = plt.cm.plasma(np.linspace(0.2, 0.9, num_features))

    fig, ax = plt.subplots(figsize=(9, 5), facecolor="#0f1117")
    ax.set_facecolor("#1a1d2e")

    bars = ax.barh(
        [feature_names[i] for i in order],
        mean_abs[order],
        color=[colours[i] for i in range(num_features)],
        edgecolor="none",
        height=0.65,
    )
    ax.bar_label(bars, fmt="%.4f", padding=4, color="white", fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP value|", color="white")
    ax.set_title("Global Feature Importance (SHAP)", color="white", fontsize=13)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")

    plt.tight_layout(pad=2.0)
    plt.savefig(save_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[shap_explainer] Global importance saved -> '{save_path}'")
    return save_path


# -----------------------------------------------------------------------------
# Plot 2: SHAP beeswarm / summary plot
# -----------------------------------------------------------------------------

def plot_shap_summary(
    shap_values  : np.ndarray,
    X_explain    : np.ndarray,
    feature_names: list,
    num_features : int,
    time_steps   : int,
    save_path    : str = None,
) -> str:
    """
    Classic SHAP beeswarm (dot plot) showing feature impact distribution.
    Aggregates across time by using mean SHAP score per feature per sample.
    """
    if save_path is None:
        save_path = str(PLOTS_DIR / "shap_summary_plot.png")

    sv = shap_values.reshape(-1, time_steps, num_features)   # (N, T, F)
    sv_collapsed = sv.mean(axis=1)                            # (N, F)   mean over time

    x_collapsed  = X_explain.reshape(-1, time_steps, num_features).mean(axis=1)  # (N, F)

    plt.figure(figsize=(10, 6), facecolor="#0f1117")
    plt.gca().set_facecolor("#1a1d2e")

    shap.summary_plot(
        sv_collapsed,
        features       = x_collapsed,
        feature_names  = feature_names,
        show           = False,
        plot_size      = None,
        color_bar      = True,
    )

    ax = plt.gca()
    ax.set_facecolor("#1a1d2e")
    ax.tick_params(colors="white")
    ax.set_xlabel("SHAP value (impact on model output)", color="white")
    ax.set_title("SHAP Summary Plot", color="white", fontsize=13)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")

    plt.tight_layout(pad=2.0)
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor="#0f1117")
    plt.close()
    print(f"[shap_explainer] Summary plot saved -> '{save_path}'")
    return save_path


# -----------------------------------------------------------------------------
# Plot 3: Temporal heatmap  (time steps x features)
# -----------------------------------------------------------------------------

def plot_temporal_heatmap(
    shap_values  : np.ndarray,
    feature_names: list,
    num_features : int,
    time_steps   : int,
    save_path    : str = None,
) -> str:
    """
    Heatmap of mean |SHAP| per (time_step, feature) pair.
    Rows = features, Columns = time steps.
    """
    if save_path is None:
        save_path = str(PLOTS_DIR / "shap_temporal_heatmap.png")

    sv = shap_values.reshape(-1, time_steps, num_features)  # (N, T, F)
    heat = np.abs(sv).mean(axis=0).T                        # (F, T)

    fig, ax = plt.subplots(
        figsize=(max(10, time_steps // 2), max(4, num_features)),
        facecolor="#0f1117",
    )
    ax.set_facecolor("#1a1d2e")

    im = ax.imshow(
        heat,
        aspect="auto",
        cmap="magma",
        interpolation="nearest",
    )

    ax.set_yticks(range(num_features))
    ax.set_yticklabels(feature_names, color="white", fontsize=9)
    ax.set_xticks(range(0, time_steps, max(1, time_steps // 12)))
    ax.set_xticklabels(
        [f"t-{time_steps - i - 1}" for i in range(0, time_steps, max(1, time_steps // 12))],
        color="white", fontsize=8, rotation=45, ha="right",
    )
    ax.set_xlabel("Time Step  (t-0 = most recent)", color="white")
    ax.set_ylabel("Feature", color="white")
    ax.set_title("SHAP Temporal Heatmap  (mean |SHAP| per feature x time step)",
                 color="white", fontsize=12)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.ax.yaxis.set_tick_params(color="white")
    cbar.set_label("Mean |SHAP|", color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    plt.tight_layout(pad=2.0)
    plt.savefig(save_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[shap_explainer] Temporal heatmap saved -> '{save_path}'")
    return save_path


# -----------------------------------------------------------------------------
# Main SHAP pipeline
# -----------------------------------------------------------------------------

def run_shap_explanation(
    X_val           : np.ndarray,
    checkpoint_path : str = CHECKPOINT_PATH,
    scaler_path     : str = SCALER_PATH,
    n_background    : int = 50,
    n_explain       : int = 30,
    seed            : int = 42,
) -> dict:
    """
    Full SHAP explanation pipeline.

    Parameters
    ----------
    X_val        : (N, TIME_STEPS, num_features)  scaled validation array.
    n_background : number of background samples for KernelExplainer.
    n_explain    : number of samples to explain (more = slower but smoother).

    Returns
    -------
    dict with keys:
        shap_values     : (n_explain, T*F) raw SHAP values
        feature_names   : base feature names used
        global_path     : path to global importance PNG
        summary_path    : path to summary plot PNG
        heatmap_path    : path to temporal heatmap PNG
    """
    np.random.seed(seed)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    num_features = X_val.shape[2]
    time_steps   = X_val.shape[1]
    # Build full feature name list matching engineered features
    base_features = list(FEATURES)
    engineered = ['PM10_lag2', 'PM10_lag3', 'PM10_rolling_mean_24', 'PM10_rolling_std_24',
                  'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos']
    all_features = base_features + [f for f in engineered if len(base_features) < num_features]
    feature_names = all_features[:num_features]

    # ---- Load model & scaler ------------------------------------------------
    model  = load_inference_model(checkpoint_path, num_features)
    scaler = load_scaler(scaler_path)

    # ---- Wrap model for SHAP ------------------------------------------------
    wrapper = FlatModelWrapper(model, time_steps, num_features)

    # ---- Background & explanation sets (flat) --------------------------------
    bg_flat      = _load_background(X_val, scaler, n_bg=n_background)
    explain_idx  = np.random.choice(len(X_val),
                                    size=min(n_explain, len(X_val)),
                                    replace=False)
    X_explain_3d = X_val[explain_idx]                        # (n_exp, T, F)
    X_flat       = X_explain_3d.reshape(len(explain_idx), -1) # (n_exp, T*F)

    # ---- KernelExplainer ---------------------------------------------------
    print(f"[shap_explainer] Fitting KernelExplainer "
          f"(background={len(bg_flat)}, explain={len(X_flat)}) ...")
    explainer   = shap.KernelExplainer(wrapper.predict, bg_flat)
    shap_values = explainer.shap_values(X_flat, nsamples=100, silent=True)
    # shap_values: (n_explain, T*F)

    print(f"[shap_explainer] SHAP values computed: shape={shap_values.shape}")

    # ---- Plots -------------------------------------------------------------
    p1 = plot_global_importance(shap_values, feature_names, num_features, time_steps)
    p2 = plot_shap_summary(shap_values, X_flat, feature_names, num_features, time_steps)
    p3 = plot_temporal_heatmap(shap_values, feature_names, num_features, time_steps)

    return {
        "shap_values"   : shap_values,
        "feature_names" : feature_names,
        "global_path"   : p1,
        "summary_path"  : p2,
        "heatmap_path"  : p3,
    }


# -----------------------------------------------------------------------------
# Script entry-point
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import numpy as np

    # Load real val data if it exists, otherwise use synthetic
    val_path = ROOT / PROCESSED_DIR / "X_val.npy"
    if val_path.exists():
        X_val = np.load(val_path)
        print(f"[shap_explainer] Loaded X_val {X_val.shape}")
    else:
        print("[shap_explainer] X_val.npy not found -- using synthetic data")
        num_features = len(FEATURES)
        X_val = np.random.rand(100, TIME_STEPS, num_features).astype("float32")

    results = run_shap_explanation(X_val, n_background=30, n_explain=20)
    print("\nAll SHAP plots saved:")
    for k in ("global_path", "summary_path", "heatmap_path"):
        print(f"  {k}: {results[k]}")
