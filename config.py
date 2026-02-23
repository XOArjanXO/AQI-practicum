# =============================================================================
# config.py — Central configuration for XAI Air Quality Project
# =============================================================================

# ── Data / Sequence ──────────────────────────────────────────────────────────
TIME_STEPS = 24          # Number of past time-steps fed as input to the model

FEATURES = [
    'PM10_lag',
    'Temperature',
    'Humidity',
    'WindSpeed',
    'WindDirection',
    'Pressure',
    'Rainfall',
]

TARGET = 'PM10'          # Column being predicted

# ── Training Hyper-parameters ─────────────────────────────────────────────────
BATCH_SIZE = 32
EPOCHS     = 50
LR         = 1e-3        # Learning rate (Adam optimiser)

# ── Model Architecture ────────────────────────────────────────────────────────
CNN_OUT_CHANNELS = 64    # Number of Conv1d filters
CNN_KERNEL_SIZE  = 3     # Kernel width for Conv1d
LSTM_HIDDEN_SIZE = 128   # Hidden units in LSTM
LSTM_NUM_LAYERS  = 2     # Number of stacked LSTM layers
DROPOUT          = 0.3   # Dropout rate (applied in LSTM & attention)

# ── Paths ─────────────────────────────────────────────────────────────────────
RAW_DATA_PATH       = 'data/raw/airnet_raw.csv'
PROCESSED_DIR       = 'data/processed/'
SCALER_PATH         = 'data/scalers/feature_scaler.pkl'
CHECKPOINT_PATH     = 'models/model_checkpoint.pth'
METRICS_PATH        = 'results/metrics.json'
TRAINING_LOG_PATH   = 'results/training_log.txt'
XAI_PLOTS_DIR       = 'xai/plots/'

# ── Reproducibility ───────────────────────────────────────────────────────────
RANDOM_SEED = 42

# ── Train / Validation Split ──────────────────────────────────────────────────
VAL_SPLIT = 0.2          # 20 % of data reserved for validation
