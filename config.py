"""
config.py — Single source of truth for all project settings.

To change any hyperparameter, edit here only.
All other files import from this module.
"""

import os
from pathlib import Path

# ──────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────

ROOT_DIR        = Path(__file__).parent.resolve()

DATA_DIR        = ROOT_DIR / "data"
STATEFARM_DIR   = DATA_DIR / "statefarm"
DDD_DIR         = DATA_DIR / "ddd" / "Driver Drowsiness Dataset (DDD)"

CHECKPOINT_DIR  = ROOT_DIR / "checkpoints"
LOG_DIR         = ROOT_DIR / "logs"

# Auto-create dirs if they don't exist
for _dir in [CHECKPOINT_DIR, LOG_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────
# DATA
# ──────────────────────────────────────────────

SEQ_LEN         = 10        # frames per sequence
IMG_SIZE        = 224       # input resolution (MobileNetV2 expects 224x224)
NUM_CLASSES     = 3         # safe / distracted / drowsy

CLASS_NAMES     = {
    0: "safe",
    1: "distracted",
    2: "drowsy",
}

# State Farm: how sequences are built
SF_MIN_IMGS_PER_GROUP   = 10    # skip groups with fewer images than SEQ_LEN
SF_VAL_RATIO            = 0.15
SF_TEST_RATIO           = 0.10
SF_SEED                 = 42

# DDD: sliding window settings
DDD_STRIDE      = 3             # overlap between sequences (lower = more sequences)
DDD_SEED        = 42

# ──────────────────────────────────────────────
# MODEL
# ──────────────────────────────────────────────

CNN_BACKBONE        = "mobilenet_v2"
CNN_OUTPUT_DIM      = 1280          # MobileNetV2 feature dim
PROJECTION_DIM      = 512           # CNN → LSTM projection

LSTM_HIDDEN_SIZE    = 512
LSTM_NUM_LAYERS     = 2
LSTM_DROPOUT        = 0.5

FC_DROPOUT          = 0.4

# ──────────────────────────────────────────────
# TRAINING — PHASE 1 (State Farm)
# ──────────────────────────────────────────────

PHASE1 = {
    "epochs"            : 15,
    "batch_size"        : 8,        # CPU-safe batch size
    "learning_rate"     : 3e-4,     # LSTM + FC only (CNN frozen)
    "weight_decay"      : 3e-4,
    "lr_patience"       : 3,        # ReduceLROnPlateau patience
    "lr_factor"         : 0.5,      # halve LR on plateau
    "early_stop_patience": 6,
    "num_workers"       : 2,
    "label_smoothing"   : 0.2,
    "checkpoint_name"   : "phase1_best.pth",
    "log_name"          : "phase1_train.csv",
    "freeze_cnn"        : True,
}

# ──────────────────────────────────────────────
# TRAINING — PHASE 2 (DDD fine-tune)
# ──────────────────────────────────────────────

PHASE2 = {
    "epochs"            : 10,
    "batch_size"        : 4,        # smaller: fine-tuning needs less noise
    "learning_rate_lstm": 1e-4,     # lower LR for already-trained LSTM
    "learning_rate_cnn" : 1e-5,     # much lower for unfrozen CNN blocks
    "weight_decay"      : 1e-4,
    "lr_patience"       : 2,
    "lr_factor"         : 0.5,
    "early_stop_patience": 4,
    "num_workers"       : 2,
    "label_smoothing"   : 0.05,     # less smoothing: fine-tuning needs precision
    "unfreeze_cnn_blocks": 2,
    "checkpoint_name"   : "phase2_best.pth",
    "log_name"          : "phase2_train.csv",
    "freeze_cnn"        : False,
}

# ──────────────────────────────────────────────
# SYSTEM
# ──────────────────────────────────────────────

import torch
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
NUM_THREADS     = 8         # torch.set_num_threads — sweet spot for 18-core CPU
SEED            = 42

# ──────────────────────────────────────────────
# INFERENCE
# ──────────────────────────────────────────────

INFERENCE_SEQ_LEN       = 10       # frames to buffer before predicting
INFERENCE_CONFIDENCE    = 0.65     # minimum confidence to show alert
WEBCAM_INDEX            = 0        # default webcam

# ──────────────────────────────────────────────
# QUICK SUMMARY
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("PROJECT CONFIGURATION")
    print("=" * 50)
    print(f"  Device        : {DEVICE}")
    print(f"  Root dir      : {ROOT_DIR}")
    print(f"  Sequence len  : {SEQ_LEN} frames")
    print(f"  Image size    : {IMG_SIZE}x{IMG_SIZE}")
    print(f"  Num classes   : {NUM_CLASSES} {list(CLASS_NAMES.values())}")
    print(f"\n  Phase 1 epochs: {PHASE1['epochs']}  batch: {PHASE1['batch_size']}  lr: {PHASE1['learning_rate']}")
    print(f"  Phase 2 epochs: {PHASE2['epochs']}  batch: {PHASE2['batch_size']}  lr_lstm: {PHASE2['learning_rate_lstm']}")
    print("=" * 50)