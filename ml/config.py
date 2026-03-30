"""
LungCancerDX – Training Configuration
"""
import os
from pathlib import Path

# ─── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).resolve().parent.parent

RAW_DATA_DIR  = (
    BASE_DIR.parent                                    # lung cancer dataset aditya mahimkar/
    / "The IQ-OTHNCCD lung cancer dataset"
    / "The IQ-OTHNCCD lung cancer dataset"
)

SPLIT_DIR     = BASE_DIR / "data_split"
TRAIN_DIR     = SPLIT_DIR / "training"
VAL_DIR       = SPLIT_DIR / "validation"
TEST_DIR      = SPLIT_DIR / "testing"
AUG_TRAIN_DIR = SPLIT_DIR / "training_aug"
MODEL_DIR     = BASE_DIR / "models"

# ─── Classes ─────────────────────────────────────────────────────────────────
CLASSES       = ["Bengin cases", "Malignant cases", "Normal cases"]  # folder names
CLASS_LABELS  = ["Benign", "Malignant", "Normal"]                    # display names
NUM_CLASSES   = len(CLASSES)

# ─── Image ───────────────────────────────────────────────────────────────────
IMG_SIZE       = 224
APPLY_CLAHE    = True
APPLY_GAUSSIAN = True

# ─── Training ────────────────────────────────────────────────────────────────
BATCH_SIZE = 16
EPOCHS     = 1             # Reduced from 50 to 1 for quick demo
LR         = 1e-4
WD         = 1e-4          # weight decay
PATIENCE   = 10            # early stopping patience

# ─── Augmentation ────────────────────────────────────────────────────────────
AUG_COPIES = 2             # augmented copies per original image

# ─── Ensemble Weights ────────────────────────────────────────────────────────
ENSEMBLE_WEIGHTS = {
    "DenseNet121":    1.0,
}

# ─── Split ───────────────────────────────────────────────────────────────────
TRAIN_RATIO = 0.80        # 80 / 10 / 10

# ─── Reproducibility ─────────────────────────────────────────────────────────
SEED = 42
