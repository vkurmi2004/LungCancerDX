#!/usr/bin/env python3
"""
LungCancerDX – Dataset Split & Augmentation Runner
Run this FIRST before training to prepare the data.

Usage:
    python scripts/run_split.py
"""
import sys
from pathlib import Path

# Allow imports from ml/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ml"))

from config import RAW_DATA_DIR, SPLIT_DIR, AUG_TRAIN_DIR, TRAIN_DIR, TRAIN_RATIO, SEED, AUG_COPIES
from preprocessing import split_dataset, augment_training_dir


def main():
    print("=" * 60)
    print("  LungCancerDX — Step 1: Dataset Preparation")
    print("=" * 60)

    if not RAW_DATA_DIR.exists():
        print(f"\n❌ Raw dataset not found at:\n   {RAW_DATA_DIR}")
        print("\n   Please ensure the IQ-OTH/NCCD dataset is in the correct location.")
        sys.exit(1)

    # ── Split ──────────────────────────────────────────────────────────────────
    if TRAIN_DIR.exists() and any(TRAIN_DIR.iterdir()):
        print(f"\n✅ Split already exists at {SPLIT_DIR}  (skipping split)")
    else:
        print(f"\n📂 Splitting dataset  →  {SPLIT_DIR}")
        counts = split_dataset(RAW_DATA_DIR, SPLIT_DIR, TRAIN_RATIO, SEED)
        for split, cls_counts in counts.items():
            total = sum(cls_counts.values())
            print(f"   {split.ljust(12)}: {total} images", end="")
            for cls, n in cls_counts.items():
                print(f"  |  {cls}: {n}", end="")
            print()

    # ── Augmentation ───────────────────────────────────────────────────────────
    if AUG_TRAIN_DIR.exists() and any(AUG_TRAIN_DIR.iterdir()):
        print(f"\n✅ Augmented data already at {AUG_TRAIN_DIR}  (skipping)")
    else:
        print(f"\n🔀 Augmenting training set  (×{AUG_COPIES + 1} per image)  →  {AUG_TRAIN_DIR}")
        aug_counts = augment_training_dir(TRAIN_DIR, AUG_TRAIN_DIR, AUG_COPIES)
        for cls, total in aug_counts.items():
            print(f"   {cls}: {total} images after augmentation")

    print("\n✅ Dataset preparation complete. You can now run training.")
    print("   → python scripts/run_train.py\n")


if __name__ == "__main__":
    main()
