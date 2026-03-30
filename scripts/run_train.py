#!/usr/bin/env python3
"""
LungCancerDX – Training Runner
Run after run_split.py to train all ensemble models.

Usage:
    python scripts/run_train.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ml"))

from train import main

if __name__ == "__main__":
    print("=" * 60)
    print("  LungCancerDX — Step 2: Model Training")
    print("=" * 60)
    main()
