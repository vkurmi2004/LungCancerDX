"""
LungCancerDX – Training Engine
Trains all ensemble models with early stopping, LR scheduling,
confusion matrices and ROC curves.
"""
import os
import sys
import time
import random
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import datasets
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, roc_curve, auc
)
from sklearn.preprocessing import label_binarize

# Allow importing sibling modules
sys.path.insert(0, str(Path(__file__).parent))
from config import *
from models import create_model, FACTORY
from preprocessing import TRAIN_TRANSFORM, VAL_TRANSFORM, split_dataset, augment_training_dir

# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(str(BASE_DIR / "training.log")),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def evaluate(model, loader):
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(DEVICE)
            out  = model(imgs)
            prob = torch.softmax(out, dim=1).cpu().numpy()
            pred = np.argmax(prob, axis=1)
            y_true.extend(labels.numpy())
            y_pred.extend(pred)
            y_prob.extend(prob)
    return np.array(y_true), np.array(y_pred), np.array(y_prob)


def plot_confusion_matrix(y_true, y_pred, class_names, title, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title, fontsize=14)
    plt.ylabel("True Label"); plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150)
    plt.close()


def plot_roc_curves(y_true, y_prob, class_names, title, save_path):
    y_bin = label_binarize(y_true, classes=list(range(len(class_names))))
    plt.figure(figsize=(6, 5))
    for i, cls in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{cls} (AUC={roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.title(title, fontsize=14)
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150)
    plt.close()


def plot_training_history(history, arch, save_dir):
    epochs = range(1, len(history["train_acc"]) + 1)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    axs[0].plot(epochs, history["train_loss"], label="Train Loss")
    axs[0].plot(epochs, history["val_loss"], label="Val Loss")
    axs[0].set_title(f"{arch} – Loss"); axs[0].legend()
    axs[1].plot(epochs, history["train_acc"], label="Train Acc")
    axs[1].plot(epochs, history["val_acc"], label="Val Acc")
    axs[1].set_title(f"{arch} – Accuracy"); axs[1].legend()
    plt.tight_layout()
    plt.savefig(str(save_dir / f"{arch}_history.png"), dpi=150)
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Training Loop
# ─────────────────────────────────────────────────────────────────────────────
def train_one_model(arch: str, train_loader, val_loader, class_names, report_dir):
    logger.info(f"\n{'='*60}\n  Training: {arch}\n{'='*60}")
    net       = create_model(arch, NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    best_val_acc  = 0.0
    best_wts      = net.state_dict()
    no_improve    = 0
    history       = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(1, EPOCHS + 1):
        # ── Train ──
        net.train()
        t_loss, t_correct, t_total = 0.0, 0, 0
        for imgs, labels in tqdm(train_loader, desc=f"  Epoch {epoch}/{EPOCHS} [train]", leave=False):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            out  = net(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            t_loss    += loss.item() * labels.size(0)
            _, preds   = torch.max(out, 1)
            t_correct += (preds == labels).sum().item()
            t_total   += labels.size(0)

        train_loss = t_loss / t_total
        train_acc  = t_correct / t_total

        # ── Validate ──
        net.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                out  = net(imgs)
                loss = criterion(out, labels)
                v_loss    += loss.item() * labels.size(0)
                _, preds   = torch.max(out, 1)
                v_correct += (preds == labels).sum().item()
                v_total   += labels.size(0)

        val_loss = v_loss / v_total
        val_acc  = v_correct / v_total
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        logger.info(
            f"  [{arch}] E{epoch:03d} | "
            f"Train Loss={train_loss:.4f} Acc={train_acc:.4f} | "
            f"Val Loss={val_loss:.4f} Acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_wts     = {k: v.clone() for k, v in net.state_dict().items()}
            no_improve   = 0
        else:
            no_improve += 1

        if no_improve >= PATIENCE:
            logger.info(f"  Early stopping at epoch {epoch}.")
            break

    net.load_state_dict(best_wts)

    # Save checkpoint
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = MODEL_DIR / f"{arch.lower()}_best.pth"
    torch.save(net.state_dict(), str(ckpt_path))
    logger.info(f"  ✅ Saved {arch} checkpoint → {ckpt_path}")

    # Plots
    plot_training_history(history, arch, report_dir)

    return net, best_val_acc


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # ── 1. Prepare data ──────────────────────────────────────────────────────
    if not TRAIN_DIR.exists() or not any(TRAIN_DIR.iterdir()):
        logger.info("Splitting dataset …")
        split_dataset(RAW_DATA_DIR, SPLIT_DIR, TRAIN_RATIO, SEED)

    if not AUG_TRAIN_DIR.exists() or not any(AUG_TRAIN_DIR.iterdir()):
        logger.info("Augmenting training set …")
        augment_training_dir(TRAIN_DIR, AUG_TRAIN_DIR, AUG_COPIES)

    # ── 2. Determine training source ─────────────────────────────────────────
    active_train_dir = AUG_TRAIN_DIR if AUG_TRAIN_DIR.exists() else TRAIN_DIR

    train_data = datasets.ImageFolder(str(active_train_dir), transform=TRAIN_TRANSFORM)
    val_data   = datasets.ImageFolder(str(VAL_DIR),  transform=VAL_TRANSFORM)
    test_data  = datasets.ImageFolder(str(TEST_DIR), transform=VAL_TRANSFORM)

    class_names = train_data.classes
    logger.info(f"Classes : {class_names}")
    logger.info(f"Train   : {len(train_data):,} | Val: {len(val_data):,} | Test: {len(test_data):,}")

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_data,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_data,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, pin_memory=True)

    # ── 3. Reports directory ─────────────────────────────────────────────────
    report_dir = BASE_DIR / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    # ── 4. Train each architecture ───────────────────────────────────────────
    trained_models = {}
    summary_rows   = []

    for arch in ENSEMBLE_WEIGHTS:
        net, best_val = train_one_model(arch, train_loader, val_loader, class_names, report_dir)
        y_true, y_pred, y_prob = evaluate(net, test_loader)
        test_acc = accuracy_score(y_true, y_pred)

        logger.info(f"\n{arch} – Test Accuracy: {test_acc:.4f}")
        logger.info(classification_report(y_true, y_pred, target_names=class_names))

        # Confusion matrix & ROC
        plot_confusion_matrix(y_true, y_pred, class_names,
                              f"Confusion Matrix – {arch}",
                              report_dir / f"{arch}_cm.png")
        plot_roc_curves(y_true, y_prob, class_names,
                        f"ROC Curves – {arch}",
                        report_dir / f"{arch}_roc.png")

        trained_models[arch] = {"net": net, "y_true": y_true, "y_pred": y_pred, "y_prob": y_prob}
        summary_rows.append({
            "Model":    arch,
            "Val Acc":  round(best_val * 100, 2),
            "Test Acc": round(test_acc * 100, 2),
        })

    # ── 5. Soft-Voting Ensemble ───────────────────────────────────────────────
    weights  = list(ENSEMBLE_WEIGHTS.values())
    y_true_0 = trained_models[list(ENSEMBLE_WEIGHTS.keys())[0]]["y_true"]

    prob_sum = np.zeros((len(y_true_0), NUM_CLASSES))
    for i, (arch, w) in enumerate(ENSEMBLE_WEIGHTS.items()):
        prob_sum += w * trained_models[arch]["y_prob"]
    prob_sum /= sum(weights)

    y_ens      = np.argmax(prob_sum, axis=1)
    ens_acc    = accuracy_score(y_true_0, y_ens)
    logger.info(f"\n🏆 Soft-Voting Ensemble Test Accuracy: {ens_acc:.4f}")

    plot_confusion_matrix(y_true_0, y_ens, class_names,
                          "Confusion Matrix – Ensemble",
                          report_dir / "Ensemble_cm.png")
    plot_roc_curves(y_true_0, prob_sum, class_names,
                    "ROC Curves – Ensemble",
                    report_dir / "Ensemble_roc.png")

    summary_rows.append({"Model": "Ensemble", "Val Acc": "—", "Test Acc": round(ens_acc * 100, 2)})

    # ── 6. Summary table & bar chart ─────────────────────────────────────────
    df = pd.DataFrame(summary_rows)
    logger.info(f"\n{df.to_string(index=False)}")
    df.to_csv(str(report_dir / "summary.csv"), index=False)

    numeric_df = df[df["Test Acc"] != "—"].copy()
    numeric_df["Test Acc"] = numeric_df["Test Acc"].astype(float)
    plt.figure(figsize=(10, 5))
    plt.bar(numeric_df["Model"], numeric_df["Test Acc"] / 100)
    plt.ylim(0, 1); plt.ylabel("Test Accuracy")
    plt.title("Model Comparison – Test Accuracy")
    plt.tight_layout()
    plt.savefig(str(report_dir / "model_comparison.png"), dpi=150)
    plt.close()

    logger.info(f"\n✅ All reports saved to: {report_dir}")
    logger.info(f"✅ All checkpoints saved to: {MODEL_DIR}")


if __name__ == "__main__":
    main()
