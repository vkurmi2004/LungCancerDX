"""
LungCancerDX – Data Preprocessing & Augmentation Pipeline
"""
import os
import cv2
import random
import shutil
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

import torch
from torchvision import transforms


# ─────────────────────────────────────────────────────────────────────────────
# CLAHE helper (tensor → tensor)
# ─────────────────────────────────────────────────────────────────────────────
def apply_clahe_to_tensor(img: torch.Tensor) -> torch.Tensor:
    arr = img.numpy().transpose(1, 2, 0)
    arr = (arr * 255).astype(np.uint8)
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    arr = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return torch.tensor(arr / 255.0, dtype=torch.float32).permute(2, 0, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Transforms
# ─────────────────────────────────────────────────────────────────────────────
IMAGENET_NORM = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)

TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.85, 1.15)),
    transforms.ToTensor(),
    transforms.Lambda(apply_clahe_to_tensor),
    IMAGENET_NORM,
])

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Lambda(apply_clahe_to_tensor),
    IMAGENET_NORM,
])


# ─────────────────────────────────────────────────────────────────────────────
# Image-level augmentation (OpenCV, for offline augmentation)
# ─────────────────────────────────────────────────────────────────────────────
def _random_flip(img):
    return cv2.flip(img, random.choice([-1, 0, 1]))

def _random_rotate(img):
    angle = random.uniform(-25, 25)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

def _random_zoom(img):
    zoom = random.uniform(0.8, 1.2)
    h, w = img.shape[:2]
    nh, nw = int(h * zoom), int(w * zoom)
    resized = cv2.resize(img, (nw, nh))
    if zoom < 1:
        ph, pw = (h - nh) // 2, (w - nw) // 2
        return cv2.copyMakeBorder(resized, ph, h - nh - ph, pw, w - nw - pw, cv2.BORDER_REFLECT)
    sh, sw = (nh - h) // 2, (nw - w) // 2
    return resized[sh:sh + h, sw:sw + w]

def _random_brightness(img):
    factor = random.uniform(0.7, 1.3)
    return np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)

def _apply_clahe_bgr(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

_AUGMENTATIONS = [_random_flip, _random_rotate, _random_zoom, _random_brightness]


def preprocess_image_cv(img_path: str, img_size: int = 224, apply_gaussian: bool = True):
    """Load, CLAHE-enhance, optional Gaussian blur an image via OpenCV."""
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Cannot read image: {img_path}")
    img = cv2.resize(img, (img_size, img_size))
    img = _apply_clahe_bgr(img)
    if apply_gaussian:
        img = cv2.GaussianBlur(img, (3, 3), 0)
    return img


# ─────────────────────────────────────────────────────────────────────────────
# Dataset Split
# ─────────────────────────────────────────────────────────────────────────────
def split_dataset(raw_dir: Path, out_dir: Path, train_ratio: float = 0.80, seed: int = 42):
    """Split raw dataset into train/val/test preserving class structure."""
    out_dir.mkdir(parents=True, exist_ok=True)
    classes = [d.name for d in raw_dir.iterdir() if d.is_dir()]
    split_counts = {split: {} for split in ["training", "validation", "testing"]}

    for cls in classes:
        cls_path = raw_dir / cls
        images = [f for f in cls_path.iterdir()
                  if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".tif")]

        if not images:
            continue

        train_imgs, temp = train_test_split(images, test_size=1 - train_ratio,
                                            random_state=seed, shuffle=True)
        val_imgs, test_imgs = train_test_split(temp, test_size=0.5,
                                               random_state=seed, shuffle=True)

        for split_name, split_imgs in zip(
            ["training", "validation", "testing"],
            [train_imgs, val_imgs, test_imgs]
        ):
            dest = out_dir / split_name / cls
            dest.mkdir(parents=True, exist_ok=True)
            for img_file in split_imgs:
                # Preprocess while copying
                preprocessed = preprocess_image_cv(str(img_file))
                cv2.imwrite(str(dest / img_file.name), preprocessed)
            split_counts[split_name][cls] = len(split_imgs)

    return split_counts


# ─────────────────────────────────────────────────────────────────────────────
# Offline Augmentation
# ─────────────────────────────────────────────────────────────────────────────
def augment_training_dir(train_dir: Path, out_dir: Path, copies: int = 2):
    """Create augmented copies of training images."""
    out_dir.mkdir(parents=True, exist_ok=True)
    aug_counts = {}

    for cls_path in train_dir.iterdir():
        if not cls_path.is_dir():
            continue
        save_cls = out_dir / cls_path.name
        save_cls.mkdir(parents=True, exist_ok=True)
        total = 0

        for img_file in cls_path.iterdir():
            if img_file.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            img = cv2.imread(str(img_file))
            cv2.imwrite(str(save_cls / img_file.name), img)
            total += 1

            for k in range(copies):
                aug = img.copy()
                for fn in random.sample(_AUGMENTATIONS, 2):
                    aug = fn(aug)
                aug_name = f"{img_file.stem}_aug{k + 1}{img_file.suffix}"
                cv2.imwrite(str(save_cls / aug_name), aug)
                total += 1

        aug_counts[cls_path.name] = total

    return aug_counts
