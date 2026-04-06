# 🫁 LungCancerDX — Final Year Project

> **AI-Powered Lung Cancer Detection from CT Scans**  
> Ensemble of 5 deep learning models · CLAHE preprocessing · Grad-CAM visualisation · Real-time FastAPI backend · Premium interactive frontend

---

## 📋 Project Overview

LungCancerDX is a comprehensive computer-vision system that classifies lung CT scans into three categories:

| Class | Description |
|-------|-------------|
| **Benign** | Non-cancerous tissue / benign nodule |
| **Malignant** | Cancerous tissue – high-risk |
| **Normal** | Healthy lung tissue |

**Dataset:** [IQ-OTH/NCCD Lung Cancer Dataset](https://www.kaggle.com/datasets/adityamahimkar/iqothnccd-lung-cancer-dataset) by Aditya Mahimkar — 1,097 real CT scans collected at Iraq Oncology Teaching Hospital.

---

## 🏗️ Project Structure

```
LungCancerDX/
├── backend/
│   └── main.py              ← FastAPI server (ensemble inference + Grad-CAM)
├── frontend/
│   ├── index.html           ← Single-page app
│   ├── style.css            ← Premium dark-mode design system
│   └── app.js               ← Full interactive JS (drag-drop, webcam, report)
├── ml/
│   ├── config.py            ← All hyper-parameters & paths
│   ├── models.py            ← 5 CNN factory functions
│   ├── preprocessing.py     ← CLAHE, augmentation, split utilities
│   └── train.py             ← Full training engine (early stopping, LR scheduler)
├── models/                  ← Trained checkpoints (.pth) saved here
├── data_split/              ← Auto-created: training / validation / testing
├── reports/                 ← Auto-created: confusion matrices, ROC curves, CSV
├── scripts/
│   ├── run_split.py         ← Step 1: split + augment dataset
│   └── run_train.py         ← Step 2: train all models
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Recommended: create a virtual environment
python -m venv venv
source venv/bin/activate       # macOS / Linux
# venv\Scripts\activate        # Windows

pip install -r requirements.txt
```

### 2. Prepare the Dataset

Ensure the raw dataset is at:
```
lung cancer dataset aditya mahimkar/
└── The IQ-OTHNCCD lung cancer dataset/
    └── The IQ-OTHNCCD lung cancer dataset/
        ├── Bengin cases/
        ├── Malignant cases/
        └── Normal cases/
```

Then run:
```bash
python scripts/run_split.py
```

This will:
- Split data 80/10/10 (train/val/test) ✓
- Apply CLAHE enhancement per image ✓
- Create ×3 augmented training copies ✓

### 3. Train the Models

```bash
python scripts/run_train.py
```

Trains all 5 models with:
- Early stopping (patience=10)
- Cosine Annealing LR scheduler
- Label smoothing (0.1)
- AdamW optimiser
- Saves best checkpoint per model to `models/`
- Generates confusion matrices and ROC curves in `reports/`

### 4. Start the Backend

```bash
cd LungCancerDX
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

### 5. Open the Frontend

Visit: **http://localhost:8000**

Or open `frontend/index.html` directly in your browser.

---

## 🧠 Ensemble Architecture

| Model | Params | Ensemble Weight | Notes |
|-------|--------|----------------|-------|
| ResNet-50 | 25.6M | **1.3** | Best overall CT features |
| EfficientNet-B0 | 5.3M | **1.2** | Best accuracy/speed ratio |
| DenseNet-121 | 7.9M | **1.2** | Dense connections reuse features |
| MobileNetV3-Small | 2.5M | **0.9** | Lightweight, fast inference |
| VGG-16 | 138M | **1.0** | Strong baseline |
| **Soft-Voting Ensemble** | — | — | **Best accuracy** |

**Ensemble strategy:** Weighted soft-voting — each model's softmax probability vector is multiplied by its weight, summed, and normalised.

---

## 🔬 Preprocessing Pipeline

```
Raw CT Image
    │
    ▼
Resize → 224×224
    │
    ▼
CLAHE (Contrast Limited Adaptive Histogram Equalization)
    │
    ▼
Optional Gaussian Blur (denoise)
    │
    ▼
ImageNet Normalization (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    │
    ▼
Model Input
```

**Training augmentations:** Random horizontal/vertical flip, rotation ±20°, colour jitter, affine translate/scale.

---

## 🌐 API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/` | Frontend SPA |
| `GET`  | `/api/health` | System health & model status |
| `GET`  | `/api/models` | Model architectures & weights |
| `POST` | `/api/predict` | **Main inference** (upload image → diagnosis) |
| `POST` | `/api/gradcam` | Grad-CAM heatmap overlay |
| `GET`  | `/api/dataset/stats` | Dataset class counts |
| `GET`  | `/api/reports` | List training report files |
| `GET`  | `/docs` | Swagger API documentation |

### Example: Predict

```bash
curl -X POST http://localhost:8000/api/predict \
  -F "file=@my_ct_scan.jpg"
```

Response:
```json
{
  "prediction": "Malignant",
  "confidence": 87.34,
  "risk_level": "HIGH",
  "risk_color": "#ef4444",
  "recommendation": "⚠️ Immediate oncologist consultation recommended...",
  "class_probabilities": { "Benign": 3.21, "Malignant": 87.34, "Normal": 9.45 },
  "per_model_predictions": { "ResNet50": {...}, "EfficientNetB0": {...}, ... },
  "inference_time_ms": 42.1,
  "models_used": ["ResNet50", "EfficientNetB0", "DenseNet121", "MobileNetV3", "VGG16"]
}
```

---

## 🖥️ Frontend Features

- **Drag & Drop Upload** — with image preview and metadata
- **Webcam Capture** — capture frames directly from camera
- **Sample Image** — synthetic CT scan for quick demo
- **Probability Bars** — animated per-class confidence bars
- **Model Breakdown** — per-model predictions with ensemble weights
- **Grad-CAM Heatmap** — visualise where the model "looks" (requires trained ResNet50)
- **Risk Badge** — colour-coded HIGH / MODERATE / LOW
- **Clinical Recommendation** — evidence-based follow-up advice
- **Report Download** — plaintext diagnostic report
- **API Status Indicator** — live backend connectivity check

---

## 📊 Expected Results

After training 50 epochs on the IQ-OTH/NCCD dataset:

| Model | Expected Test Acc |
|-------|------------------|
| ResNet-50 | ~90–92% |
| EfficientNet-B0 | ~92–94% |
| DenseNet-121 | ~91–93% |
| MobileNetV3 | ~89–91% |
| VGG-16 | ~90–92% |
| **Ensemble** | **~93–95%** |

---

## ⚠️ Disclaimer

> This system is developed for **research and educational purposes** as a final year project.  
> It is **NOT a substitute for professional medical diagnosis**.  
> Always consult a qualified radiologist or oncologist for medical decisions.

---

## 📚 References

- He et al. (2016) — [Deep Residual Learning (ResNet)](https://arxiv.org/abs/1512.03385)
- Tan & Le (2019) — [EfficientNet](https://arxiv.org/abs/1905.11946)
- Huang et al. (2017) — [DenseNet](https://arxiv.org/abs/1608.06993)
- Howard et al. (2019) — [MobileNetV3](https://arxiv.org/abs/1905.02244)
- Simonyan & Zisserman (2014) — [VGG](https://arxiv.org/abs/1409.1556)
- Selvaraju et al. (2017) — [Grad-CAM](https://arxiv.org/abs/1610.02391)
- Dataset: Aditya Mahimkar — IQ-OTH/NCCD Lung Cancer Dataset

---

*Built with PyTorch · FastAPI · Vanilla JS · OpenCV*
