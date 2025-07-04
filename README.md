
# FACECOM Challenge – Gender Classification & Face Recognition

This repository contains the full pipeline for solving the [FACECOM Hackathon Challenge], consisting of two tasks:

- **Task A:** Gender Classification (Binary Classification)
- **Task B:** Face Recognition (Multi-class Classification)

The goal is to develop models robust to **non-ideal environments** such as blur, fog, rain, overexposure, and profile/partial faces.

---

## 📂 Dataset Structure

```text
FACECOM/
│
├── Task_A/
│   ├── train/
│   │   ├── male/
│   │   └── female/
│   └── val/
│       ├── male/
│       └── female/
│
└── Task_B/
    ├── train/
    │   ├── person_id_1/
    │   ├── ...
    └── val/
        ├── person_id_1/
        ├── ...
```

---

## 🔧 Tech Stack

- Python, PyTorch, torchvision
- Albumentations (Augmentations)
- scikit-learn (metrics)
- Google Colab (CPU-based training)
- EfficientNet (Task A), MobileNetV3 (Task B)

---

## 🚀 Task A – Gender Classification

### ✔️ Challenges:
- Profile & occluded faces
- Cropped faces or half cut
- Partial shadows

### ✅ Optimizations:
- **Model:** EfficientNet-B0 with Dropout
- **Loss:** `CrossEntropyLoss` with Label Smoothing
- **Sampler:** WeightedRandomSampler to handle class imbalance
- **Augmentations:** HorizontalFlip, CLAHE, RandomShadow (light distortions only)
- **Inference:** PIL-based loader with torchvision transforms

---

## 🚀 Task B – Face Recognition

### ✔️ Challenges:
- Low light, motion blur, fog, rain, glare, overexposed
- High inter-class similarity
- Hundreds of class IDs

### ✅ Optimizations:
- **Model:** MobileNetV3-Small (pretrained on ImageNet)
- **Fine-tuning:** Initially frozen, unfreeze after 3 epochs
- **Loss:** Label Smoothing
- **Scheduler:** `ReduceLROnPlateau`
- **Augmentations:** Only light distortions to preserve identity
- **Training:** Gradient Accumulation, Weighted Sampling

---

## 📊 Evaluation Metrics

- **Task A:** Accuracy, Precision, Recall, F1-Score
- **Task B:** Top-1 Accuracy, Macro-averaged F1-Score
- **Final Score:** Combined weighted score (Task A - 30%, Task B - 70%)

---

## 🧠 Training Notes

- CPU-only training was used due to hardware limitations.
- Each Task B epoch took ~1 hour initially; after optimization and MobileNetV3 patching, reduced to ~15 mins.
- Best models saved based on validation F1.

---

## 💡 How to Run

```bash
# Clone repo
git clone https://github.com/<your-username>/facecom-challenge.git

# Launch training in Jupyter Notebook
Open `TaskA_Train.ipynb` or `TaskB_Train.ipynb` in Google Colab.
```

---

## 📁 File Structure

```text
├── TaskA_Train.ipynb
├── TaskA_Inference.ipynb
├── TaskB_Train.ipynb
├── TaskB_Inference.ipynb
├── README.md
└── best_model_taska.pth / best_model_taskb.pth
```

---

## 👩‍💻 Author

**Pranita Mahajan**  
BTech CSE | AI/ML Enthusiast | Generative AI, DL, Python, SpringBoot  
Hackathon Participant – [Comys Hackathon 5]
