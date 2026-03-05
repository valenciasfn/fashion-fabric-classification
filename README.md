# 👗 Fashion Model Classification Based on Material Types
### Using Pattern Recognition Techniques

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat-square&logo=tensorflow)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-latest-F7931E?style=flat-square&logo=scikit-learn)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)

> A machine learning project that automatically classifies fashion garment images into 6 fabric types — **Corduroy, Denim, Katun (Cotton), Linen, Organza, and Satin** — using traditional ML, deep learning (CNN), and transfer learning approaches.

---

## 📌 Table of Contents
- [Overview](#-overview)
- [Dataset](#-dataset)
- [Methodology](#-methodology)
- [Results](#-results)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Tech Stack](#-tech-stack)

---

## 🔍 Overview

In the fashion industry, manually identifying fabric types is time-consuming and error-prone. This project aims to **automate fabric classification from garment images** using computer vision and machine learning.

**Business Objective:** Reduce human error, save operational costs, and support digitalization across production and sales pipelines in the fashion industry.

**Key Challenges:**
- High visual similarity between classes (e.g., Corduroy vs. Denim, Linen vs. Satin)
- Dataset with significant class overlap as shown by t-SNE visualization
- Varying image resolutions requiring standardization

---

## 📦 Dataset

| Property | Details |
|---|---|
| **Total Images** | 1,200 images |
| **Classes** | 6 (Corduroy, Denim, Katun, Linen, Organza, Satin) |
| **Images per Class** | 100 images |
| **Source** | Shopee internal documentation & Pinterest |
| **Image Format** | RGB, various resolutions |

### Fabric Classes

| Class | Description |
|---|---|
| 🟤 Corduroy | Vertically ribbed texture with distinct grooves |
| 🔵 Denim | Twill weave with indigo-dyed warp threads |
| 🟡 Katun | Lightweight cotton fabric with soft surface |
| 🟢 Linen | Plant-fiber fabric with visible vertical/horizontal threads |
| ⚪ Organza | Sheer, lightweight, lustrous fabric |
| 🔮 Satin | Smooth, soft, and glossy surface texture |

---

## 🧪 Methodology

### 1. Data Preprocessing
- **Resize:** Standardized all images to `300×300` px (SVM/CNN) or `224×224` px (Transfer Learning)
- **Grayscale Transformation:** Reduced color complexity to focus on texture patterns
- **Normalization:** Pixel values scaled to `[0, 1]`

### 2. Exploratory Data Analysis
- Class distribution analysis (balanced dataset confirmed)
- **t-SNE visualization** — revealed overlap between Corduroy, Denim, Katun, and Organza
- Image dimension analysis across all classes

### 3. Feature Extraction (for SVM)

| Feature | Purpose |
|---|---|
| **HOG** (Histogram of Oriented Gradients) | Captures edge/gradient orientation patterns |
| **Sobel Edge Detector** | Detects sharp intensity changes for shape contours |
| **Color Histogram** | Captures color frequency distribution per RGB channel |
| **Color Moments** | Statistical color features (mean, variance, skewness) |

### 4. Data Augmentation (for CNN & Transfer Learning)
- Random flip, rotation, zoom, contrast, height & width shifts

### 5. Machine Learning Models

```
├── Classical ML
│   └── SVM (Support Vector Machine)
│       ├── HOG features
│       ├── Sobel Edge features
│       ├── Color Histogram features
│       └── Color Moment features
│
├── Deep Learning
│   └── CNN (Convolutional Neural Network)
│       ├── 3 Conv layers (32 → 64 → 128 filters)
│       ├── MaxPooling + Dropout (50%)
│       └── Softmax output (6 classes)
│
└── Transfer Learning
    ├── VGG-16
    ├── EfficientNet
    └── ResNet50
```

---

## 📊 Results

### SVM Performance by Feature

| Feature | Overall Accuracy |
|---|---|
| HOG | 62% |
| Sobel Edge | 55% |
| Color Histogram | 62% |
| Color Moment | 48% |

### Deep Learning & Transfer Learning Comparison

| Model | Accuracy | Best Class |
|---|---|---|
| **CNN** | 62% | Katun (F1: 77%) |
| **VGG-16** | 69% | Katun (F1: 93%) |
| **ResNet50** | 87% | Katun (F1: 97%) |
| **EfficientNet** ⭐ | **87%** | Katun (F1: **100%**) |

### 🏆 Best Model: EfficientNet

| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| Corduroy | 87% | 87% | **87%** |
| Denim | 82% | 93% | **88%** |
| Katun | 100% | 100% | **100%** |
| Linen | 81% | 87% | **84%** |
| Organza | 76% | 87% | **81%** |
| Satin | 100% | 67% | **80%** |

> **Key Insight:** Transfer learning models (EfficientNet & ResNet50) significantly outperformed classical SVM, especially on complex-texture classes like Organza and Linen. Katun was the easiest class to classify across all models.

---

## 📁 Project Structure

```
fashion-classification/
│
├── data/
│   ├── raw/                    # Original images (6 class folders)
│   └── processed/              # Resized & preprocessed images
│
├── notebooks/
│   ├── 01_EDA.ipynb            # Exploratory Data Analysis + t-SNE
│   ├── 02_SVM_Classification.ipynb
│   ├── 03_CNN_Classification.ipynb
│   └── 04_TransferLearning.ipynb
│
├── src/
│   ├── preprocessing.py        # Resize, grayscale, normalization
│   ├── feature_extraction.py   # HOG, Sobel, Color Histogram, Moments
│   └── models.py               # Model architectures
│
├── results/
│   ├── confusion_matrices/
│   └── classification_reports/
│
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install tensorflow scikit-learn opencv-python matplotlib seaborn numpy pillow
```

### Run on Google Colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1suaS2fEEuooYOC6aYy43silGYmmyijB?usp=sharing)

### Local Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/fashion-classification.git
cd fashion-classification

# Install dependencies
pip install -r requirements.txt

# Run SVM classification
python src/svm_classify.py

# Run CNN training
python src/train_cnn.py

# Run Transfer Learning
python src/train_transfer.py
```

---

## 🛠 Tech Stack

- **Language:** Python 3.8+
- **Deep Learning:** TensorFlow / Keras
- **Classical ML:** Scikit-learn
- **Image Processing:** OpenCV, PIL
- **Visualization:** Matplotlib, Seaborn
- **Dimensionality Reduction:** t-SNE (sklearn)
- **Notebook:** Google Colab

---

## 📄 License

This project is for academic purposes. Dataset collected from Shopee and Pinterest for educational use only.

---

<p align="center">Made with ❤️ for Machine Learning Course · UMM 2024</p>
