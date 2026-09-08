# Automated Brain Tumor Detection from MRI Images Using Transfer Learning with ResNet50

An automated brain tumor detection and classification project based on
**MRI brain images**, **ResNet50 transfer learning**, **K-Means
clustering**, and **Fuzzy Logic-based prediction confidence analysis**.

> **Academic Project:** Final Year Research Project, B.Tech. Computer
> Science & Engineering (Data Science)\
> **Academic Year:** 2025--2026\
> **Institution:** Siksha 'O' Anusandhan (Deemed to be University),
> Bhubaneswar, Odisha, India

------------------------------------------------------------------------

## 📌 Project Overview

Brain tumor detection from MRI images is a challenging medical
image-analysis problem because tumor appearance can vary in size, shape,
texture, intensity, and location. Manual MRI examination can also be
time-consuming.

This project proposes an automated framework that combines image
preprocessing, data augmentation, K-Means clustering, and a pretrained
**ResNet50** convolutional neural network. Transfer learning with
ImageNet-pretrained weights is used for feature extraction and
classification, followed by fine-tuning of higher layers.

A Fuzzy Logic component is included to categorize prediction confidence
into:

-   **High Tumor Confidence**
-   **Moderate Tumor Confidence**
-   **Low Tumor Confidence**
-   **Non-Tumor**

The report states that the framework achieved **90.84% classification
accuracy** during the reported experimental evaluation.

------------------------------------------------------------------------

## 🎯 Objectives

The main objectives of the project are:

1.  Develop an automated brain tumor detection framework using MRI
    scans.
2.  Apply transfer learning using the pretrained ResNet50 model.
3.  Perform MRI image preprocessing and data augmentation.
4.  Use K-Means clustering for MRI image grouping and analysis.
5.  Incorporate Fuzzy Logic for prediction-confidence analysis.
6.  Improve classification accuracy and robustness.
7.  Evaluate the model using Accuracy, Precision, Recall, F1-Score, and
    Confusion Matrix.

------------------------------------------------------------------------

## 🧠 Proposed Workflow

``` text
MRI Brain Image
       │
       ▼
Image Preprocessing
 ├── Resize to 224 × 224
 ├── Normalization
 ├── Pixel Value Scaling
 └── Noise Reduction
       │
       ▼
Data Augmentation
 ├── Rotation
 ├── Horizontal Flipping
 ├── Zooming
 └── Shearing
       │
       ▼
K-Means Clustering
       │
       ▼
ResNet50 Transfer Learning
 ├── ImageNet Pretrained Weights
 ├── Feature Extraction
 └── Fine-Tuning of Higher Layers
       │
       ▼
Global Average Pooling
       │
       ▼
Dense Layer + ReLU
       │
       ▼
Dropout
       │
       ▼
Sigmoid Output
       │
       ▼
Tumor / Non-Tumor
       │
       ▼
Fuzzy Logic Confidence Analysis
```

------------------------------------------------------------------------

## 🗂️ Dataset

The project report describes a Kaggle MRI brain-image dataset
containing:

  Parameter                   Value
  --------------------------- -----------------------
  Total MRI Images            3,762
  Image Format                JPG
  Medical Imaging Technique   MRI
  Classification              Binary Classification
  Classes                     Tumor / Non-Tumor
  Source                      Kaggle Dataset

The dataset is used for training, testing, and performance evaluation.

The report cites the Kaggle source as the dataset/code repository by
**Sinan Shereef**, titled *Brain Tumor Detection from MRI Images Using
CNN*.

> **Dataset note:** The MRI dataset itself should not be committed to
> GitHub unless its licensing and redistribution terms explicitly allow
> it. Keep the dataset outside the repository and document how to obtain
> it.

------------------------------------------------------------------------

## 🔧 Technologies Used

### Programming Language

-   Python

### Deep Learning

-   TensorFlow
-   Keras
-   ResNet50
-   Transfer Learning

### Machine Learning

-   Scikit-learn
-   K-Means Clustering
-   Fuzzy Logic confidence analysis

### Image Processing

-   OpenCV
-   NumPy

### Visualization

-   Matplotlib

### Development Environment

-   Google Colab
-   GPU-based training

------------------------------------------------------------------------

## 🏗️ Model Architecture

The proposed classification architecture contains:

1.  **ResNet50 Pretrained Base Model**
    -   ImageNet pretrained weights
    -   Used as the main feature extractor
    -   Input images are resized to 224 × 224 pixels
2.  **Global Average Pooling**
    -   Converts feature maps into a feature vector
    -   Reduces the feature representation before classification
3.  **Dense Layer**
    -   Learns tumor-related features
    -   Uses ReLU activation
4.  **Dropout Layer**
    -   Used as a regularization technique
    -   Helps reduce overfitting
5.  **Sigmoid Output Layer**
    -   Produces a probability for binary classification
    -   Used for Tumor / Non-Tumor prediction

------------------------------------------------------------------------

## 🔄 Transfer Learning

Instead of training a deep CNN from scratch, the project uses a
**ResNet50 model pretrained on ImageNet**.

The reported approach:

-   Uses pretrained ResNet50 weights.
-   Uses the lower layers to retain previously learned visual
    representations.
-   Fine-tunes higher layers using MRI brain images.
-   Uses the extracted deep features for tumor/non-tumor classification.

The residual-learning design of ResNet50 helps maintain feature and
gradient flow in a deep network.

------------------------------------------------------------------------

## 🧹 Image Preprocessing

The reported preprocessing pipeline includes:

### 1. Image Resizing

MRI images are resized to:

``` text
224 × 224 pixels
```

### 2. Image Normalization

Pixel values are normalized to improve consistency during model
training.

The report gives the normalization formula:

``` text
X_norm = (X - X_min) / (X_max - X_min)
```

### 3. Pixel Value Scaling

Pixel values are scaled into a narrower numerical range to improve
optimization stability.

### 4. Noise Reduction

Noise and image artifacts are reduced before feature extraction and
classification.

------------------------------------------------------------------------

## 🔁 Data Augmentation

To increase dataset diversity and reduce overfitting, the reported
framework uses:

-   Rotation
-   Horizontal Flipping
-   Zooming
-   Shearing

These transformations help the model learn more robust visual
representations from MRI images.

------------------------------------------------------------------------

## 🔬 K-Means Clustering

K-Means is used as an unsupervised learning technique for grouping MRI
images according to feature similarity.

The reported clustering procedure consists of:

1.  Initialize cluster centroids.
2.  Assign MRI samples to the nearest centroid.
3.  Recalculate cluster centroids.
4.  Repeat until convergence.

The project describes K-Means as an image-grouping and feature-analysis
component before classification with the ResNet50 transfer-learning
network.

------------------------------------------------------------------------

## 🧩 Fuzzy Logic Confidence Analysis

The framework adds Fuzzy Logic to interpret prediction confidence.

The reported confidence categories are:

  Confidence Category         Meaning
  --------------------------- ------------------------------------------
  High Tumor Confidence       Strong tumor prediction confidence
  Moderate Tumor Confidence   Intermediate tumor prediction confidence
  Low Tumor Confidence        Weak tumor prediction confidence
  Non-Tumor                   Prediction categorized as non-tumor

**Important:** The report identifies these categories but does not
provide enough explicit numerical membership-function thresholds/rules
to reproduce the exact fuzzy system from the report alone. Therefore,
this README does not invent those thresholds.

------------------------------------------------------------------------

## ⚙️ Training Configuration

The report's **Table 4.2** gives the following training configuration:

  Parameter                 Value
  ------------------------- -----------------------
  Input Image Size          224 × 224
  Batch Size                16
  Epochs                    10
  Optimizer                 Adam
  Learning Rate             0.0001
  Loss Function             Binary Crossentropy
  Activation Functions      ReLU and Sigmoid
  Transfer Learning Model   ResNet50
  Framework                 TensorFlow / Keras
  Classification Type       Binary Classification

### ⚠️ Report consistency note

There is an inconsistency in the report: an earlier subsection states
**batch size = 8 and epochs = 20**, while Table 4.2 states **batch size
= 16 and epochs = 10**. This README uses the values from **Table 4.2**
as the reported configuration and does not silently change the report's
numbers.

------------------------------------------------------------------------

## 📊 Evaluation Metrics

The project evaluates classification using:

### Accuracy

``` text
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

### Precision

``` text
Precision = TP / (TP + FP)
```

### Recall

``` text
Recall = TP / (TP + FN)
```

### F1-Score

``` text
F1 = 2 × Precision × Recall / (Precision + Recall)
```

### Confusion Matrix

The confusion matrix is used to analyze:

-   True Positive (TP)
-   True Negative (TN)
-   False Positive (FP)
-   False Negative (FN)

------------------------------------------------------------------------

## 📈 Reported Results

The report states a test/evaluation accuracy of:

### **90.84%**

with a reported final loss of:

### **0.2170**

The report's training-performance table contains the following recorded
values:

    Epoch   Accuracy     Loss   Validation Accuracy   Validation Loss
  ------- ---------- -------- --------------------- -----------------
        1     0.8897   0.2670                0.8738            0.2967
        2     0.8933   0.2426                0.9017            0.2546
        3     0.8960   0.2454                0.9084            0.2170
        4     0.8983   0.2509                0.8114            0.4053
        5     0.9020   0.2305                0.8871            0.2691
        6     0.9013   0.2357                0.8818            0.2657

### Classification Report

The report provides:

  Class            Precision   Recall   F1-Score   Support
  -------------- ----------- -------- ---------- ---------
  Class 0               0.90     0.94       0.92       419
  Class 1               0.92     0.87       0.89       334
  Macro Avg             0.91     0.90       0.91       753
  Weighted Avg          0.91     0.91       0.91       753

The report describes the classification performance as balanced between
the two classes.

------------------------------------------------------------------------

## 🖼️ Expected Repository Structure

A clean GitHub repository can be organized as:

``` text
brain-tumor-detection-resnet50/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── notebooks/
│   └── brain_tumor_detection_resnet50.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── augmentation.py
│   ├── clustering.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── fuzzy_confidence.py
│
├── models/
│   └── README.md
│
├── results/
│   ├── README.md
│   ├── accuracy_loss.png
│   ├── confusion_matrix.png
│   └── predictions/
│
├── docs/
│   └── project_report.pdf
│
└── data/
    └── README.md
```

The exact source files should match the code that is actually present in
the repository.

------------------------------------------------------------------------

## 🚀 Installation

Clone the repository:

``` bash
git clone <YOUR-GITHUB-REPOSITORY-URL>
cd brain-tumor-detection-resnet50
```

Create a virtual environment:

``` bash
python -m venv venv
```

Activate it on Windows:

``` bash
venv\Scripts\activate
```

Activate it on Linux/macOS:

``` bash
source venv/bin/activate
```

Install dependencies:

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

## 📦 Suggested `requirements.txt`

``` text
tensorflow
keras
numpy
opencv-python
matplotlib
scikit-learn
pandas
jupyter
```

If your actual notebook/code uses additional libraries, add those
packages to `requirements.txt`.

------------------------------------------------------------------------

## 📥 Dataset Setup

Because the report uses a Kaggle MRI dataset, keep the dataset outside
GitHub unless redistribution is permitted.

A typical local structure can be:

``` text
data/
├── tumor/
│   ├── image_001.jpg
│   ├── image_002.jpg
│   └── ...
│
└── non_tumor/
    ├── image_001.jpg
    ├── image_002.jpg
    └── ...
```

Update the dataset paths in the training notebook/script according to
your actual folder names.

------------------------------------------------------------------------

## ▶️ Running the Project

### Using Google Colab

1.  Upload/open the notebook in Google Colab.
2.  Enable GPU runtime if required.
3.  Upload or mount the dataset.
4.  Run preprocessing.
5.  Run data augmentation.
6.  Run K-Means clustering.
7.  Build the ResNet50 transfer-learning model.
8.  Train the model.
9.  Evaluate using the reported metrics.
10. Run prediction on sample MRI images.
11. Apply the confidence-analysis component.

### Using Python Locally

If the repository contains a training script:

``` bash
python src/train.py
```

For evaluation:

``` bash
python src/evaluate.py
```

These commands should be changed to match the actual filenames in the
repository.

------------------------------------------------------------------------

## 🔍 Prediction

The system performs binary classification:

``` text
MRI Image
    ↓
Preprocessing
    ↓
ResNet50
    ↓
Feature Extraction
    ↓
Classification
    ↓
Tumor / Non-Tumor
    ↓
Confidence Analysis
```

Example conceptual output:

``` text
Prediction: Tumor
Confidence Category: High Tumor Confidence
```

The exact confidence output depends on the implemented fuzzy-logic
rules.

------------------------------------------------------------------------

## 📁 Results and Visualizations

The project report includes:

-   Training and validation accuracy graph
-   Training and validation loss graph
-   Confusion matrix
-   Tumor prediction samples
-   Non-tumor prediction samples
-   Classification report

These can be stored under the `results/` directory in the GitHub
repository.

------------------------------------------------------------------------

## 👥 Team Members

  -----------------------------------------------------------------------
  Name                                Contribution
  ----------------------------------- -----------------------------------
  **Priyabrata Senapati**             Literature review, dataset
                                      analysis, MRI preprocessing,
                                      augmentation, K-Means clustering,
                                      training and evaluation

  **Abhishek Mohapatra**              ResNet50 model development,
                                      transfer learning, fine-tuning,
                                      Fuzzy Logic confidence system,
                                      evaluation metrics, documentation

  **Dibyam Jyoti Pradhan**            Dataset collection and
                                      organization, preprocessing and
                                      normalization, testing and
                                      validation, presentation

  **Nitesh Kumar Nayak**              Model optimization and analysis,
                                      graph plotting, result
                                      visualization, system testing and
                                      implementation, presentation
  -----------------------------------------------------------------------

------------------------------------------------------------------------

## 🧪 Limitations

According to the project report, important limitations include:

-   The effectiveness of the model depends on the size and diversity of
    the MRI dataset.
-   The dataset may not represent all variations encountered in real
    clinical environments.
-   MRI scanner and imaging-protocol differences can affect
    generalization.
-   Noise, intensity variations, tumor size, contrast, and complex tumor
    structures can affect prediction.
-   ResNet50 training requires substantial computational and memory
    resources.
-   Deep learning models can be difficult to interpret because of their
    black-box nature.

------------------------------------------------------------------------

## 🔮 Future Scope

The report identifies several future directions:

-   Use larger training datasets.
-   Explore deeper neural-network architectures.
-   Extend the system to multi-class brain tumor classification.
-   Improve model explainability using explainable AI techniques.
-   Develop real-time prediction systems for clinical applications.
-   Explore hybrid approaches to improve classification performance.

------------------------------------------------------------------------

## ⚠️ Medical Disclaimer

This project is an **academic/research prototype** and is not a medical
diagnostic device.

Predictions from the model should **not** be used as a substitute for
professional medical examination, radiological interpretation, or
clinical decision-making.

------------------------------------------------------------------------

## 📚 References

The project report references literature related to:

-   Deep learning
-   ResNet and residual learning
-   Transfer learning
-   K-Means clustering
-   Fuzzy sets and fuzzy logic
-   Medical image classification
-   TensorFlow/Keras
-   The Kaggle MRI brain-tumor dataset

The complete bibliography is available in the project report.

------------------------------------------------------------------------

## 📄 Project Report

The complete final-year project report can be placed in:

``` text
docs/project_report.pdf
```

Report title:

**Automated Brain Tumor Detection from MRI Images Using Transfer
Learning with ResNet50**

------------------------------------------------------------------------

## ⭐ Acknowledgement

We express our sincere gratitude to our project supervisor **Ms. Padmaja
Patel**, Assistant Professor, Centre for Data Science, for her guidance,
support, and encouragement throughout the project.

We also thank the faculty members of the Centre for Data Science and the
Department of Computer Science & Engineering, Siksha 'O' Anusandhan
(Deemed to be University), for providing academic support and resources.

------------------------------------------------------------------------

## 📌 Project Status

**Status:** Completed academic research project

**Primary Task:** Binary MRI brain-image classification

**Model:** ResNet50 Transfer Learning

**Reported Accuracy:** 90.84%

**Framework:** TensorFlow / Keras

**Environment:** Google Colab
