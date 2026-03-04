# 🩻 Chest X-Ray Pneumonia Classification With Deep Learning

Binary classification of chest X-ray images (Normal vs. Pneumonia) using Convolutional Neural Networks and Transfer Learning (ResNet-18).

## 🛠️ Tech Stack & Tools

**Languages & Libraries:**
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![TorchMetrics](https://img.shields.io/badge/TorchMetrics-FF7000?style=for-the-badge&logo=pytorch&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)

---

## 🎯 Objective
The objective of this project is to build a robust Deep Learning model using PyTorch to assist radiologists in pediatric patient triage. The optimization focused heavily on maximizing **Recall (Sensitivity)**, ensuring that no sick patient is dismissed without treatment (drastic minimization of false negatives).

- **Task:** Binary Classification (0: Normal | 1: Pneumonia).
- **Model:** ResNet-18 Architecture.
- **Loss Function:** `BCEWithLogitsLoss` coupled with a single-neuron output for optimal numerical stability.

## 📊 Dataset Description
The model was trained on the "Chest X-Ray Images (Pneumonia)" dataset, which contains 5,856 validated chest X-ray images.

- **Distribution:** The training set features a natural class imbalance, with a significantly higher number of pneumonia cases compared to normal exams.
- **Processing:** Images were resized to 224x224 pixels and normalized using standard ImageNet channel statistics (Mean: `[0.485, 0.456, 0.406]`, Std: `[0.229, 0.224, 0.225]`).

## 🧠 Strategy and Methodology
The project leveraged a **Transfer Learning** architecture. The feature extraction base of the ResNet-18 was frozen, and its final classification head was replaced with a linear layer (`nn.Linear`) to output raw logits.

The training loop utilized the `AdamW` optimizer, strictly focused on the newly initialized parameters. The process was controlled by an Early Stopping mechanism to prevent overfitting, saving the model weights that maximized the sensitivity metric on the validation set.

### The Data Augmentation Paradox
Two data preparation pipelines were tested: using default pre-trained transformations versus applying custom Data Augmentation (random rotations and crops). 
The experiments revealed that introducing custom Data Augmentation **did not** improve the final results. Because medical X-rays are strictly standardized and aligned, introducing geometric noise proved to be counterproductive, hindering the convolutional network's ability to extract structural anatomical context.

## 📈 Results and Conclusion
The model's performance was evaluated on an independent Test Set (624 images), achieving the following global metrics:

- **Recall (Sensitivity):** 98.46%
- **Precision:** 73.42%

### Clinical Interpretation ("Safe Screening")
The architecture successfully met its primary requirement as a preliminary diagnostic tool. A Recall close to 98.5% demonstrates that the model is highly proficient at identifying infected patients. 
Conversely, the 73.4% Precision translates to a higher rate of false positives—a predictable consequence of the imbalanced training data. In a clinical setting, this conservative approach is statistically desirable: it is always preferable to request an additional exam for a healthy patient than to misdiagnose a patient with severe pneumonia.

