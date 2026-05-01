# Optimized Iris Recognition System using CNN Model Compression Techniques

## Overview

This project presents an efficient biometric authentication system based on **Iris Recognition** using **Convolutional Neural Networks (CNNs)** and advanced **model compression techniques**. The goal is to achieve high recognition accuracy while reducing computational cost, memory usage, and inference time for deployment in real-world and low-resource environments.

The system combines image preprocessing, deep feature learning, knowledge distillation, pruning, and quantization to create a lightweight yet highly accurate iris recognition framework.

---

## Features

- Iris-based biometric authentication
- Image preprocessing and enhancement
- CNN-based feature extraction
- Teacher-Student Knowledge Distillation
- Structured Pruning
- Dynamic Quantization
- High accuracy with low computational complexity
- Cross-validation based evaluation
- Suitable for mobile and embedded systems

---

## Dataset Used

**CASIA Iris Dataset**

A standard benchmark dataset widely used for iris recognition research containing iris images under different conditions.

---

## Preprocessing Pipeline

The following image preprocessing steps were used:

1. Grayscale Conversion  
2. Gaussian Filtering  
3. Median Filtering  
4. Bilateral Filtering  
5. CLAHE (Contrast Enhancement)  
6. Pupil Detection using Hough Circle Transform  
7. Iris Segmentation  
8. Normalization  
9. Resize to 128x128

---

## Model Architecture

### Teacher Model
- ResNet-50 (Pretrained CNN)
- High feature learning capability

### Student Model
- Lightweight CNN
- Trained using Knowledge Distillation

### Compression Techniques
- Structured Pruning
- Dynamic Quantization

---

## Performance Metrics

The system was evaluated using:

- Accuracy
- ROC Curve
- AUC (Area Under Curve)
- EER (Equal Error Rate)
- FAR (False Acceptance Rate)
- FRR (False Rejection Rate)
- Confusion Matrix
- 5-Fold Cross Validation

---

## Results

| Model | Accuracy | AUC | EER |
|------|----------|------|------|
| Teacher Model | 99.00% | 0.9960 | 2.90% |
| Student Model | 97.80% | 0.9956 | 3.11% |
| Pruned Model | 97.10% | 0.9950 | 3.20% |
| Quantized Model | 96.50% | 0.9940 | 3.30% |

---

## Technologies Used

- Python
- PyTorch
- OpenCV
- NumPy
- Scikit-learn
- Matplotlib
- Pandas

---

## Project Structure

```bash
├── dataset/
├── preprocessing/
├── models/
├── training/
├── evaluation/
├── results/
├── notebooks/
├── README.md
