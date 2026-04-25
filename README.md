# Customer Churn Prediction: Algorithmic Balancing & Cross-Domain Generalization

**Course:** Deep Learning (Semester 6)  
**Institution:** FAST NUCES  
**Team Members:** * Ali Waseem (23i-2630)
* Muhammad Mudassir (23i-2562)
* Talha Zaheer (23i-2609)

## Project Overview
This repository contains the code and research for Assignment 3, which extends a baseline Multilayer Perceptron (MLP) telecom churn prediction model. This project moves beyond synthetic data generation (SMOTE) by implementing **Focal Loss** to algorithmically handle severe class imbalances. Furthermore, the architecture is evaluated on a cross-domain financial dataset (Kaggle Bank Customer Churn) to prove industry-agnostic generalization.

## Repository Structure
* `data_preprocessing.py`: Contains modular functions to load, clean, scale, and encode both the IBM Telco and Kaggle Bank datasets. Includes toggles for SMOTE vs. un-synthesized data.
* `model.py`: Defines the PyTorch 4-layer MLP architecture (with BatchNorm and Dropout).
* `train.py`: The core experimental runner. Executes a 2x2 matrix evaluating BCE+SMOTE vs. Focal Loss across both datasets. Implements dynamic threshold tuning to maximize F1-Scores.
* `evaluate.py`: Generates post-training visualizations, including confusion matrices and SHAP (Explainable AI) feature importance plots.
* `*.pth` files: The saved PyTorch model weights for the four optimal experiments.
* `*.png` files: Generated loss curves, confusion matrices, and SHAP plots.
* `i232630_i232562_i232609_DL_Assingment3.pdf`: The final 12-page comprehensive research report.

## How to Run the Code
Ensure you have the required dependencies installed (`torch`, `pandas`, `numpy`, `scikit-learn`, `imblearn`, `shap`, `matplotlib`, `seaborn`).

**1. Train the Models**
To execute the 2x2 experimental matrix, train the networks, and find the optimal probability thresholds, run:
`python train.py`
*This will automatically generate the 4 loss curves and save the 4 `.pth` weight files.*

**2. Evaluate and Generate Visuals**
To test the saved models against the test sets and generate the confusion matrices and SHAP plots, run:
`python evaluate.py`

## Key Findings
1. **Focal Loss Superiority:** Replacing SMOTE with Focal Loss successfully smoothed the validation convergence and significantly reduced False Positives across both datasets, resulting in higher overall F1-Scores.
2. **Cross-Domain Generalization:** The architecture optimized for telecommunications easily adapted to the banking sector, achieving a peak AUC of 0.8636 without any structural modifications to the hidden layers.
