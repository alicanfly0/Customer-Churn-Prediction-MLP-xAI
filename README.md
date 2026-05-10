# Deep Learning for Customer Churn Prediction (MLP & XAI)

This repository contains the code and documentation for a Deep Learning research project predicting customer churn in the telecommunications industry. The project reproduces the methodology of a recent high-performing baseline study and extends it using Explainable AI (XAI) to ensure transparency in business decision-making.

**Team Members:**
* Ali Waseem (23i-2630)
* Muhammad Mudassir (23i-2562)
* Talha Zaheer (23i-2609)

## Project Overview
Customer churn is a critical financial challenge. Retaining existing customers is significantly more cost-effective than acquiring new ones. This project implements a **Multilayer Perceptron (MLP)** using PyTorch to predict churn on the IBM Telco Customer Churn dataset. 

**Key Features:**
* **Rigorous Data Pipeline:** Handled categorical encoding, Z-score normalization, and class imbalance using SMOTE. Crucially, SMOTE was applied *only* to the training subset to prevent data leakage.
* **Deep Neural Network:** A 4-layer PyTorch MLP utilizing ReLU, Batch Normalization, and Dropout (0.3) to prevent overfitting on tabular data.
* **Explainable AI (SHAP):** Integrated SHapley Additive exPlanations to interpret the model's predictions, identifying "Monthly Charges" and "Fiber Optic Internet" as primary churn drivers.

## Repository Structure
* `src/dataset.py`: Handles data loading, cleaning, feature scaling, and train/test splitting.
* `src/model.py`: Defines the PyTorch neural network architecture (`ChurnMLP`).
* `src/utils.py`: Utility functions.
* `train.py`: Executes the training loop with the Adam optimizer, tracks Binary Cross Entropy (BCE) loss, and enforces Early Stopping.
* `inference.py`: Generates evaluation metrics (Accuracy, F1, AUC) and executes the SHAP analysis.
* `plot_results.py`: Utility script to generate training vs. validation loss curves.
* `checkpoints/best_churn_model.pth`: Saved weights of the optimal model post-training.
* `results/`: Contains evaluation metrics, confusion matrices, and SHAP plots.
* `notebooks/01_inference_demo.ipynb`: Jupyter notebook for interactive inference.

## Visualizations
**1. Model Convergence**
The model successfully minimizes BCE loss without heavy overfitting, reaching a realistic performance ceiling (AUC: 0.8174) absent of the data leakage found in the baseline paper.
*(See `loss_curve.png`)*

**2. SHAP Feature Importance**
Post-hoc explainability confirms the model learned logical, real-world business patterns rather than memorizing noise.
*(See `shap_summary_plot.png`)*

## How to Run
1. Ensure the dataset (`WA_Fn-UseC_-Telco-Customer-Churn.csv`) is in the root directory.
2. Install dependencies: `pip install -r requirements.txt`
3. Configure training settings in `config.yaml`
4. Train the model: `python train.py`
5. Evaluate and generate SHAP plots: `python inference.py`

---
*Developed for CS-4112 Deep Learning*