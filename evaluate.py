import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import shap
import pandas as pd

from data_preprocessing import load_telco_data, load_bank_data
from model import ChurnMLP

# The optimal thresholds discovered during your training run
THRESHOLDS = {
    'telco_bce': 0.55,
    'telco_focal': 0.49,
    'bank_bce': 0.59,
    'bank_focal': 0.44
}

def plot_confusion_matrix(targets, preds, title, filename):
    """Generates and saves a clean, professional confusion matrix."""
    cm = confusion_matrix(targets, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Stay', 'Churn'], yticklabels=['Stay', 'Churn'])
    plt.title(title)
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved: {filename}")

def generate_shap_plot(model, X_train, X_test, feature_names, title, filename):
    """Generates and saves SHAP feature importance plots."""
    print(f"Generating SHAP values for {title}...")
    # Added .values to extract raw numpy arrays from DataFrames
    background = torch.tensor(X_train[:100].values, dtype=torch.float32)
    test_samples = torch.tensor(X_test[:200].values, dtype=torch.float32) 
    
    explainer = shap.DeepExplainer(model, background)
    shap_values = explainer.shap_values(test_samples)
    
    plt.figure()
    plt.title(title)
    # SHAP expects numpy arrays
    shap.summary_plot(shap_values, test_samples.numpy(), feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved: {filename}")

def run_evaluation():
    # --- TELCO EVALUATION ---
    print("\n--- Evaluating Telco Models ---")
    X_train_t, X_test_t, y_train_t, y_test_t = load_telco_data(use_smote=False)
    
    # Extract feature names for Telco SHAP (approximate order based on standard pandas get_dummies)
    df_t = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
    df_t.drop(['customerID', 'gender', 'TotalCharges'], axis=1, inplace=True) 
    telco_features = pd.get_dummies(df_t.drop('Churn', axis=1), drop_first=True).columns.tolist()
    telco_features.insert(2, 'TotalCharges') # Rough insertion for visual labeling

    input_dim_t = X_test_t.shape[1]
    
    # Telco BCE
    model_t_bce = ChurnMLP(input_dim_t)
    model_t_bce.load_state_dict(torch.load('best_model_telco_bce.pth'))
    model_t_bce.eval()
    with torch.no_grad():
        probs = model_t_bce(torch.tensor(X_test_t.values, dtype=torch.float32)).numpy()
        preds = (probs >= THRESHOLDS['telco_bce']).astype(int)
        plot_confusion_matrix(y_test_t, preds, 'Telco Dataset (BCE + SMOTE)', 'cm_telco_bce.png')

    # Telco Focal
    model_t_focal = ChurnMLP(input_dim_t)
    model_t_focal.load_state_dict(torch.load('best_model_telco_focal.pth'))
    model_t_focal.eval()
    with torch.no_grad():
        probs = model_t_focal(torch.tensor(X_test_t.values, dtype=torch.float32)).numpy()
        preds = (probs >= THRESHOLDS['telco_focal']).astype(int)
        plot_confusion_matrix(y_test_t, preds, 'Telco Dataset (Focal Loss)', 'cm_telco_focal.png')
        
    generate_shap_plot(model_t_focal, X_train_t, X_test_t, telco_features, 
                       'SHAP: Telco Churn Drivers (Focal)', 'shap_telco_focal.png')


    # --- BANK EVALUATION ---
    print("\n--- Evaluating Bank Models ---")
    X_train_b, X_test_b, y_train_b, y_test_b = load_bank_data(use_smote=False)
    
    # Extract feature names for Bank SHAP
    df_b = pd.read_csv('Churn_Modelling.csv')
    df_b.drop(['RowNumber', 'CustomerId', 'Surname', 'Exited'], axis=1, inplace=True)
    bank_features = pd.get_dummies(df_b, columns=['Geography', 'Gender'], drop_first=True).columns.tolist()

    input_dim_b = X_test_b.shape[1]

    # Bank BCE
    model_b_bce = ChurnMLP(input_dim_b)
    model_b_bce.load_state_dict(torch.load('best_model_bank_bce.pth'))
    model_b_bce.eval()
    with torch.no_grad():
        probs = model_b_bce(torch.tensor(X_test_b.values, dtype=torch.float32)).numpy()
        preds = (probs >= THRESHOLDS['bank_bce']).astype(int)
        plot_confusion_matrix(y_test_b, preds, 'Bank Dataset (BCE + SMOTE)', 'cm_bank_bce.png')

    # Bank Focal
    model_b_focal = ChurnMLP(input_dim_b)
    model_b_focal.load_state_dict(torch.load('best_model_bank_focal.pth'))
    model_b_focal.eval()
    with torch.no_grad():
        probs = model_b_focal(torch.tensor(X_test_b.values, dtype=torch.float32)).numpy()
        preds = (probs >= THRESHOLDS['bank_focal']).astype(int)
        plot_confusion_matrix(y_test_b, preds, 'Bank Dataset (Focal Loss)', 'cm_bank_focal.png')

    generate_shap_plot(model_b_focal, X_train_b, X_test_b, bank_features, 
                       'SHAP: Bank Churn Drivers (Focal)', 'shap_bank_focal.png')

if __name__ == "__main__":
    run_evaluation()
