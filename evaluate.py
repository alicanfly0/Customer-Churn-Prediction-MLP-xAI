import torch
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Import from our previous files
from data_preprocessing import load_and_preprocess_data
from model import ChurnMLP

def evaluate_and_explain():
    print("Loading data and model...")
    # We need the DataFrames here to keep feature names for the SHAP plots
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    feature_names = X_test.columns.tolist()
    
    # Convert test data to tensors
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)
    
    # Load the trained model
    input_dim = X_train.shape[1]
    model = ChurnMLP(input_dim)
    model.load_state_dict(torch.load('best_churn_model.pth'))
    model.eval() # Set to evaluation mode!

    # 1. Final Model Performance Evaluation
    print("\nCalculating Final Test Metrics...")
    with torch.no_grad():
        outputs = model(X_test_tensor)
        preds = (outputs >= 0.5).float()
    
    # Convert back to numpy for sklearn metrics
    outputs_np = outputs.numpy()
    preds_np = preds.numpy()
    targets_np = y_test_tensor.numpy()
    
    acc = accuracy_score(targets_np, preds_np)
    prec = precision_score(targets_np, preds_np)
    rec = recall_score(targets_np, preds_np)
    f1 = f1_score(targets_np, preds_np)
    auc = roc_auc_score(targets_np, outputs_np) # AUC uses probabilities, not strict 0/1 preds
    
    print("-" * 30)
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"AUC:       {auc:.4f}")
    print("-" * 30)

    # 2. SHAP Explainability (The XAI Component)
    print("\nGenerating SHAP explanations (this may take a minute)...")
    
    # SHAP DeepExplainer needs a "background" dataset to integrate over. 
    # Using 100 random training samples to keep computation time low.
    background_data = torch.tensor(X_train.sample(n=100, random_state=42).values, dtype=torch.float32)
    explainer = shap.DeepExplainer(model, background_data)
    
    # We explain the predictions for a sample of the test set (e.g., 200 customers)
    test_sample = torch.tensor(X_test.sample(n=200, random_state=42).values, dtype=torch.float32)
    shap_values = explainer.shap_values(test_sample)
    
    # Generate the SHAP Summary Plot
    plt.figure(figsize=(10, 8))
    
    # Extract values and force them into a strict 2D shape (200 instances, 29 features)
    if isinstance(shap_values, list):
        shap_values_to_plot = np.array(shap_values[0])
    else:
        shap_values_to_plot = np.array(shap_values)
        
    # Strip away PyTorch's trailing dimension so SHAP plots all 29 columns correctly
    shap_values_to_plot = shap_values_to_plot.reshape(test_sample.shape[0], test_sample.shape[1])
    
    shap.summary_plot(
        shap_values_to_plot, 
        features=test_sample.numpy(), 
        feature_names=feature_names, 
        show=False # Don't block the script from finishing
    )
    
    # Save the plot for the final report
    plt.savefig('shap_summary_plot.png', bbox_inches='tight', dpi=300)
    print("\nSuccess! SHAP summary plot saved as 'shap_summary_plot.png'.")

if __name__ == "__main__":
    evaluate_and_explain()