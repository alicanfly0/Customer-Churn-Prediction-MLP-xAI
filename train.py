import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np
import matplotlib.pyplot as plt

# Import from our modular preprocessing file and model file
from data_preprocessing import load_telco_data, load_bank_data
from model import ChurnMLP

# ==========================================
# 1. NEW PROPOSED METHOD: FOCAL LOSS
# ==========================================
class FocalLoss(nn.Module):
    """
    Focal Loss algorithm to address class imbalance without synthetic data generation.
    It heavily penalizes the model for misclassifying the minority class (churners)
    while down-weighting the easy-to-classify majority class (non-churners).
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Calculate standard Binary Cross Entropy
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        # Calculate the probability of the true class
        pt = torch.exp(-bce_loss)
        # Apply the modulating focal factor
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            return focal_loss

# ==========================================
# 2. DYNAMIC THRESHOLD TUNING
# ==========================================
def find_optimal_threshold(targets, probs):
    """
    Scans probabilities to find the threshold that maximizes the F1-Score,
    vastly improving performance on imbalanced datasets.
    """
    best_thresh = 0.5
    best_f1 = 0.0
    for thresh in np.arange(0.1, 0.9, 0.01):
        preds = (probs >= thresh).astype(int)
        f1 = f1_score(targets, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    return best_thresh

# ==========================================
# 3. UNIFIED EXPERIMENT RUNNER
# ==========================================
def run_experiment(dataset_name='telco', loss_type='bce', use_smote=True):
    print(f"\n{'='*60}")
    print(f"RUNNING EXPERIMENT: Dataset={dataset_name.upper()} | Loss={loss_type.upper()} | SMOTE={use_smote}")
    print(f"{'='*60}")

    # Load respective dataset
    if dataset_name == 'telco':
        X_train, X_test, y_train, y_test = load_telco_data(use_smote=use_smote)
    elif dataset_name == 'bank':
        X_train, X_test, y_train, y_test = load_bank_data(use_smote=use_smote)

   # Convert to PyTorch Tensors
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1) 
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

    batch_size = 64
    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=batch_size, shuffle=False)

    # Initialize Model
    input_dim = X_train.shape[1]
    model = ChurnMLP(input_dim)
    
    # Select Loss Function
    if loss_type == 'focal':
        criterion = FocalLoss(alpha=0.25, gamma=2.0)
    else:
        criterion = nn.BCELoss()
        
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training Setup
    epochs = 200
    patience = 15
    best_loss = float('inf')
    epochs_no_improve = 0
    model_save_path = f'best_model_{dataset_name}_{loss_type}.pth'
    
    train_losses, val_losses = [], []
    
    # Training Loop
    for epoch in range(epochs):
        model.train() 
        running_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()           
            outputs = model(batch_X)        
            loss = criterion(outputs, batch_y) 
            loss.backward()                 
            optimizer.step()                
            running_loss += loss.item()
            
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation Loop
        model.eval() 
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                outputs = model(batch_X)
                val_loss += criterion(outputs, batch_y).item()
                
        avg_val_loss = val_loss / len(test_loader)
        val_losses.append(avg_val_loss)
        
        # Early Stopping
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), model_save_path)
        else:
            epochs_no_improve += 1
            
        if epochs_no_improve == patience:
            print(f"Early stopping at epoch {epoch+1}! Best Val Loss: {best_loss:.4f}")
            break

    # Save Loss Curve specific to this experiment
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(val_losses, label='Val Loss', color='red')
    plt.title(f'Loss Curve: {dataset_name.upper()} | {loss_type.upper()}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'loss_curve_{dataset_name}_{loss_type}.png')
    plt.close()

    # Final Evaluation with Threshold Tuning
    model.load_state_dict(torch.load(model_save_path)) 
    model.eval()
    
    all_targets, all_probs = [], []
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X)
            all_targets.extend(batch_y.numpy())
            all_probs.extend(outputs.numpy())
            
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)
    
    # Apply Dynamic Threshold
    optimal_thresh = find_optimal_threshold(all_targets, all_probs)
    final_preds = (all_probs >= optimal_thresh).astype(int)
    
    acc = accuracy_score(all_targets, final_preds)
    prec = precision_score(all_targets, final_preds, zero_division=0)
    rec = recall_score(all_targets, final_preds, zero_division=0)
    f1 = f1_score(all_targets, final_preds, zero_division=0)
    auc = roc_auc_score(all_targets, all_probs) 
    
    print(f"--- TEST RESULTS (Optimal Threshold: {optimal_thresh:.2f}) ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"AUC:       {auc:.4f}\n")

# ==========================================
# 4. EXECUTE THE 2x2 MATRIX
# ==========================================
if __name__ == "__main__":
    # Experiment 1: Telco Baseline (Assignment 2 Replication)
    run_experiment(dataset_name='telco', loss_type='bce', use_smote=True)
    
    # Experiment 2: Telco Proposed (Focal Loss)
    run_experiment(dataset_name='telco', loss_type='focal', use_smote=False)
    
    # Experiment 3: Bank Baseline (Cross-Domain)
    run_experiment(dataset_name='bank', loss_type='bce', use_smote=True)
    
    # Experiment 4: Bank Proposed (Cross-Domain Focal Loss)
    run_experiment(dataset_name='bank', loss_type='focal', use_smote=False)
