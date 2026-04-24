import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np
import matplotlib.pyplot as plt  # Added for visual reporting

# Import from our previous files
from data_preprocessing import load_and_preprocess_data
from model import ChurnMLP

def train_model():
    # 1. Load the preprocessed data
    print("Loading data...")
    # The study utilizes the Telco-Customer-Churn dataset for its analysis[cite: 18, 377].
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    
    # Convert numpy arrays to PyTorch tensors
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1) 
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

    # Create DataLoaders for batching
    batch_size = 64
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 2. Initialize the Model, Loss, and Optimizer
    input_dim = X_train.shape[1]
    # Architecture follows the Multilayer Perceptron (MLP) model introduced in the research[cite: 16, 70].
    model = ChurnMLP(input_dim)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 3. Training Loop Setup
    epochs = 200
    patience = 15
    best_loss = float('inf')
    epochs_no_improve = 0
    
    # Lists to record loss for your report's "Reproducibility Analysis"
    train_losses = []
    val_losses = []
    
    print("\nStarting Training...")
    
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
        
        # 4. Validation / Early Stopping Check
        model.eval() 
        val_loss = 0.0
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                outputs = model(batch_X)
                # Calculating loss on the 20% test subset reserved for evaluation[cite: 382].
                loss_val = criterion(outputs, batch_y)
                val_loss += loss_val.item()
                
        avg_val_loss = val_loss / len(test_loader)
        val_losses.append(avg_val_loss)
        
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), 'best_churn_model.pth')
        else:
            epochs_no_improve += 1
            
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
            
        if epochs_no_improve == patience:
            print(f"\nEarly stopping triggered at epoch {epoch+1}! Best Val Loss: {best_loss:.4f}")
            break

    # --- NEW: Generate Loss Curve Graph ---
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='red')
    plt.title('MLP Training Progress (Reproduced Baseline)')
    plt.xlabel('Epochs')
    plt.ylabel('BCE Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_curve.png') 
    print("\n[Visual Generated] Training graph saved as 'loss_curve.png'.")

    # 5. Final Evaluation on Test Set
    print("\nEvaluating Best Model on Test Set...")
    model.load_state_dict(torch.load('best_churn_model.pth')) 
    model.eval()
    
    all_preds = []
    all_targets = []
    all_probs = [] # Storing raw probabilities for a more precise AUC calculation
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X)
            preds = (outputs >= 0.5).float()
            all_preds.extend(preds.numpy())
            all_targets.extend(batch_y.numpy())
            all_probs.extend(outputs.numpy())
            
    # Standard metrics identified for evaluating churn classification performance[cite: 135].
    acc = accuracy_score(all_targets, all_preds)
    prec = precision_score(all_targets, all_preds)
    rec = recall_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds)
    auc = roc_auc_score(all_targets, all_probs) 
    
    print("-" * 30)
    print(f"Final Test Metrics:")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"AUC:       {auc:.4f}")
    print("-" * 30)

if __name__ == "__main__":
    train_model()