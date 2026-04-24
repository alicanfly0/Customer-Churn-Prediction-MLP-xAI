import torch
import torch.nn as nn

class ChurnMLP(nn.Module):
    def __init__(self, input_dim):
        """
        Multilayer Perceptron for Customer Churn Prediction.
        Architecture: Input -> 256 -> 128 -> 64 -> 32 -> 1
        Regularization: Batch Normalization and Dropout (0.3)
        """
        super(ChurnMLP, self).__init__()
        
        # Layer 1: 256 units
        self.layer1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.3)
        
        # Layer 2: 128 units
        self.layer2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.3)
        
        # Layer 3: 64 units
        self.layer3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(p=0.3)
        
        # Layer 4: 32 units
        self.layer4 = nn.Linear(64, 32)
        self.bn4 = nn.BatchNorm1d(32)
        self.relu4 = nn.ReLU()
        self.dropout4 = nn.Dropout(p=0.3)
        
        # Output Layer: 1 unit (Binary Classification)
        self.output_layer = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass of the neural network.
        """
        # Pass through Layer 1
        x = self.layer1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        
        # Pass through Layer 2
        x = self.layer2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        
        # Pass through Layer 3
        x = self.layer3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.dropout3(x)
        
        # Pass through Layer 4
        x = self.layer4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.dropout4(x)
        
        # Pass through Output Layer
        x = self.output_layer(x)
        x = self.sigmoid(x)
        
        return x

# Quick test block to ensure the network initializes correctly
if __name__ == "__main__":
    # Assuming ~30 features after one-hot encoding categorical variables
    dummy_input_dim = 30 
    model = ChurnMLP(input_dim=dummy_input_dim)
    
    # Create a dummy batch of 64 customers
    dummy_data = torch.randn(64, dummy_input_dim) 
    
    # Pass data through the model
    predictions = model(dummy_data)
    
    print(model)
    print(f"\nOutput shape: {predictions.shape} (Expected: [64, 1] for a batch of 64)")