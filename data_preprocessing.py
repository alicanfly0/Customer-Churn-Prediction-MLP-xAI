import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

def load_and_preprocess_data(filepath='WA_Fn-UseC_-Telco-Customer-Churn.csv'):
    """
    Loads the Telco dataset, cleans it, and returns SMOTE-balanced train and test sets.
    Matches the preprocessing pipeline from the baseline paper.
    """
    print("Loading dataset...")
    df = pd.read_csv(filepath)

    # 1. Basic Cleaning
    # TotalCharges is stored as an object/string. Force it to numeric and drop the 11 missing rows.
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(subset=['TotalCharges'], inplace=True)
    
    # Drop customerID and gender (as per the paper's feature selection step)
    df.drop(['customerID', 'gender'], axis=1, inplace=True)

    # 2. Separate Features and Target
    X = df.drop('Churn', axis=1)
    y = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

    # 3. Categorical Encoding
    # Identify categorical vs numerical columns
    categorical_cols = X.select_dtypes(include=['object']).columns
    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']

    # One-hot encode the categorical columns
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    # 4. Train/Test Split (80/20 split as defined in the baseline paper)
    print("Splitting data (80% Train, 20% Test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # 5. Feature Scaling (Z-score normalization)
    # Fit the scaler ONLY on the training data to prevent data leakage
    scaler = StandardScaler()
    
    # Scale numerical columns
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    X_train_scaled[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test_scaled[numerical_cols] = scaler.transform(X_test[numerical_cols])

    # 6. Handle Class Imbalance with SMOTE
    # CRITICAL: Apply SMOTE ONLY to the training data!
    print(f"Before SMOTE - Churners in train: {sum(y_train==1)}, Non-Churners: {sum(y_train==0)}")
    
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
    
    print(f"After SMOTE  - Churners in train: {sum(y_train_resampled==1)}, Non-Churners: {sum(y_train_resampled==0)}")
    
    # Convert all data to float32 for PyTorch compatibility
    X_train_resampled = X_train_resampled.astype(np.float32)
    X_test_scaled = X_test_scaled.astype(np.float32)
    y_train_resampled = y_train_resampled.astype(np.float32)
    y_test = y_test.astype(np.float32)

    print("Data preprocessing complete!")
    
    return X_train_resampled, X_test_scaled, y_train_resampled, y_test

# Quick test block to run this file standalone
if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    print(f"\nTraining features shape: {X_train.shape}")
    print(f"Testing features shape: {X_test.shape}")