import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

def load_telco_data(filepath='WA_Fn-UseC_-Telco-Customer-Churn.csv', use_smote=True):
    """
    Loads and preprocesses the original IBM Telco dataset.
    Added 'use_smote' toggle for Assignment 3 Focal Loss experiments.
    """
    print(f"Loading Telco dataset... (SMOTE={use_smote})")
    df = pd.read_csv(filepath)

    # 1. Basic Cleaning
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(subset=['TotalCharges'], inplace=True)
    df.drop(['customerID', 'gender'], axis=1, inplace=True)

    # 2. Separate Features and Target
    X = df.drop('Churn', axis=1)
    y = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

    # 3. Categorical Encoding
    categorical_cols = X.select_dtypes(include=['object']).columns
    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    # 4. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # 5. Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test_scaled[numerical_cols] = scaler.transform(X_test[numerical_cols])

    # 6. Handle Class Imbalance
    if use_smote:
        smote = SMOTE(random_state=42)
        X_train_final, y_train_final = smote.fit_resample(X_train_scaled, y_train)
    else:
        X_train_final, y_train_final = X_train_scaled, y_train

    # Convert to PyTorch tensors
    return (X_train_final.astype(np.float32), X_test_scaled.astype(np.float32), 
            y_train_final.astype(np.float32), y_test.astype(np.float32))


def load_bank_data(filepath='Churn_Modelling.csv', use_smote=True):
    """
    Loads and preprocesses the Kaggle Bank Customer Churn dataset for cross-domain evaluation.
    """
    print(f"Loading Bank dataset... (SMOTE={use_smote})")
    df = pd.read_csv(filepath)

    # 1. Basic Cleaning
    # Drop columns that have no predictive power
    df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True)

    # 2. Separate Features and Target
    X = df.drop('Exited', axis=1)
    y = df['Exited']

    # 3. Categorical Encoding
    categorical_cols = ['Geography', 'Gender']
    numerical_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
    # Note: HasCrCard and IsActiveMember are already binary (0/1), so we don't scale or encode them.
    
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    # 4. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # 5. Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test_scaled[numerical_cols] = scaler.transform(X_test[numerical_cols])

    # 6. Handle Class Imbalance
    if use_smote:
        smote = SMOTE(random_state=42)
        X_train_final, y_train_final = smote.fit_resample(X_train_scaled, y_train)
    else:
        X_train_final, y_train_final = X_train_scaled, y_train

    # Convert to PyTorch tensors
    return (X_train_final.astype(np.float32), X_test_scaled.astype(np.float32), 
            y_train_final.astype(np.float32), y_test.astype(np.float32))

# Quick test block to ensure both load properly
if __name__ == "__main__":
    print("--- Testing Telco Pipeline ---")
    X_train_t, X_test_t, y_train_t, y_test_t = load_telco_data(use_smote=False)
    print(f"Telco Train Shape: {X_train_t.shape}, Test Shape: {X_test_t.shape}")

    print("\n--- Testing Bank Pipeline ---")
    X_train_b, X_test_b, y_train_b, y_test_b = load_bank_data(use_smote=False)
    print(f"Bank Train Shape: {X_train_b.shape}, Test Shape: {X_test_b.shape}")
