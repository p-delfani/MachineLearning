"""
Data Preprocessing Module for Intrusion Detection Systems (IDS)

This file handles:
- Loading raw network traffic data
- Cleaning missing values
- Feature scaling
- Train-test split
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_dataset(path: str) -> pd.DataFrame:
    """Load dataset from CSV file."""
    return pd.read_csv(path)

def preprocess_data(df: pd.DataFrame, target_column: str):
    """
    Preprocess the dataset:
    - Separate features and labels
    - Scale numerical features
    - Split into train and test sets
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test
