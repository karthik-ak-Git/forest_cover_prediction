"""
Simple test script for data preprocessing
"""

import pandas as pd
import numpy as np
import config
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def simple_preprocessing_test():
    print("Testing data preprocessing...")

    # Load data
    print(f"Loading data from {config.TRAIN_DATA_PATH}")
    df = pd.read_csv(config.TRAIN_DATA_PATH)
    print(f"Dataset shape: {df.shape}")

    # Basic preprocessing
    if 'Id' in df.columns:
        df = df.drop('Id', axis=1)

    X = df.drop('Cover_Type', axis=1)
    y = df['Cover_Type']

    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Target distribution:")
    print(y.value_counts().sort_index())

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"Scaled train features shape: {X_train_scaled.shape}")
    print(f"Scaled test features shape: {X_test_scaled.shape}")

    print("✅ Basic preprocessing completed successfully!")
    return True


if __name__ == "__main__":
    simple_preprocessing_test()
