"""
Quick Feature Importance Analysis while optimization runs
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import config
import joblib
import os


def quick_feature_analysis():
    """Quick analysis of current model performance and feature importance"""

    print("="*60)
    print("QUICK FEATURE IMPORTANCE ANALYSIS")
    print("="*60)

    # Load data
    print("Loading data...")
    data = pd.read_csv(config.TRAIN_DATA_PATH)

    # Prepare features
    X = data.drop(['Cover_Type', 'Id'], axis=1)
    y = data['Cover_Type']

    print(f"Data shape: {X.shape}")
    print(f"Classes: {sorted(y.unique())}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features
    print("\nScaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Quick Random Forest for feature importance
    print("Training Random Forest for feature importance...")
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )

    rf.fit(X_train_scaled, y_train)

    # Get accuracy
    train_accuracy = rf.score(X_train_scaled, y_train)
    test_accuracy = rf.score(X_test_scaled, y_test)

    print(f"Random Forest Results:")
    print(
        f"  Train Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
    print(f"  Test Accuracy:  {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)

    print(f"\nTop 15 Most Important Features:")
    print("-" * 50)
    for i, (_, row) in enumerate(feature_importance.head(15).iterrows()):
        print(f"{i+1:2d}. {row['feature']:35s} {row['importance']:.4f}")

    # Check if we have a saved model to compare
    model_path = os.path.join(config.MODELS_DIR, 'best_model_lightgbm.pkl')
    if os.path.exists(model_path):
        print(f"\nLoading saved LightGBM model...")
        try:
            lgb_model = joblib.load(model_path)
            lgb_accuracy = lgb_model.score(X_test_scaled, y_test)
            print(
                f"LightGBM Test Accuracy: {lgb_accuracy:.4f} ({lgb_accuracy*100:.2f}%)")

            # Compare
            improvement = test_accuracy - lgb_accuracy
            print(
                f"Difference (RF - LGB): {improvement:.4f} ({improvement*100:.2f}%)")

        except Exception as e:
            print(f"Error loading LightGBM model: {e}")

    # Analyze class predictions
    y_pred = rf.predict(X_test_scaled)

    print(f"\nClass-wise Accuracy:")
    print("-" * 30)
    for class_val in sorted(y_test.unique()):
        mask = y_test == class_val
        class_accuracy = (y_pred[mask] == y_test[mask]).mean()
        class_count = mask.sum()
        print(
            f"Class {class_val}: {class_accuracy:.4f} ({class_accuracy*100:.2f}%) - {class_count} samples")

    # Target analysis
    print(f"\nTarget: 99% accuracy")
    print(f"Current: {test_accuracy*100:.2f}%")
    print(f"Gap: {99 - test_accuracy*100:.2f}%")

    return feature_importance, test_accuracy


if __name__ == "__main__":
    feature_importance, accuracy = quick_feature_analysis()
