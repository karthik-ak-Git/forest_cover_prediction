"""
Quick and effective optimization for 99% accuracy
Focus on the most important findings from analysis
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import lightgbm as lgb
import config
import joblib
import os


def quick_optimization():
    """Quick optimization based on key insights"""

    print("="*60)
    print("QUICK OPTIMIZATION FOR 99% ACCURACY")
    print("="*60)

    # Load data
    print("Loading data...")
    data = pd.read_csv(config.TRAIN_DATA_PATH)

    # Prepare features
    X = data.drop(['Cover_Type', 'Id'], axis=1)
    y = data['Cover_Type']

    print(f"Original data shape: {X.shape}")

    # Enhanced feature engineering based on analysis
    print("Creating key enhanced features...")
    X_enhanced = X.copy()

    # Key insights from analysis:
    # 1. Elevation is most important (0.2384 importance)
    # 2. Distance features are crucial
    # 3. Hillshade features have strong correlations
    # 4. Class 2 has lower accuracy (62.50%) - needs attention

    # Elevation features (most important)
    X_enhanced['Elevation_Squared'] = X['Elevation'] ** 2
    X_enhanced['Elevation_Zone'] = pd.cut(
        X['Elevation'], bins=4, labels=[1, 2, 3, 4])

    # Distance composite features
    X_enhanced['Total_Distance'] = (X['Horizontal_Distance_To_Roadways'] +
                                    X['Horizontal_Distance_To_Fire_Points'] +
                                    X['Horizontal_Distance_To_Hydrology'])

    # Hillshade interaction (they're correlated -0.78)
    X_enhanced['Hillshade_Diff'] = X['Hillshade_9am'] - X['Hillshade_3pm']
    X_enhanced['Hillshade_Mean'] = (
        X['Hillshade_9am'] + X['Hillshade_Noon'] + X['Hillshade_3pm']) / 3

    print(f"Enhanced data shape: {X_enhanced.shape}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        # Smaller test set for more training data
        X_enhanced, y, test_size=0.15, random_state=42, stratify=y
    )

    # Scale features
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Optimized LightGBM (performed best in analysis)
    print("Training optimized LightGBM...")
    lgb_model = lgb.LGBMClassifier(
        n_estimators=3000,      # More estimators
        max_depth=20,           # Deeper trees
        learning_rate=0.03,     # Lower learning rate
        num_leaves=100,         # More leaves
        feature_fraction=0.9,   # Use more features
        bagging_fraction=0.9,   # Use more data
        bagging_freq=5,
        min_child_samples=10,   # Reduce overfitting
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )

    # Fit model
    lgb_model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_test_scaled, y_test)],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
    )

    # Predictions
    y_pred = lgb_model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"LightGBM Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

    # Detailed analysis
    print(f"\nDetailed Classification Report:")
    print("="*50)
    print(classification_report(y_test, y_pred))

    # Class-wise analysis
    print(f"\nClass-wise Performance:")
    print("-"*30)
    for class_val in sorted(y_test.unique()):
        mask = y_test == class_val
        class_accuracy = (y_pred[mask] == y_test[mask]).mean()
        class_count = mask.sum()
        print(
            f"Class {class_val}: {class_accuracy:.4f} ({class_accuracy*100:.2f}%) - {class_count} samples")

    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X_enhanced.columns,
        'importance': lgb_model.feature_importances_
    }).sort_values('importance', ascending=False)

    print(f"\nTop 10 Enhanced Features:")
    print("-"*40)
    for i, (_, row) in enumerate(feature_importance.head(10).iterrows()):
        print(f"{i+1:2d}. {row['feature']:35s} {row['importance']:.4f}")

    # Save if good
    if accuracy > 0.84:  # Better than current best
        model_path = os.path.join(
            config.MODELS_DIR, 'quick_optimized_model.pkl')
        joblib.dump({
            'model': lgb_model,
            'scaler': scaler,
            'accuracy': accuracy,
            'feature_names': X_enhanced.columns.tolist()
        }, model_path)
        print(f"\nQuick optimized model saved to: {model_path}")

    print(f"\nOptimization Results:")
    print(f"Current Best: {accuracy*100:.2f}%")
    print(f"Target: 99.00%")
    print(f"Remaining Gap: {99 - accuracy*100:.2f}%")

    if accuracy >= 0.99:
        print("🎉 TARGET ACHIEVED! 99% accuracy reached!")
    elif accuracy >= 0.90:
        print("🔥 Excellent progress! Very close to target!")
    elif accuracy >= 0.85:
        print("🚀 Good improvement! Getting closer to target!")
    else:
        print("🎯 Continue optimization needed")

    return lgb_model, accuracy, scaler


if __name__ == "__main__":
    model, accuracy, scaler = quick_optimization()
