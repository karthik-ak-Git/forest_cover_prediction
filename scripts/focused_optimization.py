"""
Focused optimization approach for 99% accuracy based on analysis
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import VotingClassifier
import config
import joblib
import os


def focused_optimization():
    """Focused optimization based on feature analysis"""

    print("="*70)
    print("FOCUSED OPTIMIZATION FOR 99% ACCURACY")
    print("="*70)

    # Load data
    print("Loading data...")
    data = pd.read_csv(config.TRAIN_DATA_PATH)

    # Prepare features
    X = data.drop(['Cover_Type', 'Id'], axis=1)
    y = data['Cover_Type']

    print(f"Data shape: {X.shape}")

    # Create enhanced features based on analysis
    print("Creating enhanced features...")
    X_enhanced = X.copy()

    # Top features identified: Elevation, Distance features, Wilderness areas
    # Create interaction features
    X_enhanced['Elevation_Squared'] = X['Elevation'] ** 2
    X_enhanced['Elevation_Log'] = np.log1p(X['Elevation'])

    # Distance interactions
    X_enhanced['Total_Distance'] = (X['Horizontal_Distance_To_Roadways'] +
                                    X['Horizontal_Distance_To_Fire_Points'] +
                                    X['Horizontal_Distance_To_Hydrology'])

    X_enhanced['Distance_Ratio'] = (X['Horizontal_Distance_To_Roadways'] /
                                    (X['Horizontal_Distance_To_Fire_Points'] + 1))

    # Hillshade features
    X_enhanced['Hillshade_Mean'] = (
        X['Hillshade_9am'] + X['Hillshade_Noon'] + X['Hillshade_3pm']) / 3
    X_enhanced['Hillshade_Range'] = X[['Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm']].max(axis=1) - \
        X[['Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm']].min(axis=1)

    # Elevation zones (based on domain knowledge)
    X_enhanced['Elevation_Zone'] = pd.cut(
        X['Elevation'], bins=5, labels=[1, 2, 3, 4, 5])

    # Soil type clusters (group similar soil types)
    soil_cols = [col for col in X.columns if col.startswith('Soil_Type')]
    X_enhanced['Soil_Count'] = X[soil_cols].sum(axis=1)

    print(f"Enhanced features shape: {X_enhanced.shape}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_enhanced, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define models with optimized parameters
    models = {
        'lightgbm': lgb.LGBMClassifier(
            n_estimators=2000,
            max_depth=15,
            learning_rate=0.05,
            num_leaves=50,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=5,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        ),
        'xgboost': xgb.XGBClassifier(
            n_estimators=2000,
            max_depth=12,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            eval_metric='mlogloss'
        ),
        'rf_deep': RandomForestClassifier(
            n_estimators=1000,
            max_depth=25,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        ),
        'extra_trees': ExtraTreesClassifier(
            n_estimators=1000,
            max_depth=25,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        ),
        'gradient_boost': GradientBoostingClassifier(
            n_estimators=1000,
            max_depth=8,
            learning_rate=0.05,
            random_state=42
        )
    }

    print(f"Training {len(models)} models...")
    trained_models = {}

    # Cross-validation setup
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in models.items():
        print(f"\nTraining {name}...")

        # Cross-validation
        cv_scores = cross_val_score(
            model, X_train_scaled, y_train, cv=cv, scoring='accuracy', n_jobs=-1)

        # Fit model
        model.fit(X_train_scaled, y_train)

        # Test accuracy
        test_pred = model.predict(X_test_scaled)
        test_accuracy = accuracy_score(y_test, test_pred)

        trained_models[name] = model

        print(
            f"  CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        print(
            f"  Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

    # Create ensemble
    print(f"\nCreating ensemble...")
    voting_clf = VotingClassifier(
        estimators=[(name, model) for name, model in trained_models.items()],
        voting='soft'
    )

    # Fit ensemble
    voting_clf.fit(X_train_scaled, y_train)

    # Ensemble predictions
    ensemble_pred = voting_clf.predict(X_test_scaled)
    ensemble_accuracy = accuracy_score(y_test, ensemble_pred)

    print(
        f"Ensemble Test Accuracy: {ensemble_accuracy:.4f} ({ensemble_accuracy*100:.2f}%)")

    # Detailed classification report
    print(f"\nDetailed Classification Report:")
    print("="*50)
    print(classification_report(y_test, ensemble_pred))

    # Check which classes need improvement
    print(f"\nClass-wise Performance Analysis:")
    print("-"*40)
    for class_val in sorted(y_test.unique()):
        mask = y_test == class_val
        class_accuracy = (ensemble_pred[mask] == y_test[mask]).mean()
        class_count = mask.sum()
        print(
            f"Class {class_val}: {class_accuracy:.4f} ({class_accuracy*100:.2f}%) - {class_count} samples")

        if class_accuracy < 0.95:
            print(f"  → Class {class_val} needs improvement!")

    # Save best model
    if ensemble_accuracy > 0.85:  # Only save if better than current
        model_path = os.path.join(
            config.MODELS_DIR, 'enhanced_ensemble_model.pkl')
        joblib.dump({
            'model': voting_clf,
            'scaler': scaler,
            'accuracy': ensemble_accuracy,
            'feature_names': X_enhanced.columns.tolist()
        }, model_path)
        print(f"\nEnhanced model saved to: {model_path}")

    print(f"\nOptimization Results:")
    print(f"Current Best: {ensemble_accuracy*100:.2f}%")
    print(f"Target: 99.00%")
    print(f"Remaining Gap: {99 - ensemble_accuracy*100:.2f}%")

    if ensemble_accuracy >= 0.99:
        print("🎉 TARGET ACHIEVED! 99% accuracy reached!")
    else:
        print("🎯 Continue optimization needed for 99% target")

    return voting_clf, ensemble_accuracy


if __name__ == "__main__":
    model, accuracy = focused_optimization()
