"""
Advanced Model Optimization for 99% Accuracy
This script implements sophisticated techniques to achieve the target 99% accuracy:
- Advanced feature engineering
- Hyperparameter optimization with Optuna
- Advanced ensemble methods
- Data augmentation techniques
- Cross-validation optimization
"""

import config
import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')


class AdvancedModelOptimizer:
    """Advanced model optimization for achieving 99% accuracy"""

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.best_models = {}
        self.best_scores = {}
        self.feature_selector = None
        self.scaler = StandardScaler()

    def advanced_feature_engineering(self, X, y=None):
        """Advanced feature engineering techniques"""
        print("Applying advanced feature engineering...")

        # Convert to DataFrame for easier manipulation
        if isinstance(X, np.ndarray):
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]
            df = pd.DataFrame(X, columns=feature_names)
        else:
            df = X.copy()

        original_features = df.shape[1]

        # 1. Polynomial features for important interactions
        print("Creating polynomial features...")
        # Select top features first to avoid explosion
        if y is not None:
            selector = SelectKBest(f_classif, k=min(20, df.shape[1]))
            top_features = selector.fit_transform(df, y)
            selected_indices = selector.get_support(indices=True)

            # Create polynomial features for top features only
            poly = PolynomialFeatures(
                degree=2, interaction_only=True, include_bias=False)
            poly_features = poly.fit_transform(top_features)

            # Add polynomial features back to original dataframe
            poly_feature_names = [f'poly_{i}' for i in range(
                poly_features.shape[1] - top_features.shape[1])]
            poly_df = pd.DataFrame(
                poly_features[:, top_features.shape[1]:],
                columns=poly_feature_names,
                index=df.index
            )
            df = pd.concat([df, poly_df], axis=1)

        # 2. Statistical features
        print("Creating statistical features...")
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) > 1:
            # Row-wise statistics
            df['row_mean'] = df[numeric_cols].mean(axis=1)
            df['row_std'] = df[numeric_cols].std(axis=1)
            df['row_max'] = df[numeric_cols].max(axis=1)
            df['row_min'] = df[numeric_cols].min(axis=1)
            df['row_range'] = df['row_max'] - df['row_min']
            df['row_skew'] = df[numeric_cols].skew(axis=1)

        # 3. Binning features for non-linear relationships
        print("Creating binned features...")
        # Limit to avoid too many features
        for i, col in enumerate(numeric_cols[:10]):
            try:
                df[f'{col}_binned'] = pd.cut(df[col], bins=5, labels=False)
            except:
                continue

        # 4. Distance-based features
        print("Creating distance features...")
        if len(numeric_cols) >= 2:
            # Euclidean distance from origin
            df['euclidean_dist'] = np.sqrt((df[numeric_cols]**2).sum(axis=1))

            # Manhattan distance
            df['manhattan_dist'] = df[numeric_cols].abs().sum(axis=1)

        print(
            f"Feature engineering completed: {original_features} -> {df.shape[1]} features")
        return df.values

    def optimize_xgboost(self, X, y, cv_folds=5, n_trials=100):
        """Optimize XGBoost with Optuna"""
        print("Optimizing XGBoost hyperparameters...")

        def objective(trial):
            params = {
                'objective': 'multi:softprob',
                'num_class': 7,
                'max_depth': trial.suggest_int('max_depth', 6, 20),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 500, 3000),
                'subsample': trial.suggest_float('subsample', 0.7, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
                'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.7, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 0, 5),
                'random_state': self.random_state,
                'verbosity': 0,
                'tree_method': 'gpu_hist' if torch.cuda.is_available() else 'hist'
            }

            model = xgb.XGBClassifier(**params)

            # Cross-validation
            cv_scores = cross_val_score(
                model, X, y-1, cv=cv_folds,
                scoring='accuracy', n_jobs=-1
            )

            return cv_scores.mean()

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        best_params = study.best_params
        best_params.update({
            'objective': 'multi:softprob',
            'num_class': 7,
            'random_state': self.random_state,
            'verbosity': 0,
            'tree_method': 'gpu_hist' if torch.cuda.is_available() else 'hist'
        })

        best_model = xgb.XGBClassifier(**best_params)
        best_model.fit(X, y-1)

        self.best_models['xgboost'] = best_model
        self.best_scores['xgboost'] = study.best_value

        print(f"Best XGBoost CV score: {study.best_value:.4f}")
        return best_model, study.best_value

    def optimize_lightgbm(self, X, y, cv_folds=5, n_trials=100):
        """Optimize LightGBM with Optuna"""
        print("Optimizing LightGBM hyperparameters...")

        def objective(trial):
            params = {
                'objective': 'multiclass',
                'num_class': 7,
                'max_depth': trial.suggest_int('max_depth', 6, 20),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 500, 3000),
                'subsample': trial.suggest_float('subsample', 0.7, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'num_leaves': trial.suggest_int('num_leaves', 20, 500),
                'random_state': self.random_state,
                'verbosity': -1,
                'device': 'gpu' if torch.cuda.is_available() else 'cpu'
            }

            model = lgb.LGBMClassifier(**params)

            # Cross-validation
            cv_scores = cross_val_score(
                model, X, y-1, cv=cv_folds,
                scoring='accuracy', n_jobs=-1
            )

            return cv_scores.mean()

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        best_params = study.best_params
        best_params.update({
            'objective': 'multiclass',
            'num_class': 7,
            'random_state': self.random_state,
            'verbosity': -1,
            'device': 'gpu' if torch.cuda.is_available() else 'cpu'
        })

        best_model = lgb.LGBMClassifier(**best_params)
        best_model.fit(X, y-1)

        self.best_models['lightgbm'] = best_model
        self.best_scores['lightgbm'] = study.best_value

        print(f"Best LightGBM CV score: {study.best_value:.4f}")
        return best_model, study.best_value

    def optimize_random_forest(self, X, y, cv_folds=5, n_trials=50):
        """Optimize Random Forest with Optuna"""
        print("Optimizing Random Forest hyperparameters...")

        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 500, 2000),
                'max_depth': trial.suggest_int('max_depth', 10, 50),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
                'max_samples': trial.suggest_float('max_samples', 0.7, 1.0) if trial.params['bootstrap'] else None,
                'random_state': self.random_state,
                'n_jobs': -1
            }

            if not params['bootstrap']:
                params.pop('max_samples')

            model = RandomForestClassifier(**params)

            # Cross-validation
            cv_scores = cross_val_score(
                model, X, y-1, cv=cv_folds,
                scoring='accuracy', n_jobs=-1
            )

            return cv_scores.mean()

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        best_params = study.best_params
        best_params.update({
            'random_state': self.random_state,
            'n_jobs': -1
        })

        best_model = RandomForestClassifier(**best_params)
        best_model.fit(X, y-1)

        self.best_models['random_forest'] = best_model
        self.best_scores['random_forest'] = study.best_value

        print(f"Best Random Forest CV score: {study.best_value:.4f}")
        return best_model, study.best_value

    def create_advanced_ensemble(self, X, y, cv_folds=5):
        """Create advanced ensemble with multiple levels"""
        print("Creating advanced ensemble model...")

        if len(self.best_models) < 2:
            raise ValueError("Need at least 2 base models for ensemble")

        # Level 1: Base models (already trained)
        base_estimators = [(name, model)
                           for name, model in self.best_models.items()]

        # Level 2: Meta-learner optimization
        def optimize_meta_learner(trial):
            meta_params = {
                'C': trial.suggest_float('C', 0.1, 10, log=True),
                'penalty': trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet']),
                'solver': 'saga',
                'max_iter': 2000,
                'random_state': self.random_state
            }

            if meta_params['penalty'] == 'elasticnet':
                meta_params['l1_ratio'] = trial.suggest_float('l1_ratio', 0, 1)

            meta_learner = LogisticRegression(**meta_params)

            # Stacking ensemble
            stacking_clf = StackingClassifier(
                estimators=base_estimators,
                final_estimator=meta_learner,
                cv=cv_folds,
                passthrough=True,
                n_jobs=-1
            )

            cv_scores = cross_val_score(
                stacking_clf, X, y-1, cv=cv_folds,
                scoring='accuracy', n_jobs=-1
            )

            return cv_scores.mean()

        study = optuna.create_study(direction='maximize')
        study.optimize(optimize_meta_learner, n_trials=50,
                       show_progress_bar=True)

        # Create final ensemble with best meta-learner
        best_meta_params = study.best_params
        best_meta_params.update({
            'solver': 'saga',
            'max_iter': 2000,
            'random_state': self.random_state
        })

        meta_learner = LogisticRegression(**best_meta_params)

        ensemble_model = StackingClassifier(
            estimators=base_estimators,
            final_estimator=meta_learner,
            cv=cv_folds,
            passthrough=True,
            n_jobs=-1
        )

        ensemble_model.fit(X, y-1)

        # Evaluate ensemble
        cv_scores = cross_val_score(
            ensemble_model, X, y-1, cv=cv_folds,
            scoring='accuracy', n_jobs=-1
        )

        ensemble_score = cv_scores.mean()

        self.best_models['ensemble'] = ensemble_model
        self.best_scores['ensemble'] = ensemble_score

        print(f"Advanced ensemble CV score: {ensemble_score:.4f}")
        return ensemble_model, ensemble_score

    def run_complete_optimization(self, X, y, n_trials_per_model=100):
        """Run complete optimization pipeline"""
        print("="*70)
        print("ADVANCED MODEL OPTIMIZATION FOR 99% ACCURACY")
        print("="*70)

        # Feature engineering
        X_engineered = self.advanced_feature_engineering(X, y)

        # Feature scaling
        X_scaled = self.scaler.fit_transform(X_engineered)

        # Feature selection on engineered features
        print("Performing feature selection...")
        selector = SelectKBest(f_classif, k=min(100, X_scaled.shape[1]))
        X_selected = selector.fit_transform(X_scaled, y)
        self.feature_selector = selector

        print(
            f"Selected {X_selected.shape[1]} features from {X_scaled.shape[1]}")

        # Optimize individual models
        all_scores = {}

        try:
            _, score = self.optimize_xgboost(
                X_selected, y, n_trials=n_trials_per_model)
            all_scores['xgboost'] = score
        except Exception as e:
            print(f"XGBoost optimization failed: {e}")

        try:
            _, score = self.optimize_lightgbm(
                X_selected, y, n_trials=n_trials_per_model)
            all_scores['lightgbm'] = score
        except Exception as e:
            print(f"LightGBM optimization failed: {e}")

        try:
            _, score = self.optimize_random_forest(
                X_selected, y, n_trials=n_trials_per_model//2)
            all_scores['random_forest'] = score
        except Exception as e:
            print(f"Random Forest optimization failed: {e}")

        # Create ensemble if we have multiple models
        if len(self.best_models) >= 2:
            try:
                _, ensemble_score = self.create_advanced_ensemble(
                    X_selected, y)
                all_scores['ensemble'] = ensemble_score
            except Exception as e:
                print(f"Ensemble creation failed: {e}")

        # Print results
        print("\\n" + "="*70)
        print("OPTIMIZATION RESULTS")
        print("="*70)

        sorted_scores = sorted(
            all_scores.items(), key=lambda x: x[1], reverse=True)
        for model_name, score in sorted_scores:
            percentage = score * 100
            print(f"{model_name:20}: {score:.4f} ({percentage:.2f}%)")

            if percentage >= 99.0:
                print(f"🎯 {model_name} ACHIEVED 99% TARGET! 🎯")

        best_model_name = sorted_scores[0][0] if sorted_scores else None
        best_score = sorted_scores[0][1] if sorted_scores else 0

        print(f"\\n🏆 Best model: {best_model_name}")
        print(f"🎯 Best score: {best_score:.4f} ({best_score*100:.2f}%)")

        if best_score >= 0.99:
            print("\\n🎉 SUCCESS: 99% ACCURACY TARGET ACHIEVED! 🎉")
        else:
            needed = (0.99 - best_score) * 100
            print(f"\\n📈 Need {needed:.2f}% more accuracy to reach 99% target")

        return self.best_models[best_model_name], best_score

    def save_best_model(self, filepath):
        """Save the best performing model"""
        import pickle

        if not self.best_models:
            print("No models to save")
            return

        best_model_name = max(self.best_scores, key=self.best_scores.get)
        best_model = self.best_models[best_model_name]

        model_data = {
            'model': best_model,
            'scaler': self.scaler,
            'feature_selector': self.feature_selector,
            'model_name': best_model_name,
            'score': self.best_scores[best_model_name],
            'all_scores': self.best_scores
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"Best model ({best_model_name}) saved to {filepath}")


def main():
    """Main optimization function"""
    print("Loading data...")

    # Load data
    df = pd.read_csv(config.TRAIN_DATA_PATH)

    if 'Id' in df.columns:
        df = df.drop('Id', axis=1)

    X = df.drop('Cover_Type', axis=1).values
    y = df['Cover_Type'].values

    print(f"Dataset shape: {X.shape}")
    print(f"Target distribution: {np.bincount(y)}")

    # Create optimizer
    optimizer = AdvancedModelOptimizer(random_state=42)

    # Run optimization
    best_model, best_score = optimizer.run_complete_optimization(
        X, y, n_trials_per_model=50)

    # Save results
    import os
    model_path = os.path.join(config.MODELS_DIR, 'optimized_model_99.pkl')
    optimizer.save_best_model(model_path)

    print(f"\\nOptimization completed!")
    print(f"Best model score: {best_score:.4f} ({best_score*100:.2f}%)")


if __name__ == "__main__":
    import torch  # Make sure torch is imported for CUDA checks
    main()
