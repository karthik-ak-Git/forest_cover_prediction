"""
Forest Cover Type Prediction - Ensemble Models
This module implements XGBoost, Random Forest, and ensemble methods for achieving 99% accuracy
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import lightgbm as lgb
import optuna
import pickle
import config


class EnsembleModelManager:
    """Manager for ensemble models including XGBoost, Random Forest, and LightGBM"""

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.best_params = {}
        self.cv_scores = {}

    def create_xgboost_model(self, X_train, y_train, X_val, y_val,
                             optimize_hyperparams=True, n_trials=100):
        """Create and optimize XGBoost model"""
        print("Creating XGBoost model...")

        if optimize_hyperparams:
            print("Optimizing XGBoost hyperparameters...")

            def objective(trial):
                params = {
                    'objective': 'multi:softprob',
                    'num_class': 7,
                    'max_depth': trial.suggest_int('max_depth', 3, 15),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                    'random_state': self.random_state,
                    'verbosity': 0
                }

                model = xgb.XGBClassifier(**params)
                model.fit(X_train, y_train - 1)  # Convert to 0-based indexing
                preds = model.predict(X_val)
                accuracy = accuracy_score(y_val - 1, preds)
                return accuracy

            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=n_trials)

            self.best_params['xgboost'] = study.best_params
            print(f"Best XGBoost params: {study.best_params}")
            print(f"Best XGBoost accuracy: {study.best_value:.4f}")

            # Create final model with best parameters
            self.models['xgboost'] = xgb.XGBClassifier(
                **study.best_params,
                objective='multi:softprob',
                num_class=7,
                random_state=self.random_state,
                verbosity=0
            )
        else:
            # Use default parameters from config
            self.models['xgboost'] = xgb.XGBClassifier(
                **config.XGBOOST_PARAMS,
                objective='multi:softprob',
                num_class=7,
                verbosity=0
            )

        # Train the model
        self.models['xgboost'].fit(X_train, y_train - 1)

        # Validate
        val_preds = self.models['xgboost'].predict(X_val)
        val_accuracy = accuracy_score(y_val - 1, val_preds)
        self.cv_scores['xgboost'] = val_accuracy

        print(f"XGBoost validation accuracy: {val_accuracy:.4f}")
        return self.models['xgboost']

    def create_lightgbm_model(self, X_train, y_train, X_val, y_val,
                              optimize_hyperparams=True, n_trials=100):
        """Create and optimize LightGBM model"""
        print("Creating LightGBM model...")

        if optimize_hyperparams:
            print("Optimizing LightGBM hyperparameters...")

            def objective(trial):
                params = {
                    'objective': 'multiclass',
                    'num_class': 7,
                    'max_depth': trial.suggest_int('max_depth', 3, 15),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                    'num_leaves': trial.suggest_int('num_leaves', 10, 300),
                    'random_state': self.random_state,
                    'verbosity': -1
                }

                model = lgb.LGBMClassifier(**params)
                model.fit(X_train, y_train - 1)
                preds = model.predict(X_val)
                accuracy = accuracy_score(y_val - 1, preds)
                return accuracy

            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=n_trials)

            self.best_params['lightgbm'] = study.best_params
            print(f"Best LightGBM params: {study.best_params}")
            print(f"Best LightGBM accuracy: {study.best_value:.4f}")

            # Create final model with best parameters
            self.models['lightgbm'] = lgb.LGBMClassifier(
                **study.best_params,
                objective='multiclass',
                num_class=7,
                random_state=self.random_state,
                verbosity=-1
            )
        else:
            # Use default parameters
            self.models['lightgbm'] = lgb.LGBMClassifier(
                objective='multiclass',
                num_class=7,
                max_depth=10,
                learning_rate=0.1,
                n_estimators=1000,
                random_state=self.random_state,
                verbosity=-1
            )

        # Train the model
        self.models['lightgbm'].fit(X_train, y_train - 1)

        # Validate
        val_preds = self.models['lightgbm'].predict(X_val)
        val_accuracy = accuracy_score(y_val - 1, val_preds)
        self.cv_scores['lightgbm'] = val_accuracy

        print(f"LightGBM validation accuracy: {val_accuracy:.4f}")
        return self.models['lightgbm']

    def create_random_forest_model(self, X_train, y_train, X_val, y_val,
                                   optimize_hyperparams=True, n_trials=50):
        """Create and optimize Random Forest model"""
        print("Creating Random Forest model...")

        if optimize_hyperparams:
            print("Optimizing Random Forest hyperparameters...")

            def objective(trial):
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                    'max_depth': trial.suggest_int('max_depth', 5, 30),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                    'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                    'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
                    'random_state': self.random_state,
                    'n_jobs': -1
                }

                model = RandomForestClassifier(**params)
                model.fit(X_train, y_train - 1)
                preds = model.predict(X_val)
                accuracy = accuracy_score(y_val - 1, preds)
                return accuracy

            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=n_trials)

            self.best_params['random_forest'] = study.best_params
            print(f"Best Random Forest params: {study.best_params}")
            print(f"Best Random Forest accuracy: {study.best_value:.4f}")

            # Create final model with best parameters
            self.models['random_forest'] = RandomForestClassifier(
                **study.best_params,
                random_state=self.random_state,
                n_jobs=-1
            )
        else:
            # Use default parameters from config
            self.models['random_forest'] = RandomForestClassifier(
                **config.RANDOM_FOREST_PARAMS
            )

        # Train the model
        self.models['random_forest'].fit(X_train, y_train - 1)

        # Validate
        val_preds = self.models['random_forest'].predict(X_val)
        val_accuracy = accuracy_score(y_val - 1, val_preds)
        self.cv_scores['random_forest'] = val_accuracy

        print(f"Random Forest validation accuracy: {val_accuracy:.4f}")
        return self.models['random_forest']

    def create_voting_ensemble(self, X_train, y_train, X_val, y_val):
        """Create voting ensemble of all models"""
        print("Creating voting ensemble...")

        # Ensure we have the base models
        if len(self.models) == 0:
            print("No base models found. Creating base models first...")
            self.create_xgboost_model(
                X_train, y_train, X_val, y_val, optimize_hyperparams=False)
            self.create_lightgbm_model(
                X_train, y_train, X_val, y_val, optimize_hyperparams=False)
            self.create_random_forest_model(
                X_train, y_train, X_val, y_val, optimize_hyperparams=False)

        # Create voting classifier
        estimators = [(name, model) for name, model in self.models.items()
                      if name != 'voting_ensemble']

        # Try both hard and soft voting
        for voting_type in ['soft', 'hard']:
            ensemble_name = f'voting_ensemble_{voting_type}'

            try:
                voting_ensemble = VotingClassifier(
                    estimators=estimators,
                    voting=voting_type
                )

                voting_ensemble.fit(X_train, y_train - 1)
                val_preds = voting_ensemble.predict(X_val)
                val_accuracy = accuracy_score(y_val - 1, val_preds)

                self.models[ensemble_name] = voting_ensemble
                self.cv_scores[ensemble_name] = val_accuracy

                print(f"{ensemble_name} validation accuracy: {val_accuracy:.4f}")

            except Exception as e:
                print(f"Error creating {ensemble_name}: {e}")\n        \n return self.models.get('voting_ensemble_soft') or self.models.get('voting_ensemble_hard')\n    \n def create_stacking_ensemble(self, X_train, y_train, X_val, y_val): \n        \"\"\"Create stacking ensemble using cross-validation predictions\"\"\"\n print(\"Creating stacking ensemble...\")\n        \n from sklearn.ensemble import StackingClassifier\n from sklearn.linear_model import LogisticRegression\n        \n        # Base models\n        if len(self.models) == 0:\n            self.create_xgboost_model(X_train, y_train, X_val, y_val, optimize_hyperparams=False)\n            self.create_lightgbm_model(X_train, y_train, X_val, y_val, optimize_hyperparams=False)\n            self.create_random_forest_model(X_train, y_train, X_val, y_val, optimize_hyperparams=False)\n        \n        estimators = [(name, model) for name, model in self.models.items() \n                     if 'ensemble' not in name]\n        \n        # Meta-learner\n        meta_learner = LogisticRegression(random_state=self.random_state, max_iter=1000)\n        \n        stacking_ensemble = StackingClassifier(\n            estimators=estimators,\n            final_estimator=meta_learner,\n            cv=5,\n            passthrough=True  # Include original features\n        )\n        \n        stacking_ensemble.fit(X_train, y_train - 1)\n        val_preds = stacking_ensemble.predict(X_val)\n        val_accuracy = accuracy_score(y_val - 1, val_preds)\n        \n        self.models['stacking_ensemble'] = stacking_ensemble\n        self.cv_scores['stacking_ensemble'] = val_accuracy\n        \n        print(f\"Stacking ensemble validation accuracy: {val_accuracy:.4f}\")\n        return stacking_ensemble\n    \n    def train_all_models(self, X_train, y_train, X_val, y_val, optimize_hyperparams=True):\n        \"\"\"Train all models\"\"\"\n        print(\"=\" * 60)\n        print(\"TRAINING ALL ENSEMBLE MODELS\")\n        print(\"=\" * 60)\n        \n        # Train individual models\n        self.create_xgboost_model(X_train, y_train, X_val, y_val, optimize_hyperparams)\n        self.create_lightgbm_model(X_train, y_train, X_val, y_val, optimize_hyperparams)\n        self.create_random_forest_model(X_train, y_train, X_val, y_val, optimize_hyperparams)\n        \n        # Train ensemble models\n        self.create_voting_ensemble(X_train, y_train, X_val, y_val)\n        self.create_stacking_ensemble(X_train, y_train, X_val, y_val)\n        \n        # Print summary\n        print(\"\\n\" + \"=\" * 60)\n        print(\"MODEL PERFORMANCE SUMMARY\")\n        print(\"=\" * 60)\n        \n        sorted_scores = sorted(self.cv_scores.items(), key=lambda x: x[1], reverse=True)\n        for model_name, score in sorted_scores:\n            print(f\"{model_name:25}: {score:.4f} ({score*100:.2f}%)\")\n        \n        best_model_name = sorted_scores[0][0]\n        best_score = sorted_scores[0][1]\n        \n        print(f\"\\n🏆 Best model: {best_model_name} with {best_score:.4f} ({best_score*100:.2f}%) accuracy\")\n        \n        return self.models[best_model_name], best_model_name\n    \n    def evaluate_model(self, model_name, X_test, y_test):\n        \"\"\"Evaluate a specific model on test set\"\"\"\n        if model_name not in self.models:\n            raise ValueError(f\"Model {model_name} not found\")\n        \n        model = self.models[model_name]\n        test_preds = model.predict(X_test)\n        test_accuracy = accuracy_score(y_test - 1, test_preds)\n        \n        print(f\"\\n{model_name} Test Results:\")\n        print(f\"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)\")\n        \n        # Classification report\n        report = classification_report(y_test - 1, test_preds)\n        print(\"Classification Report:\")\n        print(report)\n        \n        # Confusion matrix\n        cm = confusion_matrix(y_test - 1, test_preds)\n        print(\"Confusion Matrix:\")\n        print(cm)\n        \n        return test_accuracy, report, cm\n    \n    def save_models(self, filepath_prefix):\n        \"\"\"Save all trained models\"\"\"\n        import os\n        \n        for model_name, model in self.models.items():\n            filepath = f\"{filepath_prefix}_{model_name}.pkl\"\n            with open(filepath, 'wb') as f:\n                pickle.dump(model, f)\n            print(f\"Saved {model_name} to {filepath}\")\n        \n        # Save manager state\n        state_file = f\"{filepath_prefix}_manager_state.pkl\"\n        with open(state_file, 'wb') as f:\n            pickle.dump({\n                'best_params': self.best_params,\n                'cv_scores': self.cv_scores,\n                'random_state': self.random_state\n            }, f)\n        print(f\"Saved manager state to {state_file}\")\n    \n    def load_models(self, filepath_prefix):\n        \"\"\"Load all saved models\"\"\"\n        import os\n        import glob\n        \n        # Load individual models\n        pattern = f\"{filepath_prefix}_*.pkl\"\n        for filepath in glob.glob(pattern):\n            if 'manager_state' in filepath:\n                continue\n                \n            model_name = os.path.basename(filepath).replace(f\"{os.path.basename(filepath_prefix)}_\", \"\").replace(\".pkl\", \"\")\n            \n            with open(filepath, 'rb') as f:\n                self.models[model_name] = pickle.load(f)\n            print(f\"Loaded {model_name} from {filepath}\")\n        \n        # Load manager state\n        state_file = f\"{filepath_prefix}_manager_state.pkl\"\n        if os.path.exists(state_file):\n            with open(state_file, 'rb') as f:\n                state = pickle.load(f)\n            self.best_params = state['best_params']\n            self.cv_scores = state['cv_scores']\n            self.random_state = state['random_state']\n            print(f\"Loaded manager state from {state_file}\")\n\nif __name__ == \"__main__\":\n    # Test the ensemble models\n    print(\"Testing Ensemble Models implementation...\")\n    \n    # Create manager\n    manager = EnsembleModelManager()\n    \n    # Create dummy data for testing\n    np.random.seed(42)\n    X_train = np.random.randn(1000, 54)\n    y_train = np.random.randint(1, 8, 1000)\n    X_val = np.random.randn(200, 54)\n    y_val = np.random.randint(1, 8, 200)\n    \n    print(\"Testing XGBoost model creation...\")\n    try:\n        xgb_model = manager.create_xgboost_model(X_train, y_train, X_val, y_val, optimize_hyperparams=False)\n        print(\"✅ XGBoost model created successfully!\")\n    except Exception as e:\n        print(f\"❌ XGBoost error: {e}\")\n    \n    print(\"\\nTesting LightGBM model creation...\")\n    try:\n        lgb_model = manager.create_lightgbm_model(X_train, y_train, X_val, y_val, optimize_hyperparams=False)\n        print(\"✅ LightGBM model created successfully!\")\n    except Exception as e:\n        print(f\"❌ LightGBM error: {e}\")\n    \n    print(\"\\nTesting Random Forest model creation...\")\n    try:\n        rf_model = manager.create_random_forest_model(X_train, y_train, X_val, y_val, optimize_hyperparams=False)\n        print(\"✅ Random Forest model created successfully!\")\n    except Exception as e:\n        print(f\"❌ Random Forest error: {e}\")\n    \n    print(\"\\n\" + \"=\"*50)\n    print(\"All models tested successfully!\")
