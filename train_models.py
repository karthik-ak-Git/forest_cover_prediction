"""
Forest Cover Type Prediction - Main Training Script
This script orchestrates the complete training pipeline to achieve 99% accuracy
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import lightgbm as lgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import config

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from neural_networks import ForestCoverNet, ForestCoverNetAdvanced, NeuralNetworkTrainer
except ImportError:
    print("Note: Neural network modules not available, using basic models only")


class ForestCoverPredictor:
    """Complete prediction system for forest cover type"""

    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.best_model = None
        self.best_model_name = None
        self.best_accuracy = 0

    def load_and_preprocess_data(self, filepath):
        """Load and preprocess the data"""
        print("Loading and preprocessing data...")

        # Load data
        df = pd.read_csv(filepath)
        print(f"Dataset shape: {df.shape}")

        # Remove Id column if present
        if 'Id' in df.columns:
            df = df.drop('Id', axis=1)

        # Separate features and target
        X = df.drop('Cover_Type', axis=1).values
        y = df['Cover_Type'].values

        print(f"Features shape: {X.shape}")
        print(f"Target distribution: {np.bincount(y)}")

        # Split data
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.4, stratify=y, random_state=config.RANDOM_STATE
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=config.RANDOM_STATE
        )

        print(f"Train set: {X_train.shape[0]} samples")
        print(f"Validation set: {X_val.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)

        return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test

    def train_xgboost(self, X_train, y_train, X_val, y_val):
        """Train XGBoost model"""
        print("Training XGBoost model...")

        model = xgb.XGBClassifier(
            objective='multi:softprob',
            num_class=7,
            max_depth=10,
            learning_rate=0.1,
            n_estimators=1000,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=config.RANDOM_STATE,
            eval_metric='mlogloss'
        )

        # Train with early stopping
        model.fit(
            X_train, y_train - 1,  # Convert to 0-based indexing
            eval_set=[(X_val, y_val - 1)],
            early_stopping_rounds=50,
            verbose=False
        )

        # Validate
        val_preds = model.predict(X_val)
        val_accuracy = accuracy_score(y_val - 1, val_preds)

        self.models['xgboost'] = model
        print(
            f"XGBoost validation accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")

        if val_accuracy > self.best_accuracy:
            self.best_accuracy = val_accuracy
            self.best_model = model
            self.best_model_name = 'xgboost'

        return val_accuracy

    def train_lightgbm(self, X_train, y_train, X_val, y_val):
        """Train LightGBM model"""
        print("Training LightGBM model...")

        model = lgb.LGBMClassifier(
            objective='multiclass',
            num_class=7,
            max_depth=10,
            learning_rate=0.1,
            n_estimators=1000,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=config.RANDOM_STATE,
            verbosity=-1
        )

        # Train with early stopping
        model.fit(
            X_train, y_train - 1,
            eval_set=[(X_val, y_val - 1)],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )

        # Validate
        val_preds = model.predict(X_val)
        val_accuracy = accuracy_score(y_val - 1, val_preds)

        self.models['lightgbm'] = model
        print(
            f"LightGBM validation accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")

        if val_accuracy > self.best_accuracy:
            self.best_accuracy = val_accuracy
            self.best_model = model
            self.best_model_name = 'lightgbm'

        return val_accuracy

    def train_random_forest(self, X_train, y_train, X_val, y_val):
        """Train Random Forest model"""
        print("Training Random Forest model...")

        model = RandomForestClassifier(
            n_estimators=500,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=config.RANDOM_STATE,
            n_jobs=-1
        )

        model.fit(X_train, y_train - 1)

        # Validate
        val_preds = model.predict(X_val)
        val_accuracy = accuracy_score(y_val - 1, val_preds)

        self.models['random_forest'] = model
        print(
            f"Random Forest validation accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")

        if val_accuracy > self.best_accuracy:
            self.best_accuracy = val_accuracy
            self.best_model = model
            self.best_model_name = 'random_forest'

        return val_accuracy

    def train_neural_network(self, X_train, y_train, X_val, y_val):
        """Train Neural Network model"""
        try:
            print("Training Neural Network model...")

            # Create data loaders
            train_dataset = TensorDataset(
                torch.FloatTensor(X_train),
                torch.LongTensor(y_train - 1)
            )
            val_dataset = TensorDataset(
                torch.FloatTensor(X_val),
                torch.LongTensor(y_val - 1)
            )

            train_loader = DataLoader(
                train_dataset, batch_size=512, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)

            # Create model
            model = ForestCoverNet(
                input_dim=X_train.shape[1],
                hidden_layers=[512, 256, 128, 64],
                dropout_rate=0.3,
                num_classes=7
            )

            # Train
            trainer = NeuralNetworkTrainer(model, config.DEVICE)
            val_accuracy = trainer.train(
                train_loader, val_loader,
                epochs=100, learning_rate=0.001,
                patience=15, target_accuracy=99.0
            )

            self.models['neural_network'] = trainer.model
            print(
                f"Neural Network validation accuracy: {val_accuracy:.4f} ({val_accuracy:.2f}%)")

            if val_accuracy > self.best_accuracy:
                self.best_accuracy = val_accuracy
                self.best_model = trainer.model
                self.best_model_name = 'neural_network'

            return val_accuracy

        except Exception as e:
            print(f"Neural Network training failed: {e}")
            return 0

    def create_ensemble(self, X_train, y_train, X_val, y_val):
        """Create ensemble of best models"""
        print("Creating ensemble model...")

        if len(self.models) < 2:
            print("Need at least 2 models for ensemble")
            return 0

        # Get predictions from all models
        val_predictions = []

        for name, model in self.models.items():
            if name == 'neural_network':
                # Handle neural network predictions
                model.eval()
                with torch.no_grad():
                    val_tensor = torch.FloatTensor(X_val).to(config.DEVICE)
                    output = model(val_tensor)
                    preds = output.argmax(dim=1).cpu().numpy()
            else:
                preds = model.predict(X_val)

            val_predictions.append(preds)

        # Majority voting
        val_predictions = np.array(val_predictions)
        ensemble_preds = []

        for i in range(val_predictions.shape[1]):
            votes = val_predictions[:, i]
            # Get most common prediction
            unique, counts = np.unique(votes, return_counts=True)
            ensemble_pred = unique[np.argmax(counts)]
            ensemble_preds.append(ensemble_pred)

        ensemble_preds = np.array(ensemble_preds)
        ensemble_accuracy = accuracy_score(y_val - 1, ensemble_preds)

        print(
            f"Ensemble validation accuracy: {ensemble_accuracy:.4f} ({ensemble_accuracy*100:.2f}%)")

        if ensemble_accuracy > self.best_accuracy:
            self.best_accuracy = ensemble_accuracy
            self.best_model_name = 'ensemble'

        return ensemble_accuracy

    def evaluate_best_model(self, X_test, y_test):
        """Evaluate the best model on test set"""
        print(f"\\nEvaluating best model: {self.best_model_name}")
        print("=" * 50)

        if self.best_model_name == 'ensemble':
            # Ensemble prediction
            test_predictions = []

            for name, model in self.models.items():
                if name == 'neural_network':
                    model.eval()
                    with torch.no_grad():
                        test_tensor = torch.FloatTensor(
                            X_test).to(config.DEVICE)
                        output = model(test_tensor)
                        preds = output.argmax(dim=1).cpu().numpy()
                else:
                    preds = model.predict(X_test)

                test_predictions.append(preds)

            test_predictions = np.array(test_predictions)
            final_preds = []

            for i in range(test_predictions.shape[1]):
                votes = test_predictions[:, i]
                unique, counts = np.unique(votes, return_counts=True)
                final_pred = unique[np.argmax(counts)]
                final_preds.append(final_pred)

            final_preds = np.array(final_preds)

        elif self.best_model_name == 'neural_network':
            self.best_model.eval()
            with torch.no_grad():
                test_tensor = torch.FloatTensor(X_test).to(config.DEVICE)
                output = self.best_model(test_tensor)
                final_preds = output.argmax(dim=1).cpu().numpy()
        else:
            final_preds = self.best_model.predict(X_test)

        test_accuracy = accuracy_score(y_test - 1, final_preds)

        print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

        if test_accuracy >= 0.99:
            print("🎯 TARGET ACCURACY OF 99% ACHIEVED! 🎯")
        else:
            print(
                f"Target accuracy not reached. Need {0.99 - test_accuracy:.4f} more.")

        # Classification report
        print("\\nClassification Report:")
        print(classification_report(y_test - 1, final_preds))

        return test_accuracy

    def save_best_model(self, filepath):
        """Save the best model"""
        if self.best_model_name == 'ensemble':
            print("Ensemble model cannot be saved directly")
            return

        import pickle

        model_data = {
            'model': self.best_model,
            'scaler': self.scaler,
            'model_name': self.best_model_name,
            'accuracy': self.best_accuracy
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"Best model saved to {filepath}")

    def run_complete_training(self, data_path):
        """Run the complete training pipeline"""
        print("=" * 60)
        print("FOREST COVER TYPE PREDICTION - TRAINING PIPELINE")
        print("=" * 60)

        # Load and preprocess data
        X_train, X_val, X_test, y_train, y_val, y_test = self.load_and_preprocess_data(
            data_path)

        print("\\n" + "=" * 60)
        print("TRAINING MODELS")
        print("=" * 60)

        # Train all models
        accuracies = {}

        try:
            accuracies['xgboost'] = self.train_xgboost(
                X_train, y_train, X_val, y_val)
        except Exception as e:
            print(f"XGBoost training failed: {e}")

        try:
            accuracies['lightgbm'] = self.train_lightgbm(
                X_train, y_train, X_val, y_val)
        except Exception as e:
            print(f"LightGBM training failed: {e}")

        try:
            accuracies['random_forest'] = self.train_random_forest(
                X_train, y_train, X_val, y_val)
        except Exception as e:
            print(f"Random Forest training failed: {e}")

        try:
            accuracies['neural_network'] = self.train_neural_network(
                X_train, y_train, X_val, y_val)
        except Exception as e:
            print(f"Neural Network training failed: {e}")

        # Create ensemble
        try:
            accuracies['ensemble'] = self.create_ensemble(
                X_train, y_train, X_val, y_val)
        except Exception as e:
            print(f"Ensemble creation failed: {e}")

        # Print results summary
        print("\\n" + "=" * 60)
        print("VALIDATION RESULTS SUMMARY")
        print("=" * 60)

        sorted_accuracies = sorted(
            accuracies.items(), key=lambda x: x[1], reverse=True)
        for model_name, accuracy in sorted_accuracies:
            print(f"{model_name:20}: {accuracy:.4f} ({accuracy*100:.2f}%)")

        # Final evaluation
        test_accuracy = self.evaluate_best_model(X_test, y_test)

        # Save best model
        model_path = os.path.join(
            config.MODELS_DIR, f'best_model_{self.best_model_name}.pkl')
        self.save_best_model(model_path)

        print("\\n" + "=" * 60)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 60)

        return test_accuracy


def main():
    """Main function"""
    predictor = ForestCoverPredictor()
    final_accuracy = predictor.run_complete_training(config.TRAIN_DATA_PATH)

    print(
        f"\\nFinal Test Accuracy: {final_accuracy:.4f} ({final_accuracy*100:.2f}%)")

    if final_accuracy >= 0.99:
        print("🎉 SUCCESS: 99% accuracy target achieved!")
    else:
        print(
            f"🎯 Target: Need {(0.99 - final_accuracy)*100:.2f}% more accuracy")


if __name__ == "__main__":
    main()
