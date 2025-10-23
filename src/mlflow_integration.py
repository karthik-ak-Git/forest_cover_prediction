"""
MLflow integration for experiment tracking and model versioning
"""
import mlflow
import mlflow.sklearn
import mlflow.pytorch
from mlflow.tracking import MlflowClient
from mlflow.models import infer_signature
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from pathlib import Path
import logging
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class MLflowTracker:
    """MLflow experiment tracker for forest cover prediction"""

    def __init__(
        self,
        experiment_name: str = "forest_cover_prediction",
        tracking_uri: Optional[str] = None
    ):
        """
        Initialize MLflow tracker

        Args:
            experiment_name: Name of the experiment
            tracking_uri: MLflow tracking server URI (None for local)
        """
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        # Set or create experiment
        try:
            self.experiment = mlflow.set_experiment(experiment_name)
            self.experiment_id = self.experiment.experiment_id
        except Exception as e:
            logger.error(f"Error setting experiment: {e}")
            self.experiment_id = None

        self.client = MlflowClient()

    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> mlflow.ActiveRun:
        """
        Start a new MLflow run

        Args:
            run_name: Name for the run
            tags: Dictionary of tags

        Returns:
            Active MLflow run
        """
        return mlflow.start_run(run_name=run_name, tags=tags)

    def log_params(self, params: Dict[str, Any]):
        """Log parameters"""
        mlflow.log_params(params)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log metrics

        Args:
            metrics: Dictionary of metrics
            step: Optional step number for sequential logging
        """
        mlflow.log_metrics(metrics, step=step)

    def log_model(
        self,
        model: Any,
        artifact_path: str = "model",
        signature: Optional[Any] = None,
        input_example: Optional[pd.DataFrame] = None,
        registered_model_name: Optional[str] = None
    ):
        """
        Log model to MLflow

        Args:
            model: Trained model
            artifact_path: Path within run artifacts
            signature: Model signature
            input_example: Example input
            registered_model_name: Name for model registry
        """
        try:
            # Auto-detect model type and log appropriately
            model_type = type(model).__name__

            if 'sklearn' in str(type(model).__module__):
                mlflow.sklearn.log_model(
                    model,
                    artifact_path,
                    signature=signature,
                    input_example=input_example,
                    registered_model_name=registered_model_name
                )
            elif 'torch' in str(type(model).__module__):
                mlflow.pytorch.log_model(
                    model,
                    artifact_path,
                    signature=signature,
                    input_example=input_example,
                    registered_model_name=registered_model_name
                )
            else:
                # Generic pickling
                mlflow.pyfunc.log_model(
                    artifact_path,
                    python_model=model,
                    signature=signature,
                    input_example=input_example,
                    registered_model_name=registered_model_name
                )

            logger.info(f"Model logged successfully: {model_type}")
        except Exception as e:
            logger.error(f"Error logging model: {e}")

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """Log artifact file"""
        mlflow.log_artifact(local_path, artifact_path)

    def log_dict(self, dictionary: Dict, filename: str):
        """Log dictionary as JSON artifact"""
        mlflow.log_dict(dictionary, filename)

    def log_figure(self, figure, filename: str):
        """Log matplotlib figure"""
        mlflow.log_figure(figure, filename)

    def log_confusion_matrix(self, y_true, y_pred, labels: Optional[List] = None):
        """Log confusion matrix"""
        from sklearn.metrics import confusion_matrix
        import matplotlib.pyplot as plt
        import seaborn as sns

        cm = confusion_matrix(y_true, y_pred, labels=labels)

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels, yticklabels=labels)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')

        mlflow.log_figure(plt.gcf(), "confusion_matrix.png")
        plt.close()

    def log_feature_importance(
        self,
        feature_names: List[str],
        importances: np.ndarray,
        top_n: int = 20
    ):
        """
        Log feature importance plot

        Args:
            feature_names: List of feature names
            importances: Array of importance values
            top_n: Number of top features to show
        """
        import matplotlib.pyplot as plt

        # Get top features
        indices = np.argsort(importances)[-top_n:]
        top_features = [feature_names[i] for i in indices]
        top_importances = importances[indices]

        # Plot
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(top_features)), top_importances)
        plt.yticks(range(len(top_features)), top_features)
        plt.xlabel('Importance')
        plt.title(f'Top {top_n} Feature Importances')
        plt.tight_layout()

        mlflow.log_figure(plt.gcf(), "feature_importance.png")
        plt.close()

        # Also log as dict
        importance_dict = dict(zip(feature_names, importances.tolist()))
        mlflow.log_dict(importance_dict, "feature_importance.json")

    def log_training_results(
        self,
        model,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        params: Dict[str, Any],
        run_name: Optional[str] = None
    ):
        """
        Comprehensive logging of training results

        Args:
            model: Trained model
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            params: Model parameters
            run_name: Optional run name
        """
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score,
            f1_score, classification_report
        )

        with self.start_run(run_name=run_name) as run:
            # Log parameters
            self.log_params(params)

            # Training predictions
            y_train_pred = model.predict(X_train)
            train_metrics = {
                'train_accuracy': accuracy_score(y_train, y_train_pred),
                'train_precision': precision_score(y_train, y_train_pred, average='weighted'),
                'train_recall': recall_score(y_train, y_train_pred, average='weighted'),
                'train_f1': f1_score(y_train, y_train_pred, average='weighted'),
            }

            # Validation predictions
            y_val_pred = model.predict(X_val)
            val_metrics = {
                'val_accuracy': accuracy_score(y_val, y_val_pred),
                'val_precision': precision_score(y_val, y_val_pred, average='weighted'),
                'val_recall': recall_score(y_val, y_val_pred, average='weighted'),
                'val_f1': f1_score(y_val, y_val_pred, average='weighted'),
            }

            # Log metrics
            self.log_metrics({**train_metrics, **val_metrics})

            # Log confusion matrix
            self.log_confusion_matrix(
                y_val, y_val_pred, labels=list(range(1, 8)))

            # Log feature importance if available
            if hasattr(model, 'feature_importances_'):
                self.log_feature_importance(
                    X_train.columns.tolist(),
                    model.feature_importances_
                )

            # Log classification report
            report = classification_report(y_val, y_val_pred, output_dict=True)
            self.log_dict(report, "classification_report.json")

            # Log model with signature
            try:
                signature = infer_signature(X_train, y_train_pred)
                self.log_model(
                    model,
                    signature=signature,
                    input_example=X_train.head(1),
                    registered_model_name="forest_cover_classifier"
                )
            except Exception as e:
                logger.warning(f"Could not log model with signature: {e}")
                self.log_model(model)

            logger.info(f"Training results logged to run: {run.info.run_id}")

            return run.info.run_id

    def get_best_run(self, metric: str = "val_accuracy", ascending: bool = False):
        """
        Get best run based on a metric

        Args:
            metric: Metric name to optimize
            ascending: Whether lower is better

        Returns:
            Best run data
        """
        runs = self.client.search_runs(
            experiment_ids=[self.experiment_id],
            order_by=[f"metrics.{metric} {'ASC' if ascending else 'DESC'}"],
            max_results=1
        )

        if runs:
            return runs[0]
        return None

    def load_model(self, run_id: str, artifact_path: str = "model"):
        """
        Load model from a run

        Args:
            run_id: MLflow run ID
            artifact_path: Path to model artifact

        Returns:
            Loaded model
        """
        model_uri = f"runs:/{run_id}/{artifact_path}"
        return mlflow.sklearn.load_model(model_uri)

    def register_model(
        self,
        run_id: str,
        model_name: str,
        artifact_path: str = "model"
    ) -> str:
        """
        Register model to MLflow Model Registry

        Args:
            run_id: MLflow run ID
            model_name: Name for registered model
            artifact_path: Path to model artifact

        Returns:
            Model version
        """
        model_uri = f"runs:/{run_id}/{artifact_path}"
        model_details = mlflow.register_model(model_uri, model_name)
        return model_details.version

    def transition_model_stage(
        self,
        model_name: str,
        version: str,
        stage: str
    ):
        """
        Transition model to a stage (Staging, Production, Archived)

        Args:
            model_name: Registered model name
            version: Model version
            stage: Target stage
        """
        self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage
        )
        logger.info(f"Model {model_name} v{version} transitioned to {stage}")


def setup_mlflow_tracking(tracking_uri: Optional[str] = None) -> MLflowTracker:
    """
    Setup MLflow tracking for the project

    Args:
        tracking_uri: MLflow server URI (None for local)

    Returns:
        Configured MLflowTracker instance
    """
    tracker = MLflowTracker(
        experiment_name="forest_cover_prediction",
        tracking_uri=tracking_uri
    )

    logger.info("MLflow tracking initialized")
    return tracker
