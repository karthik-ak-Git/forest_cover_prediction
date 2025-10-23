"""
Model Explainability Module using SHAP
Provides interpretable explanations for forest cover predictions
"""

import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from typing import Dict, List, Tuple, Optional, Union
import joblib
import logging

logger = logging.getLogger(__name__)


class ModelExplainer:
    """
    Wrapper for SHAP explainability with caching and visualization
    """

    def __init__(self, model, feature_names: List[str], model_type: str = "tree"):
        """
        Initialize explainer with model and feature names

        Args:
            model: Trained ML model
            feature_names: List of feature names
            model_type: Type of model ('tree', 'linear', 'kernel')
        """
        self.model = model
        self.feature_names = feature_names
        self.model_type = model_type
        self.explainer = None
        self._initialize_explainer()

    def _initialize_explainer(self):
        """Initialize SHAP explainer based on model type"""
        try:
            if self.model_type == "tree":
                # For tree-based models (RandomForest, XGBoost, LightGBM)
                self.explainer = shap.TreeExplainer(self.model)
                logger.info("Initialized TreeExplainer")
            elif self.model_type == "linear":
                # For linear models
                self.explainer = shap.LinearExplainer(
                    self.model, feature_names=self.feature_names)
                logger.info("Initialized LinearExplainer")
            else:
                # Fallback to KernelExplainer (slower but works for any model)
                self.explainer = shap.KernelExplainer(
                    self.model.predict_proba,
                    shap.sample(np.zeros((100, len(self.feature_names))))
                )
                logger.info("Initialized KernelExplainer")
        except Exception as e:
            logger.error(f"Error initializing explainer: {e}")
            raise

    def explain_prediction(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        prediction_class: Optional[int] = None
    ) -> Dict:
        """
        Generate SHAP explanations for a single prediction

        Args:
            X: Input features (single sample)
            prediction_class: Optional class to explain (for multi-class)

        Returns:
            Dictionary with SHAP values and feature contributions
        """
        try:
            # Ensure X is 2D
            if len(X.shape) == 1:
                X = X.reshape(1, -1)

            # Calculate SHAP values
            shap_values = self.explainer.shap_values(X)

            # Handle multi-class output
            if isinstance(shap_values, list):
                if prediction_class is not None:
                    shap_values_class = shap_values[prediction_class][0]
                else:
                    # Use the predicted class
                    prediction = self.model.predict(X)[0]
                    shap_values_class = shap_values[prediction][0]
            else:
                shap_values_class = shap_values[0]

            # Get base value (expected value)
            if isinstance(self.explainer.expected_value, np.ndarray):
                base_value = self.explainer.expected_value[prediction_class or 0]
            else:
                base_value = self.explainer.expected_value

            # Create feature contribution dictionary
            contributions = []
            for i, (feature, shap_val, feature_val) in enumerate(
                zip(self.feature_names, shap_values_class, X[0])
            ):
                contributions.append({
                    "feature": feature,
                    "value": float(feature_val),
                    "shap_value": float(shap_val),
                    "contribution": "positive" if shap_val > 0 else "negative",
                    "importance": abs(float(shap_val))
                })

            # Sort by absolute SHAP value
            contributions.sort(key=lambda x: x["importance"], reverse=True)

            return {
                "shap_values": [float(v) for v in shap_values_class],
                "base_value": float(base_value),
                "feature_contributions": contributions,
                "top_features": contributions[:10]  # Top 10 most important
            }

        except Exception as e:
            logger.error(f"Error explaining prediction: {e}")
            raise

    def explain_batch(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        max_samples: int = 100
    ) -> Dict:
        """
        Generate SHAP explanations for multiple predictions

        Args:
            X: Input features (multiple samples)
            max_samples: Maximum number of samples to explain

        Returns:
            Dictionary with aggregated SHAP values and statistics
        """
        try:
            # Limit samples for performance
            if len(X) > max_samples:
                X = X[:max_samples]
                logger.info(
                    f"Limited batch explanation to {max_samples} samples")

            # Calculate SHAP values
            shap_values = self.explainer.shap_values(X)

            # Handle multi-class output
            if isinstance(shap_values, list):
                # Average across classes for simplicity
                shap_values_avg = np.mean([np.abs(sv)
                                          for sv in shap_values], axis=0)
            else:
                shap_values_avg = np.abs(shap_values)

            # Calculate mean absolute SHAP value per feature
            mean_shap = np.mean(shap_values_avg, axis=0)

            # Create feature importance dictionary
            feature_importance = []
            for feature, importance in zip(self.feature_names, mean_shap):
                feature_importance.append({
                    "feature": feature,
                    "mean_abs_shap": float(importance)
                })

            # Sort by importance
            feature_importance.sort(
                key=lambda x: x["mean_abs_shap"], reverse=True)

            return {
                "num_samples": len(X),
                "feature_importance": feature_importance,
                "top_features": feature_importance[:10],
                "explanation": "Higher mean absolute SHAP values indicate more important features"
            }

        except Exception as e:
            logger.error(f"Error explaining batch: {e}")
            raise

    def get_global_importance(self, X_background: np.ndarray) -> Dict:
        """
        Calculate global feature importance using SHAP

        Args:
            X_background: Background dataset for SHAP calculation

        Returns:
            Dictionary with global feature importance
        """
        try:
            # Sample background data if too large
            if len(X_background) > 1000:
                indices = np.random.choice(
                    len(X_background), 1000, replace=False)
                X_background = X_background[indices]

            # Calculate SHAP values
            shap_values = self.explainer.shap_values(X_background)

            # Handle multi-class
            if isinstance(shap_values, list):
                shap_values_combined = np.concatenate(shap_values, axis=0)
            else:
                shap_values_combined = shap_values

            # Calculate mean absolute SHAP
            mean_abs_shap = np.mean(np.abs(shap_values_combined), axis=0)

            # Create importance dictionary
            importance = []
            for feature, imp in zip(self.feature_names, mean_abs_shap):
                importance.append({
                    "feature": feature,
                    "importance": float(imp),
                    "importance_percentage": float(imp / np.sum(mean_abs_shap) * 100)
                })

            # Sort by importance
            importance.sort(key=lambda x: x["importance"], reverse=True)

            return {
                "global_importance": importance,
                "num_samples_analyzed": len(X_background)
            }

        except Exception as e:
            logger.error(f"Error calculating global importance: {e}")
            raise

    def generate_waterfall_plot(
        self,
        X: np.ndarray,
        prediction_class: Optional[int] = None,
        max_display: int = 10
    ) -> str:
        """
        Generate SHAP waterfall plot as base64 image

        Args:
            X: Input features (single sample)
            prediction_class: Class to explain
            max_display: Maximum features to display

        Returns:
            Base64 encoded PNG image
        """
        try:
            # Ensure X is 2D
            if len(X.shape) == 1:
                X = X.reshape(1, -1)

            # Calculate SHAP values
            shap_values = self.explainer.shap_values(X)

            # Handle multi-class
            if isinstance(shap_values, list):
                if prediction_class is not None:
                    shap_values_class = shap_values[prediction_class]
                else:
                    prediction = self.model.predict(X)[0]
                    shap_values_class = shap_values[prediction]
                base_value = self.explainer.expected_value[prediction_class or 0]
            else:
                shap_values_class = shap_values
                base_value = self.explainer.expected_value

            # Create waterfall plot
            plt.figure(figsize=(10, 6))
            shap.waterfall_plot(
                shap.Explanation(
                    values=shap_values_class[0],
                    base_values=base_value,
                    data=X[0],
                    feature_names=self.feature_names
                ),
                max_display=max_display,
                show=False
            )

            # Convert to base64
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()

            return img_base64

        except Exception as e:
            logger.error(f"Error generating waterfall plot: {e}")
            return None

    def generate_summary_plot(
        self,
        X: np.ndarray,
        plot_type: str = "bar"
    ) -> str:
        """
        Generate SHAP summary plot as base64 image

        Args:
            X: Input features (multiple samples)
            plot_type: 'bar', 'dot', or 'violin'

        Returns:
            Base64 encoded PNG image
        """
        try:
            # Limit samples for performance
            if len(X) > 500:
                X = X[:500]

            # Calculate SHAP values
            shap_values = self.explainer.shap_values(X)

            # Handle multi-class (average across classes)
            if isinstance(shap_values, list):
                # Use first class for visualization
                shap_values_plot = shap_values[0]
            else:
                shap_values_plot = shap_values

            # Create summary plot
            plt.figure(figsize=(10, 8))
            shap.summary_plot(
                shap_values_plot,
                X,
                feature_names=self.feature_names,
                plot_type=plot_type,
                show=False
            )

            # Convert to base64
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()

            return img_base64

        except Exception as e:
            logger.error(f"Error generating summary plot: {e}")
            return None


def load_explainer(model_path: str, feature_names: List[str], model_type: str = "tree") -> ModelExplainer:
    """
    Load model and create explainer

    Args:
        model_path: Path to saved model
        feature_names: List of feature names
        model_type: Type of model

    Returns:
        ModelExplainer instance
    """
    try:
        model = joblib.load(model_path)
        explainer = ModelExplainer(model, feature_names, model_type)
        logger.info(f"Loaded explainer from {model_path}")
        return explainer
    except Exception as e:
        logger.error(f"Error loading explainer: {e}")
        raise
