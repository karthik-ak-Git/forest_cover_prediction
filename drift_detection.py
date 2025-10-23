# Model drift detection script

import numpy as np
import pandas as pd
from scipy import stats
from datetime import datetime, timedelta
import logging
import psycopg2
from typing import Dict, List, Tuple
import json

logger = logging.getLogger(__name__)


class DriftDetector:
    """Detect data and model drift"""

    def __init__(self, db_connection_string: str, threshold: float = 0.05):
        self.db_conn = db_connection_string
        self.threshold = threshold
        self.baseline_stats = {}

    def calculate_baseline_statistics(self, data: pd.DataFrame) -> Dict:
        """Calculate baseline statistics for each feature"""
        stats_dict = {}

        for column in data.columns:
            if data[column].dtype in [np.float64, np.int64]:
                stats_dict[column] = {
                    'mean': data[column].mean(),
                    'std': data[column].std(),
                    'min': data[column].min(),
                    'max': data[column].max(),
                    'median': data[column].median()
                }

        self.baseline_stats = stats_dict
        return stats_dict

    def detect_feature_drift(
        self,
        current_data: pd.DataFrame,
        feature: str
    ) -> Tuple[bool, float, str]:
        """
        Detect drift in a single feature using Kolmogorov-Smirnov test

        Returns:
            (is_drifting, p_value, interpretation)
        """
        if feature not in self.baseline_stats:
            return False, 1.0, "No baseline statistics available"

        baseline_mean = self.baseline_stats[feature]['mean']
        baseline_std = self.baseline_stats[feature]['std']

        # Generate baseline distribution
        np.random.seed(42)
        baseline_samples = np.random.normal(
            baseline_mean,
            baseline_std,
            size=len(current_data)
        )

        # Perform KS test
        statistic, p_value = stats.ks_2samp(
            baseline_samples,
            current_data[feature]
        )

        is_drifting = p_value < self.threshold

        if is_drifting:
            interpretation = f"Significant drift detected (p={p_value:.4f})"
        else:
            interpretation = f"No significant drift (p={p_value:.4f})"

        return is_drifting, p_value, interpretation

    def detect_covariate_drift(
        self,
        baseline_data: pd.DataFrame,
        current_data: pd.DataFrame
    ) -> Dict[str, Dict]:
        """Detect covariate drift across all features"""
        drift_results = {}

        for feature in baseline_data.columns:
            if baseline_data[feature].dtype in [np.float64, np.int64]:
                is_drifting, p_value, interp = self.detect_feature_drift(
                    current_data,
                    feature
                )

                drift_results[feature] = {
                    'is_drifting': is_drifting,
                    'p_value': p_value,
                    'interpretation': interp,
                    'baseline_mean': self.baseline_stats[feature]['mean'],
                    'current_mean': current_data[feature].mean(),
                    'baseline_std': self.baseline_stats[feature]['std'],
                    'current_std': current_data[feature].std()
                }

        return drift_results

    def detect_prediction_drift(
        self,
        baseline_predictions: np.ndarray,
        current_predictions: np.ndarray
    ) -> Tuple[bool, float]:
        """Detect drift in model predictions"""
        statistic, p_value = stats.ks_2samp(
            baseline_predictions,
            current_predictions
        )

        is_drifting = p_value < self.threshold
        return is_drifting, p_value

    def monitor_model_performance(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        baseline_accuracy: float
    ) -> Dict:
        """Monitor model performance degradation"""
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support

        current_accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted'
        )

        accuracy_drop = baseline_accuracy - current_accuracy
        is_degraded = accuracy_drop > 0.05  # 5% threshold

        return {
            'current_accuracy': current_accuracy,
            'baseline_accuracy': baseline_accuracy,
            'accuracy_drop': accuracy_drop,
            'is_degraded': is_degraded,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'timestamp': datetime.utcnow().isoformat()
        }

    def log_drift_to_database(self, drift_results: Dict):
        """Log drift detection results to database"""
        try:
            conn = psycopg2.connect(self.db_conn)
            cursor = conn.cursor()

            for feature, results in drift_results.items():
                cursor.execute("""
                    INSERT INTO drift_monitoring 
                    (feature_name, baseline_mean, current_mean, baseline_std, 
                     current_std, drift_score, is_drifting, checked_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    feature,
                    results['baseline_mean'],
                    results['current_mean'],
                    results['baseline_std'],
                    results['current_std'],
                    results['p_value'],
                    results['is_drifting'],
                    datetime.utcnow()
                ))

            conn.commit()
            cursor.close()
            conn.close()

            logger.info(f"Drift monitoring results logged to database")

        except Exception as e:
            logger.error(f"Error logging drift results: {e}")

    def generate_drift_report(self, drift_results: Dict) -> str:
        """Generate human-readable drift report"""
        report_lines = [
            "=" * 80,
            "DRIFT DETECTION REPORT",
            "=" * 80,
            f"Timestamp: {datetime.utcnow().isoformat()}",
            f"Threshold: {self.threshold}",
            "",
            "FEATURES WITH DETECTED DRIFT:",
            "-" * 80
        ]

        drifting_features = [
            (feat, res) for feat, res in drift_results.items()
            if res['is_drifting']
        ]

        if drifting_features:
            for feature, results in drifting_features:
                report_lines.extend([
                    f"\nFeature: {feature}",
                    f"  P-value: {results['p_value']:.6f}",
                    f"  Baseline Mean: {results['baseline_mean']:.4f}",
                    f"  Current Mean: {results['current_mean']:.4f}",
                    f"  Baseline Std: {results['baseline_std']:.4f}",
                    f"  Current Std: {results['current_std']:.4f}",
                    f"  {results['interpretation']}"
                ])
        else:
            report_lines.append(
                "\nNo significant drift detected in any feature.")

        report_lines.extend([
            "",
            "=" * 80,
            f"Total features checked: {len(drift_results)}",
            f"Features with drift: {len(drifting_features)}",
            "=" * 80
        ])

        return "\n".join(report_lines)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Create sample data
    np.random.seed(42)
    baseline_data = pd.DataFrame({
        'Elevation': np.random.normal(2800, 500, 1000),
        'Aspect': np.random.uniform(0, 360, 1000),
        'Slope': np.random.gamma(2, 10, 1000)
    })

    current_data = pd.DataFrame({
        'Elevation': np.random.normal(2900, 550, 1000),  # Slight drift
        'Aspect': np.random.uniform(0, 360, 1000),
        'Slope': np.random.gamma(2, 10, 1000)
    })

    # Initialize detector
    detector = DriftDetector(
        db_connection_string="postgresql://user:pass@localhost/forest_cover_db",
        threshold=0.05
    )

    # Calculate baseline
    detector.calculate_baseline_statistics(baseline_data)

    # Detect drift
    drift_results = detector.detect_covariate_drift(
        baseline_data, current_data)

    # Generate report
    report = detector.generate_drift_report(drift_results)
    print(report)
