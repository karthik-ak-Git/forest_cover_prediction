"""
Unit tests for model training and prediction
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestModelTraining:
    """Test model training functionality"""

    @pytest.fixture
    def sample_data(self):
        """Create sample training data"""
        np.random.seed(42)
        n_samples = 1000
        n_features = 10

        X = np.random.randn(n_samples, n_features)
        y = np.random.randint(1, 8, n_samples)  # 7 classes

        return X, y

    def test_random_forest_training(self, sample_data):
        """Test Random Forest training"""
        X, y = sample_data

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)

        predictions = model.predict(X)
        accuracy = accuracy_score(y, predictions)

        assert accuracy > 0.1  # Should do better than random
        assert len(predictions) == len(y)

    def test_model_save_load(self, sample_data, tmp_path):
        """Test model saving and loading"""
        import joblib

        X, y = sample_data
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)

        # Save model
        model_path = tmp_path / "test_model.pkl"
        joblib.dump(model, model_path)

        # Load model
        loaded_model = joblib.load(model_path)

        # Test predictions match
        pred_original = model.predict(X[:10])
        pred_loaded = loaded_model.predict(X[:10])

        np.testing.assert_array_equal(pred_original, pred_loaded)

    def test_feature_importance(self, sample_data):
        """Test feature importance extraction"""
        X, y = sample_data

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)

        importances = model.feature_importances_

        assert len(importances) == X.shape[1]
        assert np.all(importances >= 0)
        assert np.abs(np.sum(importances) - 1.0) < 0.01  # Should sum to ~1


class TestPredictionAccuracy:
    """Test prediction accuracy"""

    def test_prediction_shape(self):
        """Test prediction output shape"""
        X = np.random.randn(5, 10)
        model = RandomForestClassifier(n_estimators=10, random_state=42)

        # Train on dummy data
        X_train = np.random.randn(100, 10)
        y_train = np.random.randint(1, 8, 100)
        model.fit(X_train, y_train)

        predictions = model.predict(X)

        assert predictions.shape == (5,)
        assert all(1 <= p <= 7 for p in predictions)

    def test_probability_predictions(self):
        """Test probability predictions"""
        X = np.random.randn(5, 10)
        model = RandomForestClassifier(n_estimators=10, random_state=42)

        X_train = np.random.randn(100, 10)
        y_train = np.random.randint(1, 8, 100)
        model.fit(X_train, y_train)

        probabilities = model.predict_proba(X)

        assert probabilities.shape == (5, 7)  # 5 samples, 7 classes
        assert np.all(probabilities >= 0)
        assert np.all(probabilities <= 1)
        assert np.allclose(probabilities.sum(axis=1), 1.0)


class TestFeatureEngineering:
    """Test feature engineering functions"""

    def test_distance_calculation(self):
        """Test Euclidean distance calculation"""
        horizontal = np.array([3, 4, 0])
        vertical = np.array([4, 3, 5])

        expected = np.array([5, 5, 5])
        actual = np.sqrt(horizontal**2 + vertical**2)

        np.testing.assert_array_almost_equal(actual, expected)

    def test_mean_hillshade(self):
        """Test mean hillshade calculation"""
        hillshade_9am = np.array([200, 210, 220])
        hillshade_noon = np.array([220, 230, 240])
        hillshade_3pm = np.array([140, 150, 160])

        expected = np.array([186.67, 196.67, 206.67])
        actual = (hillshade_9am + hillshade_noon + hillshade_3pm) / 3

        np.testing.assert_array_almost_equal(actual, expected, decimal=2)


class TestDataPreprocessing:
    """Test data preprocessing"""

    def test_standard_scaling(self):
        """Test StandardScaler"""
        from sklearn.preprocessing import StandardScaler

        X = np.array([[1, 2], [3, 4], [5, 6]])
        scaler = StandardScaler()

        X_scaled = scaler.fit_transform(X)

        # After scaling, mean should be ~0 and std should be ~1
        assert np.abs(X_scaled.mean()) < 0.01
        assert np.abs(X_scaled.std() - 1.0) < 0.1

    def test_train_test_split(self):
        """Test train/test splitting"""
        from sklearn.model_selection import train_test_split

        X = np.random.randn(100, 10)
        y = np.random.randint(1, 8, 100)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        assert len(X_train) == 80
        assert len(X_test) == 20
        assert len(y_train) == 80
        assert len(y_test) == 20


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
