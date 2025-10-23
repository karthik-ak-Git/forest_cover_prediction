"""
Integration tests for end-to-end pipeline
"""
import pytest
import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path


class TestEndToEndPipeline:
    """Test complete ML pipeline from data to prediction"""

    @pytest.fixture
    def sample_data(self):
        """Create sample dataset"""
        np.random.seed(42)
        n_samples = 500

        data = {
            'elevation': np.random.randint(1500, 3500, n_samples),
            'aspect': np.random.randint(0, 360, n_samples),
            'slope': np.random.randint(0, 60, n_samples),
            'horizontal_distance_to_hydrology': np.random.randint(0, 1000, n_samples),
            'vertical_distance_to_hydrology': np.random.randint(-200, 200, n_samples),
            'horizontal_distance_to_roadways': np.random.randint(0, 5000, n_samples),
            'hillshade_9am': np.random.randint(0, 255, n_samples),
            'hillshade_noon': np.random.randint(0, 255, n_samples),
            'hillshade_3pm': np.random.randint(0, 255, n_samples),
            'horizontal_distance_to_fire_points': np.random.randint(0, 7000, n_samples),
        }

        for i in range(4):
            data[f'wilderness_area_{i}'] = np.random.choice([0, 1], n_samples)

        for i in range(40):
            data[f'soil_type_{i}'] = np.random.choice([0, 1], n_samples)

        data['cover_type'] = np.random.randint(1, 8, n_samples)

        return pd.DataFrame(data)

    def test_data_loading(self):
        """Test data loading from CSV"""
        data_path = Path('data/train.csv')

        if data_path.exists():
            df = pd.read_csv(data_path)

            assert df is not None
            assert len(df) > 0
            assert 'Cover_Type' in df.columns or 'cover_type' in df.columns

    def test_preprocessing_pipeline(self, sample_data):
        """Test complete preprocessing pipeline"""
        from src.data_preprocessing import DataPreprocessor

        X = sample_data.drop('cover_type', axis=1)
        y = sample_data['cover_type']

        preprocessor = DataPreprocessor()
        X_processed = preprocessor.fit_transform(X, y)

        assert X_processed is not None
        assert len(X_processed) == len(X)
        assert not X_processed.isna().any().any()

    def test_model_training_pipeline(self, sample_data):
        """Test complete model training pipeline"""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from src.data_preprocessing import DataPreprocessor

        X = sample_data.drop('cover_type', axis=1)
        y = sample_data['cover_type']

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Preprocess
        preprocessor = DataPreprocessor()
        X_train_processed = preprocessor.fit_transform(X_train, y_train)
        X_test_processed = preprocessor.transform(X_test)

        # Train model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train_processed, y_train)

        # Predict
        predictions = model.predict(X_test_processed)

        assert predictions is not None
        assert len(predictions) == len(X_test)
        assert all(p in range(1, 8) for p in predictions)

    def test_model_persistence(self, sample_data, tmp_path):
        """Test saving and loading models"""
        from sklearn.ensemble import RandomForestClassifier
        from src.data_preprocessing import DataPreprocessor

        X = sample_data.drop('cover_type', axis=1)
        y = sample_data['cover_type']

        # Train model
        preprocessor = DataPreprocessor()
        X_processed = preprocessor.fit_transform(X, y)

        model = RandomForestClassifier(n_estimators=5, random_state=42)
        model.fit(X_processed, y)

        # Save
        model_path = tmp_path / "test_model.joblib"
        joblib.dump(model, model_path)

        # Load
        loaded_model = joblib.load(model_path)

        # Predictions should match
        pred1 = model.predict(X_processed[:10])
        pred2 = loaded_model.predict(X_processed[:10])

        np.testing.assert_array_equal(pred1, pred2)

    def test_prediction_pipeline(self, sample_data):
        """Test complete prediction pipeline"""
        from sklearn.ensemble import RandomForestClassifier
        from src.data_preprocessing import DataPreprocessor

        X = sample_data.drop('cover_type', axis=1)
        y = sample_data['cover_type']

        # Train
        preprocessor = DataPreprocessor()
        X_processed = preprocessor.fit_transform(X, y)

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_processed, y)

        # New data for prediction
        X_new = X.iloc[:5]
        X_new_processed = preprocessor.transform(X_new)

        predictions = model.predict(X_new_processed)
        probabilities = model.predict_proba(X_new_processed)

        assert len(predictions) == 5
        assert probabilities.shape == (5, 7)  # 7 classes
        assert all(0 <= p <= 1 for row in probabilities for p in row)

    def test_batch_prediction(self, sample_data):
        """Test batch prediction performance"""
        from sklearn.ensemble import RandomForestClassifier
        from src.data_preprocessing import DataPreprocessor
        import time

        X = sample_data.drop('cover_type', axis=1)
        y = sample_data['cover_type']

        preprocessor = DataPreprocessor()
        X_processed = preprocessor.fit_transform(X, y)

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_processed, y)

        # Time batch prediction
        batch_size = 100
        X_batch = preprocessor.transform(X.iloc[:batch_size])

        start_time = time.time()
        predictions = model.predict(X_batch)
        end_time = time.time()

        elapsed = end_time - start_time

        assert len(predictions) == batch_size
        assert elapsed < 1.0  # Should be fast

    def test_error_handling(self):
        """Test error handling in pipeline"""
        from src.data_preprocessing import DataPreprocessor

        preprocessor = DataPreprocessor()

        # Test with invalid data
        with pytest.raises((ValueError, KeyError, AttributeError)):
            invalid_data = pd.DataFrame({'invalid_col': [1, 2, 3]})
            preprocessor.transform(invalid_data)

    def test_memory_efficiency(self, sample_data):
        """Test memory usage during processing"""
        from src.data_preprocessing import DataPreprocessor
        import sys

        X = sample_data.drop('cover_type', axis=1)
        y = sample_data['cover_type']

        # Get initial memory
        initial_size = sys.getsizeof(X)

        preprocessor = DataPreprocessor()
        X_processed = preprocessor.fit_transform(X, y)

        processed_size = sys.getsizeof(X_processed)

        # Processed data shouldn't be excessively larger
        assert processed_size < initial_size * 5
