"""
Comprehensive tests for data preprocessing module
"""
import pytest
import pandas as pd
import numpy as np
from src.data_preprocessing import DataPreprocessor


class TestDataPreprocessor:
    """Test suite for DataPreprocessor class"""

    @pytest.fixture
    def sample_data(self):
        """Create sample dataset for testing"""
        np.random.seed(42)
        n_samples = 100

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

        # Add wilderness areas
        for i in range(4):
            data[f'wilderness_area_{i}'] = np.random.choice([0, 1], n_samples)

        # Add soil types
        for i in range(40):
            data[f'soil_type_{i}'] = np.random.choice([0, 1], n_samples)

        # Add target
        data['cover_type'] = np.random.randint(1, 8, n_samples)

        return pd.DataFrame(data)

    @pytest.fixture
    def preprocessor(self):
        """Create preprocessor instance"""
        return DataPreprocessor()

    def test_initialization(self, preprocessor):
        """Test preprocessor initialization"""
        assert preprocessor is not None
        assert hasattr(preprocessor, 'scaler')

    def test_fit_transform(self, preprocessor, sample_data):
        """Test fit_transform method"""
        X = sample_data.drop('cover_type', axis=1)
        y = sample_data['cover_type']

        X_transformed = preprocessor.fit_transform(X, y)

        assert X_transformed is not None
        assert len(X_transformed) == len(X)
        assert X_transformed.shape[1] >= X.shape[1]

    def test_transform(self, preprocessor, sample_data):
        """Test transform method after fitting"""
        X = sample_data.drop('cover_type', axis=1)
        y = sample_data['cover_type']

        # Fit first
        preprocessor.fit_transform(X, y)

        # Then transform
        X_test = X.iloc[:10]
        X_transformed = preprocessor.transform(X_test)

        assert X_transformed is not None
        assert len(X_transformed) == len(X_test)

    def test_feature_engineering(self, preprocessor, sample_data):
        """Test feature engineering creates new features"""
        X = sample_data.drop('cover_type', axis=1)
        y = sample_data['cover_type']

        X_transformed = preprocessor.fit_transform(X, y)

        # Should have more features after engineering
        assert X_transformed.shape[1] > X.shape[1]

    def test_scaling(self, preprocessor, sample_data):
        """Test that features are properly scaled"""
        X = sample_data.drop('cover_type', axis=1)
        y = sample_data['cover_type']

        X_transformed = preprocessor.fit_transform(X, y)

        # Check that numerical features are scaled (mean ~0, std ~1)
        numerical_cols = ['elevation', 'aspect', 'slope']
        for col in numerical_cols:
            if col in X_transformed.columns:
                col_data = X_transformed[col]
                assert abs(col_data.mean()) < 2  # Should be close to 0
                assert abs(col_data.std() - 1) < 1  # Should be close to 1

    def test_missing_values_handling(self, preprocessor, sample_data):
        """Test handling of missing values"""
        X = sample_data.drop('cover_type', axis=1)
        y = sample_data['cover_type']

        # Introduce missing values
        X_missing = X.copy()
        X_missing.iloc[0, 0] = np.nan
        X_missing.iloc[5, 2] = np.nan

        # Should handle missing values without error
        X_transformed = preprocessor.fit_transform(X_missing, y)

        assert not X_transformed.isna().any().any()

    def test_inverse_transform(self, preprocessor, sample_data):
        """Test inverse transform functionality"""
        X = sample_data.drop('cover_type', axis=1)
        y = sample_data['cover_type']

        X_transformed = preprocessor.fit_transform(X, y)

        # If inverse_transform exists
        if hasattr(preprocessor, 'inverse_transform'):
            X_inverse = preprocessor.inverse_transform(X_transformed)

            # Shape should match
            assert X_inverse.shape[0] == X.shape[0]

    def test_reproducibility(self, sample_data):
        """Test that preprocessing is reproducible"""
        X = sample_data.drop('cover_type', axis=1)
        y = sample_data['cover_type']

        preprocessor1 = DataPreprocessor()
        preprocessor2 = DataPreprocessor()

        X1 = preprocessor1.fit_transform(X.copy(), y)
        X2 = preprocessor2.fit_transform(X.copy(), y)

        # Results should be identical
        pd.testing.assert_frame_equal(X1, X2)

    def test_edge_cases(self, preprocessor):
        """Test edge cases"""
        # Single sample
        X_single = pd.DataFrame({
            'elevation': [2500],
            'aspect': [180],
            'slope': [15],
            'horizontal_distance_to_hydrology': [500],
            'vertical_distance_to_hydrology': [50],
            'horizontal_distance_to_roadways': [1000],
            'hillshade_9am': [200],
            'hillshade_noon': [230],
            'hillshade_3pm': [150],
            'horizontal_distance_to_fire_points': [3000],
        })

        # Add wilderness and soil
        for i in range(4):
            X_single[f'wilderness_area_{i}'] = 0
        for i in range(40):
            X_single[f'soil_type_{i}'] = 0

        y_single = pd.Series([1])

        # Should handle single sample
        X_transformed = preprocessor.fit_transform(X_single, y_single)
        assert len(X_transformed) == 1
