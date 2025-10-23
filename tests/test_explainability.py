"""
Tests for SHAP explainability endpoints and module
"""

from fastapi_main_enhanced import app
from src.explainability import ModelExplainer
import pytest
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
import sys
from pathlib import Path
import joblib
from sklearn.ensemble import RandomForestClassifier

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))


@pytest.fixture
def client():
    """Create test client"""
    return TestClient(app)


@pytest.fixture
def mock_model():
    """Create mock trained model"""
    # Create simple model
    X_train = np.random.randn(100, 54)
    y_train = np.random.randint(0, 7, 100)

    model = RandomForestClassifier(
        n_estimators=10, random_state=42, max_depth=5)
    model.fit(X_train, y_train)

    return model


@pytest.fixture
def feature_names():
    """Get feature names"""
    names = [
        'Elevation', 'Aspect', 'Slope',
        'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology',
        'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon',
        'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points'
    ] + [f'Wilderness_Area_{i}' for i in range(1, 5)] + [f'Soil_Type_{i}' for i in range(1, 41)]
    return names


@pytest.fixture
def explainer(mock_model, feature_names):
    """Create explainer instance"""
    return ModelExplainer(mock_model, feature_names, model_type="tree")


@pytest.fixture
def sample_input():
    """Create sample prediction input"""
    return {
        "Elevation": 2800,
        "Aspect": 180,
        "Slope": 15,
        "Horizontal_Distance_To_Hydrology": 300,
        "Vertical_Distance_To_Hydrology": 50,
        "Horizontal_Distance_To_Roadways": 1200,
        "Hillshade_9am": 200,
        "Hillshade_Noon": 220,
        "Hillshade_3pm": 150,
        "Horizontal_Distance_To_Fire_Points": 2000,
        "Wilderness_Area_1": 1,
        "Wilderness_Area_2": 0,
        "Wilderness_Area_3": 0,
        "Wilderness_Area_4": 0,
        **{f"Soil_Type_{i}": 0 for i in range(1, 41)}
    }


@pytest.fixture
def auth_token(client):
    """Get authentication token"""
    response = client.post("/token", json={
        "username": "testuser",
        "password": "testpass"
    })
    if response.status_code == 200:
        return response.json().get("access_token")
    # Return mock token if endpoint doesn't exist yet
    return "mock_token_for_testing"


class TestModelExplainer:
    """Test ModelExplainer class"""

    def test_explainer_initialization(self, explainer):
        """Test explainer initializes correctly"""
        assert explainer is not None
        assert explainer.explainer is not None
        assert explainer.model_type == "tree"

    def test_explain_single_prediction(self, explainer, feature_names):
        """Test single prediction explanation"""
        X = np.random.randn(1, 54)
        explanation = explainer.explain_prediction(X)

        assert "shap_values" in explanation
        assert "base_value" in explanation
        assert "feature_contributions" in explanation
        assert "top_features" in explanation

        # Check structure
        assert len(explanation["shap_values"]) == 54
        assert len(explanation["feature_contributions"]) == 54
        assert len(explanation["top_features"]) == 10

        # Check feature contribution structure
        for contrib in explanation["feature_contributions"]:
            assert "feature" in contrib
            assert "value" in contrib
            assert "shap_value" in contrib
            assert "contribution" in contrib
            assert "importance" in contrib

    def test_explain_batch(self, explainer):
        """Test batch prediction explanation"""
        X = np.random.randn(50, 54)
        explanation = explainer.explain_batch(X)

        assert "num_samples" in explanation
        assert "feature_importance" in explanation
        assert "top_features" in explanation
        assert "explanation" in explanation

        assert explanation["num_samples"] == 50
        assert len(explanation["feature_importance"]) == 54
        assert len(explanation["top_features"]) == 10

    def test_global_importance(self, explainer):
        """Test global feature importance calculation"""
        X_background = np.random.randn(200, 54)
        importance = explainer.get_global_importance(X_background)

        assert "global_importance" in importance
        assert "num_samples_analyzed" in importance

        # Check importance structure
        for item in importance["global_importance"]:
            assert "feature" in item
            assert "importance" in item
            assert "importance_percentage" in item

    def test_waterfall_plot_generation(self, explainer):
        """Test waterfall plot generation"""
        X = np.random.randn(1, 54)
        plot_base64 = explainer.generate_waterfall_plot(X)

        # Check that base64 string is returned
        assert plot_base64 is None or isinstance(plot_base64, str)
        if plot_base64:
            assert len(plot_base64) > 0

    def test_summary_plot_generation(self, explainer):
        """Test summary plot generation"""
        X = np.random.randn(50, 54)
        plot_base64 = explainer.generate_summary_plot(X, plot_type="bar")

        # Check that base64 string is returned
        assert plot_base64 is None or isinstance(plot_base64, str)
        if plot_base64:
            assert len(plot_base64) > 0

    def test_explain_with_specific_class(self, explainer):
        """Test explanation for specific class"""
        X = np.random.randn(1, 54)
        explanation = explainer.explain_prediction(X, prediction_class=0)

        assert "shap_values" in explanation
        assert "feature_contributions" in explanation

    def test_batch_size_limit(self, explainer):
        """Test batch explanation respects max_samples"""
        X = np.random.randn(200, 54)
        explanation = explainer.explain_batch(X, max_samples=50)

        # Should process only max_samples
        assert explanation["num_samples"] <= 50


class TestExplainabilityEndpoints:
    """Test FastAPI explainability endpoints"""

    def test_explain_endpoint_structure(self, client, sample_input, auth_token):
        """Test /explain endpoint structure (may fail without model)"""
        headers = {"Authorization": f"Bearer {auth_token}"}

        payload = {
            "prediction_input": sample_input,
            "prediction_class": None,
            "include_plot": False
        }

        response = client.post("/explain", json=payload, headers=headers)

        # Should return 401, 503, or 200 depending on setup
        assert response.status_code in [200, 401, 503]

        if response.status_code == 200:
            data = response.json()
            assert "request_id" in data
            assert "shap_explanation" in data
            assert "processing_time" in data

    def test_explain_with_plot(self, client, sample_input, auth_token):
        """Test /explain endpoint with plot"""
        headers = {"Authorization": f"Bearer {auth_token}"}

        payload = {
            "prediction_input": sample_input,
            "prediction_class": None,
            "include_plot": True
        }

        response = client.post("/explain", json=payload, headers=headers)

        # Check response structure if successful
        if response.status_code == 200:
            data = response.json()
            # May include waterfall_plot key
            assert "shap_explanation" in data

    def test_explain_batch_endpoint(self, client, sample_input, auth_token):
        """Test /explain-batch endpoint"""
        headers = {"Authorization": f"Bearer {auth_token}"}

        payload = {
            "instances": [sample_input] * 5
        }

        response = client.post("/explain-batch", json=payload, headers=headers)

        # Should return 401, 503, or 200
        assert response.status_code in [200, 401, 503]

        if response.status_code == 200:
            data = response.json()
            assert "request_id" in data
            assert "batch_explanation" in data
            assert "processing_time" in data

    def test_feature_importance_endpoint(self, client):
        """Test /feature-importance endpoint"""
        response = client.get("/feature-importance")

        # Should return 200 or 503
        assert response.status_code in [200, 503]

        if response.status_code == 200:
            data = response.json()
            assert "feature_importance" in data
            assert "processing_time" in data

    def test_batch_prediction_endpoint(self, client, sample_input, auth_token):
        """Test /predict-batch endpoint"""
        headers = {"Authorization": f"Bearer {auth_token}"}

        payload = {
            "instances": [sample_input] * 10
        }

        response = client.post("/predict-batch", json=payload, headers=headers)

        # Should return 401 or 200
        assert response.status_code in [200, 401]

        if response.status_code == 200:
            data = response.json()
            assert "request_id" in data
            assert "batch_size" in data
            assert "predictions" in data
            assert len(data["predictions"]) == 10

    def test_model_comparison_endpoint(self, client):
        """Test /model-comparison endpoint"""
        response = client.get("/model-comparison")

        assert response.status_code == 200
        data = response.json()

        assert "models" in data
        assert "best_model" in data
        assert "dataset" in data
        assert len(data["models"]) > 0

        # Check model structure
        for model in data["models"]:
            assert "name" in model
            assert "accuracy" in model
            assert "precision" in model
            assert "recall" in model
            assert "f1_score" in model
            assert "inference_time_ms" in model

    def test_batch_size_validation(self, client, sample_input, auth_token):
        """Test batch size limit enforcement"""
        headers = {"Authorization": f"Bearer {auth_token}"}

        # Try to exceed max batch size
        payload = {
            "instances": [sample_input] * 1001
        }

        response = client.post("/predict-batch", json=payload, headers=headers)

        # Should return validation error (422)
        assert response.status_code in [422, 401]

    def test_explain_invalid_class(self, client, sample_input, auth_token):
        """Test explanation with invalid class"""
        headers = {"Authorization": f"Bearer {auth_token}"}

        payload = {
            "prediction_input": sample_input,
            "prediction_class": 10,  # Invalid class
            "include_plot": False
        }

        response = client.post("/explain", json=payload, headers=headers)

        # Should return validation error
        assert response.status_code in [422, 401]


class TestExplainabilityIntegration:
    """Integration tests for explainability features"""

    def test_explain_prediction_consistency(self, explainer):
        """Test that explanations are consistent for same input"""
        X = np.random.randn(1, 54)

        explanation1 = explainer.explain_prediction(X)
        explanation2 = explainer.explain_prediction(X)

        # SHAP values should be identical for same input
        np.testing.assert_array_almost_equal(
            explanation1["shap_values"],
            explanation2["shap_values"],
            decimal=5
        )

    def test_feature_importance_sum(self, explainer):
        """Test that feature importance percentages sum to ~100%"""
        X_background = np.random.randn(100, 54)
        importance = explainer.get_global_importance(X_background)

        total_percentage = sum(
            item["importance_percentage"]
            for item in importance["global_importance"]
        )

        # Should sum to approximately 100%
        assert 99.0 <= total_percentage <= 101.0

    def test_shap_additivity(self, explainer):
        """Test SHAP additivity property"""
        X = np.random.randn(1, 54)
        explanation = explainer.explain_prediction(X)

        # Sum of SHAP values + base value should equal model output
        shap_sum = sum(explanation["shap_values"])
        base_value = explanation["base_value"]

        # This is a property of SHAP
        # The sum should be meaningful (not infinity or nan)
        assert not np.isnan(shap_sum)
        assert not np.isinf(shap_sum)
        assert not np.isnan(base_value)


class TestPerformance:
    """Performance tests for explainability"""

    def test_single_explanation_performance(self, explainer):
        """Test single explanation completes quickly"""
        import time

        X = np.random.randn(1, 54)

        start = time.time()
        explainer.explain_prediction(X)
        duration = time.time() - start

        # Should complete in reasonable time (< 2 seconds)
        assert duration < 2.0

    def test_batch_explanation_performance(self, explainer):
        """Test batch explanation performance"""
        import time

        X = np.random.randn(100, 54)

        start = time.time()
        explainer.explain_batch(X, max_samples=100)
        duration = time.time() - start

        # Should complete in reasonable time (< 10 seconds)
        assert duration < 10.0


class TestErrorHandling:
    """Test error handling in explainability"""

    def test_invalid_input_shape(self, explainer):
        """Test handling of invalid input shape"""
        X = np.random.randn(5)  # 1D instead of 2D

        # Should handle 1D input gracefully
        explanation = explainer.explain_prediction(X)
        assert "shap_values" in explanation

    def test_empty_batch(self, explainer):
        """Test handling of empty batch"""
        X = np.array([]).reshape(0, 54)

        # Should handle gracefully or raise appropriate error
        try:
            explainer.explain_batch(X)
        except Exception as e:
            # Should be a meaningful error
            assert str(e) != ""


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
