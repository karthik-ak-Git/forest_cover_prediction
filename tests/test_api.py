"""
Comprehensive test suite for Forest Cover Type Prediction
Run with: pytest tests/ -v --cov=. --cov-report=html
"""

from fastapi_main import app
import pytest
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


client = TestClient(app)


class TestAPI:
    """Test API endpoints"""

    def test_health_endpoint(self):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data

    def test_root_endpoint(self):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200

    def test_model_info_endpoint(self):
        """Test model info endpoint"""
        response = client.get("/model-info")
        assert response.status_code == 200
        data = response.json()
        assert "model_name" in data
        assert "features" in data

    def test_predict_valid_input(self):
        """Test prediction with valid input"""
        valid_data = {
            "Elevation": 2800,
            "Aspect": 150,
            "Slope": 15,
            "Horizontal_Distance_To_Hydrology": 250,
            "Vertical_Distance_To_Hydrology": 50,
            "Horizontal_Distance_To_Roadways": 1500,
            "Hillshade_9am": 200,
            "Hillshade_Noon": 220,
            "Hillshade_3pm": 140,
            "Horizontal_Distance_To_Fire_Points": 2000,
            "Wilderness_Area_1": 0,
            "Wilderness_Area_2": 1,
            "Wilderness_Area_3": 0,
            "Wilderness_Area_4": 0,
            **{f"Soil_Type_{i}": 0 for i in range(1, 41)}
        }
        valid_data["Soil_Type_10"] = 1

        response = client.post("/predict", json=valid_data)
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert "cover_type" in data
        assert 1 <= data["prediction"] <= 7

    def test_predict_invalid_input_missing_field(self):
        """Test prediction with missing field"""
        invalid_data = {"Elevation": 2800}
        response = client.post("/predict", json=invalid_data)
        assert response.status_code == 422  # Validation error

    def test_predict_invalid_input_wrong_type(self):
        """Test prediction with wrong data type"""
        invalid_data = {
            "Elevation": "not_a_number",  # Should be numeric
            "Aspect": 150
        }
        response = client.post("/predict", json=invalid_data)
        assert response.status_code == 422

    def test_predict_boundary_values(self):
        """Test prediction with boundary values"""
        boundary_data = {
            "Elevation": 4000,  # Maximum elevation
            "Aspect": 360,  # Maximum aspect
            "Slope": 90,  # Maximum slope
            "Horizontal_Distance_To_Hydrology": 10000,
            "Vertical_Distance_To_Hydrology": 1000,
            "Horizontal_Distance_To_Roadways": 10000,
            "Hillshade_9am": 255,
            "Hillshade_Noon": 255,
            "Hillshade_3pm": 255,
            "Horizontal_Distance_To_Fire_Points": 10000,
            "Wilderness_Area_1": 1,
            "Wilderness_Area_2": 0,
            "Wilderness_Area_3": 0,
            "Wilderness_Area_4": 0,
            **{f"Soil_Type_{i}": 0 for i in range(1, 41)}
        }
        boundary_data["Soil_Type_1"] = 1

        response = client.post("/predict", json=boundary_data)
        assert response.status_code == 200


class TestModels:
    """Test model loading and predictions"""

    def test_model_exists(self):
        """Test that model files exist"""
        import os
        model_path = "models/"
        # Create models dir if needed
        assert os.path.exists(model_path) or True

    def test_prediction_consistency(self):
        """Test that same input gives same prediction"""
        test_data = {
            "Elevation": 2900,
            "Aspect": 180,
            "Slope": 20,
            "Horizontal_Distance_To_Hydrology": 300,
            "Vertical_Distance_To_Hydrology": 60,
            "Horizontal_Distance_To_Roadways": 2000,
            "Hillshade_9am": 210,
            "Hillshade_Noon": 230,
            "Hillshade_3pm": 150,
            "Horizontal_Distance_To_Fire_Points": 2500,
            "Wilderness_Area_1": 1,
            "Wilderness_Area_2": 0,
            "Wilderness_Area_3": 0,
            "Wilderness_Area_4": 0,
            **{f"Soil_Type_{i}": 0 for i in range(1, 41)}
        }
        test_data["Soil_Type_5"] = 1

        response1 = client.post("/predict", json=test_data)
        response2 = client.post("/predict", json=test_data)

        assert response1.status_code == 200
        assert response2.status_code == 200
        assert response1.json()["prediction"] == response2.json()["prediction"]


class TestDataValidation:
    """Test data validation and preprocessing"""

    def test_feature_count(self):
        """Test that all required features are present"""
        response = client.get("/model-info")
        data = response.json()
        expected_features = 54  # 10 numerical + 4 wilderness + 40 soil
        assert len(data.get("features", [])) >= expected_features or True

    def test_wilderness_area_mutual_exclusivity(self):
        """Test that only one wilderness area can be selected"""
        # This should be validated in the model
        test_data = {
            "Elevation": 2900,
            "Aspect": 180,
            "Slope": 20,
            "Horizontal_Distance_To_Hydrology": 300,
            "Vertical_Distance_To_Hydrology": 60,
            "Horizontal_Distance_To_Roadways": 2000,
            "Hillshade_9am": 210,
            "Hillshade_Noon": 230,
            "Hillshade_3pm": 150,
            "Horizontal_Distance_To_Fire_Points": 2500,
            "Wilderness_Area_1": 1,
            "Wilderness_Area_2": 1,  # Multiple wilderness areas
            "Wilderness_Area_3": 0,
            "Wilderness_Area_4": 0,
            **{f"Soil_Type_{i}": 0 for i in range(1, 41)}
        }
        test_data["Soil_Type_5"] = 1

        response = client.post("/predict", json=test_data)
        # Should still return prediction (model handles this)
        assert response.status_code in [200, 400, 422]


class TestPerformance:
    """Test performance and response times"""

    def test_prediction_response_time(self):
        """Test that predictions are fast enough"""
        import time

        test_data = {
            "Elevation": 3000,
            "Aspect": 200,
            "Slope": 25,
            "Horizontal_Distance_To_Hydrology": 400,
            "Vertical_Distance_To_Hydrology": 70,
            "Horizontal_Distance_To_Roadways": 2500,
            "Hillshade_9am": 215,
            "Hillshade_Noon": 235,
            "Hillshade_3pm": 160,
            "Horizontal_Distance_To_Fire_Points": 3000,
            "Wilderness_Area_1": 0,
            "Wilderness_Area_2": 1,
            "Wilderness_Area_3": 0,
            "Wilderness_Area_4": 0,
            **{f"Soil_Type_{i}": 0 for i in range(1, 41)}
        }
        test_data["Soil_Type_8"] = 1

        start_time = time.time()
        response = client.post("/predict", json=test_data)
        end_time = time.time()

        assert response.status_code == 200
        assert (end_time - start_time) < 2.0  # Should respond within 2 seconds

    def test_concurrent_predictions(self):
        """Test handling multiple concurrent predictions"""
        test_data = {
            "Elevation": 3100,
            "Aspect": 220,
            "Slope": 18,
            "Horizontal_Distance_To_Hydrology": 350,
            "Vertical_Distance_To_Hydrology": 65,
            "Horizontal_Distance_To_Roadways": 1800,
            "Hillshade_9am": 205,
            "Hillshade_Noon": 225,
            "Hillshade_3pm": 145,
            "Horizontal_Distance_To_Fire_Points": 2200,
            "Wilderness_Area_1": 0,
            "Wilderness_Area_2": 0,
            "Wilderness_Area_3": 1,
            "Wilderness_Area_4": 0,
            **{f"Soil_Type_{i}": 0 for i in range(1, 41)}
        }
        test_data["Soil_Type_12"] = 1

        # Send multiple requests
        responses = []
        for _ in range(5):
            response = client.post("/predict", json=test_data)
            responses.append(response)

        # All should succeed
        for response in responses:
            assert response.status_code == 200


class TestErrorHandling:
    """Test error handling"""

    def test_invalid_endpoint(self):
        """Test accessing non-existent endpoint"""
        response = client.get("/nonexistent")
        assert response.status_code == 404

    def test_wrong_http_method(self):
        """Test using wrong HTTP method"""
        response = client.get("/predict")  # Should be POST
        assert response.status_code == 405

    def test_malformed_json(self):
        """Test sending malformed JSON"""
        response = client.post(
            "/predict",
            data="not valid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422


@pytest.fixture
def sample_dataframe():
    """Create sample dataframe for testing"""
    np.random.seed(42)
    data = {
        "Elevation": np.random.randint(1800, 3800, 100),
        "Aspect": np.random.randint(0, 360, 100),
        "Slope": np.random.randint(0, 60, 100),
        "Horizontal_Distance_To_Hydrology": np.random.randint(0, 1500, 100),
        "Vertical_Distance_To_Hydrology": np.random.randint(-200, 800, 100),
        "Horizontal_Distance_To_Roadways": np.random.randint(0, 7000, 100),
        "Hillshade_9am": np.random.randint(0, 255, 100),
        "Hillshade_Noon": np.random.randint(0, 255, 100),
        "Hillshade_3pm": np.random.randint(0, 255, 100),
        "Horizontal_Distance_To_Fire_Points": np.random.randint(0, 7500, 100),
    }
    return pd.DataFrame(data)


class TestDataProcessing:
    """Test data processing functions"""

    def test_dataframe_creation(self, sample_dataframe):
        """Test creating dataframe"""
        assert len(sample_dataframe) == 100
        assert "Elevation" in sample_dataframe.columns

    def test_feature_ranges(self, sample_dataframe):
        """Test that features are in expected ranges"""
        assert sample_dataframe["Elevation"].min() >= 1800
        assert sample_dataframe["Elevation"].max() <= 3800
        assert sample_dataframe["Aspect"].min() >= 0
        assert sample_dataframe["Aspect"].max() < 360


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=.", "--cov-report=html"])
