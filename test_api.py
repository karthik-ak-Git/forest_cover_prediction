"""
Test FastAPI endpoints to ensure everything is working properly
"""
import requests
import json

BASE_URL = "http://localhost:8001"


def test_health_endpoint():
    """Test the health check endpoint"""
    print("Testing /health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed: {data}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False


def test_model_info_endpoint():
    """Test the model info endpoint"""
    print("\nTesting /model/info endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/model/info")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Model info retrieved: {data}")
            return True
        else:
            print(f"❌ Model info failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Model info error: {e}")
        return False


def test_prediction_endpoint():
    """Test the prediction endpoint"""
    print("\nTesting /predict endpoint...")
    try:
        # Sample prediction data
        prediction_data = {
            "elevation": 3200.0,
            "aspect": 215.0,
            "slope": 18.0,
            "horizontal_distance_to_hydrology": 450.0,
            "vertical_distance_to_hydrology": 75.0,
            "horizontal_distance_to_roadways": 1200.0,
            "hillshade_9am": 195.0,
            "hillshade_noon": 235.0,
            "hillshade_3pm": 158.0,
            "horizontal_distance_to_fire_points": 2100.0,
            "wilderness_area3": 1,
            "soil_type10": 1
        }

        response = requests.post(
            f"{BASE_URL}/predict",
            headers={"Content-Type": "application/json"},
            json=prediction_data
        )

        if response.status_code == 200:
            data = response.json()
            print(f"✅ Prediction successful:")
            print(f"   Prediction: {data['prediction']}")
            print(f"   Confidence: {data['confidence']:.2%}")
            print(f"   Description: {data['description']}")
            print(f"   Reasoning steps: {len(data['reasoning'])}")
            return True
        else:
            print(f"❌ Prediction failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return False


def main():
    """Run all API tests"""
    print("🚀 Testing FastAPI Endpoints\n")

    tests = [
        test_health_endpoint,
        test_model_info_endpoint,
        test_prediction_endpoint
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1

    print(f"\n📊 Test Results: {passed}/{total} passed")

    if passed == total:
        print("🎉 All API tests passed! Frontend should be working.")
    else:
        print("❌ Some API tests failed. Check server logs.")

    return passed == total


if __name__ == "__main__":
    main()
