"""
Test script to verify FastAPI setup and dependencies
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def test_imports():
    """Test all required imports"""
    print("Testing imports...")

    try:
        # Test FastAPI imports
        from fastapi import FastAPI, HTTPException, UploadFile, File
        print("✅ FastAPI imports successful")

        # Test other imports
        import pandas as pd
        import numpy as np
        import joblib
        print("✅ Data science libraries imported")

        # Test config
        import config
        print(f"✅ Config loaded - Device: {config.DEVICE}")

        # Test predictor import
        from src.chatgpt_predictor import ChatGPTStylePredictor
        print("✅ ChatGPT predictor imported")

        return True

    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False


def test_model_loading():
    """Test model loading"""
    print("\nTesting model loading...")

    try:
        import config
        from src.chatgpt_predictor import ChatGPTStylePredictor

        # Check if model files exist
        model_path = os.path.join(config.MODELS_DIR, 'best_model_lightgbm.pkl')
        if not os.path.exists(model_path):
            print(f"❌ Model file not found: {model_path}")
            return False

        print(f"✅ Model file found: {model_path}")

        # Try to initialize predictor
        predictor = ChatGPTStylePredictor(model_path)
        print("✅ Predictor initialized successfully")

        return True

    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False


def test_sample_prediction():
    """Test a sample prediction"""
    print("\nTesting sample prediction...")

    try:
        import config
        import pandas as pd
        from src.chatgpt_predictor import ChatGPTStylePredictor

        model_path = os.path.join(config.MODELS_DIR, 'best_model_lightgbm.pkl')
        predictor = ChatGPTStylePredictor(model_path)

        # Create sample data
        feature_names = [
            'Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
            'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
            'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
            'Horizontal_Distance_To_Fire_Points'
        ] + [f'Wilderness_Area{i}' for i in range(1, 5)] + [f'Soil_Type{i}' for i in range(1, 41)]

        # Sample data
        sample_data = [3200, 215, 18, 450, 75, 1200,
                       195, 235, 158, 2100] + [0, 0, 1, 0] + [0]*40
        sample_data[13] = 1  # Set one soil type

        df = pd.DataFrame([sample_data], columns=feature_names)

        # Make prediction
        result = predictor.predict(df)
        print(f"✅ Sample prediction successful: {result}")

        return True

    except Exception as e:
        print(f"❌ Sample prediction failed: {e}")
        return False


def main():
    """Run all tests"""
    print("🌲 Forest Cover Prediction - System Verification\n")

    tests = [
        test_imports,
        test_model_loading,
        test_sample_prediction
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1
        print()

    print(f"Test Results: {passed}/{total} passed")

    if passed == total:
        print("🎉 All tests passed! System is ready.")
        return True
    else:
        print("❌ Some tests failed. Please fix issues before starting server.")
        return False


if __name__ == "__main__":
    main()
