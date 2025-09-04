"""
Test the 5-step ChatGPT-like prediction pipeline
"""
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
from src.chatgpt_predictor import ChatGPTStylePredictor
import config


def test_prediction_pipeline():
    """Test the complete prediction pipeline with sample data"""

    print("="*70)
    print("FOREST COVER TYPE PREDICTION - 5-STEP CHATGPT PIPELINE TEST")
    print("="*70)

    # Load test data
    print("\n1. Loading test data...")
    data = pd.read_csv(config.TRAIN_DATA_PATH)

    # Take a random sample for testing
    test_sample = data.sample(n=5, random_state=42)
    # Remove both target and Id
    X_test = test_sample.drop(['Cover_Type', 'Id'], axis=1)
    y_true = test_sample['Cover_Type'].values

    print(f"Sample data shape: {X_test.shape}")
    print(f"True labels: {y_true}")

    # Initialize predictor
    print("\n2. Initializing ChatGPT-style predictor...")

    # Check if best model exists
    model_path = os.path.join(config.MODELS_DIR, 'best_model_lightgbm.pkl')
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        print("Running training first...")
        os.system("python train_models.py")

    try:
        predictor = ChatGPTStylePredictor(model_path)
        print("✅ Predictor initialized successfully!")
    except Exception as e:
        print(f"❌ Error initializing predictor: {e}")
        return

    # Test predictions
    print("\n3. Running 5-step prediction pipeline...")
    print("-" * 50)

    for i, (idx, row) in enumerate(X_test.iterrows()):
        print(f"\n🌲 PREDICTION {i+1}/5 - Sample ID: {idx}")
        print("=" * 50)

        # Convert to dict for better display
        input_data = row.to_dict()

        try:
            # Run the 5-step ChatGPT prediction - pass as DataFrame row
            prediction_result = predictor.predict(row.to_frame().T)

            # Display results
            print(f"📊 True Cover Type: {y_true[i]}")
            print(
                f"🎯 Predicted Cover Type: {prediction_result.get('prediction', 'Unknown')}")
            print(
                f"🔥 Confidence: {prediction_result.get('confidence', 0.0):.2%}")
            print(
                f"✅ Correct: {'Yes' if prediction_result.get('prediction') == y_true[i] else 'No'}")

            # Show feature importance
            if 'feature_importance' in prediction_result:
                print(f"\n📈 Top 3 Important Features:")
                for j, (feature, importance) in enumerate(prediction_result['feature_importance'][:3]):
                    print(f"   {j+1}. {feature}: {importance:.3f}")

            # Show reasoning
            if 'reasoning' in prediction_result:
                print(f"\n🧠 AI Reasoning:")
                print(f"   {prediction_result['reasoning']}")

        except Exception as e:
            print(f"❌ Prediction failed: {e}")

        print("-" * 50)

    print("\n4. Testing single prediction...")
    try:
        # Test single prediction - pass as DataFrame
        sample_result = predictor.predict(X_test.iloc[[0]])

        print(f"✅ Single prediction completed!")
        print(f"📊 Prediction: {sample_result.get('prediction', 'Unknown')}")
        print(f"🔥 Confidence: {sample_result.get('confidence', 0.0):.2%}")

    except Exception as e:
        print(f"❌ Single prediction failed: {e}")

    print("\n5. Summary")
    print("=" * 50)
    print("🎯 5-Step ChatGPT Pipeline Components:")
    print("   1. Input Processing & Validation")
    print("   2. Context Analysis & Feature Engineering")
    print("   3. AI Reasoning & Model Inference")
    print("   4. Response Generation & Confidence Scoring")
    print("   5. Output Refinement & Explanation")
    print("\n✅ Pipeline test completed successfully!")


if __name__ == "__main__":
    test_prediction_pipeline()
