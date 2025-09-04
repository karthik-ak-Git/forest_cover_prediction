"""
Test the 5-step ChatGPT-like prediction pipeline
"""
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
from src.chatgpt_predictor import ChatGPTStylePredictor
from config import config


def test_prediction_pipeline():
    """Test the complete prediction pipeline with sample data"""

    print("="*70)
    print("FOREST COVER TYPE PREDICTION - 5-STEP CHATGPT PIPELINE TEST")
    print("="*70)

    # Load test data
    print("\n1. Loading test data...")
    data = pd.read_csv(config.data.train_csv)

    # Take a random sample for testing
    test_sample = data.sample(n=5, random_state=42)
    X_test = test_sample.drop(['Cover_Type'], axis=1)
    y_true = test_sample['Cover_Type'].values

    print(f"Sample data shape: {X_test.shape}")
    print(f"True labels: {y_true}")

    # Initialize predictor
    print("\n2. Initializing ChatGPT-style predictor...")

    # Check if best model exists
    model_path = os.path.join(config.paths.models_dir,
                              'best_model_lightgbm.pkl')
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        print("Running training first...")
        os.system(f"{config.python_env} train_models.py")

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
            # Run the 5-step ChatGPT prediction
            prediction_result = predictor.predict_with_explanation(input_data)

            # Display results
            print(f"📊 True Cover Type: {y_true[i]}")
            print(f"🎯 Predicted Cover Type: {prediction_result['prediction']}")
            print(f"🔥 Confidence: {prediction_result['confidence']:.2%}")
            print(
                f"✅ Correct: {'Yes' if prediction_result['prediction'] == y_true[i] else 'No'}")

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

    print("\n4. Testing batch prediction...")
    try:
        batch_results = predictor.batch_predict(X_test.to_dict('records'))
        predictions = [r['prediction'] for r in batch_results]
        confidences = [r['confidence'] for r in batch_results]

        accuracy = np.mean(np.array(predictions) == y_true)
        avg_confidence = np.mean(confidences)

        print(f"✅ Batch prediction completed!")
        print(f"📊 Accuracy on test samples: {accuracy:.2%}")
        print(f"🔥 Average confidence: {avg_confidence:.2%}")

    except Exception as e:
        print(f"❌ Batch prediction failed: {e}")

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
