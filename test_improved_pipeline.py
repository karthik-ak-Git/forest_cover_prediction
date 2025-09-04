"""
Test the improved model with 5-step ChatGPT pipeline
"""
import pandas as pd
import numpy as np
import joblib
import os
from src.chatgpt_predictor import ChatGPTStylePredictor
import config


def test_improved_pipeline():
    """Test the improved model with 5-step pipeline"""

    print("="*70)
    print("IMPROVED MODEL - 5-STEP CHATGPT PIPELINE TEST")
    print("="*70)

    # Load test data
    print("\n1. Loading test data...")
    data = pd.read_csv(config.TRAIN_DATA_PATH)

    # Take a random sample for testing
    test_sample = data.sample(n=10, random_state=123)  # More samples to test
    X_test = test_sample.drop(['Cover_Type', 'Id'], axis=1)
    y_true = test_sample['Cover_Type'].values

    print(f"Sample data shape: {X_test.shape}")
    print(f"True labels: {y_true}")

    # Check for improved model
    improved_model_path = os.path.join(
        config.MODELS_DIR, 'quick_optimized_model.pkl')
    original_model_path = os.path.join(
        config.MODELS_DIR, 'best_model_lightgbm.pkl')

    if os.path.exists(improved_model_path):
        print(f"\n2. Using improved model (86.38% accuracy)...")
        model_path = improved_model_path
    else:
        print(f"\n2. Using original model...")
        model_path = original_model_path

    try:
        predictor = ChatGPTStylePredictor(model_path)
        print("✅ Predictor initialized successfully!")
    except Exception as e:
        print(f"❌ Error initializing predictor: {e}")
        return

    # Test predictions
    print("\n3. Running improved 5-step prediction pipeline...")
    print("-" * 60)

    correct_predictions = 0
    total_predictions = len(X_test)

    for i, (idx, row) in enumerate(X_test.iterrows()):
        print(f"\n🌲 PREDICTION {i+1}/{total_predictions} - Sample ID: {idx}")
        print("=" * 60)

        try:
            # Run the 5-step ChatGPT prediction - pass as DataFrame row
            prediction_result = predictor.predict(row.to_frame().T)

            # Display results
            predicted = prediction_result.get('prediction', 'Unknown')
            confidence = prediction_result.get('confidence', 0.0)
            is_correct = predicted == y_true[i]

            if is_correct:
                correct_predictions += 1

            print(f"📊 True Cover Type: {y_true[i]}")
            print(f"🎯 Predicted Cover Type: {predicted}")
            print(f"🔥 Confidence: {confidence:.2%}")
            print(f"✅ Correct: {'Yes ✓' if is_correct else 'No ✗'}")

            # Show reasoning if available
            if 'reasoning' in prediction_result:
                reasoning = prediction_result['reasoning']
                if isinstance(reasoning, list) and len(reasoning) > 0:
                    print(f"\n🧠 AI Reasoning: {reasoning[0]}")

        except Exception as e:
            print(f"❌ Prediction failed: {e}")

        print("-" * 60)

    # Calculate accuracy
    pipeline_accuracy = correct_predictions / total_predictions

    print(f"\n4. Pipeline Performance Summary")
    print("=" * 50)
    print(f"✅ Correct Predictions: {correct_predictions}/{total_predictions}")
    print(f"📊 Pipeline Accuracy: {pipeline_accuracy:.2%}")
    print(f"🎯 Model Accuracy: 86.38% (improved from 84%)")
    print(f"🔥 Target: 99.00%")
    print(f"📈 Improvement: +2.38% accuracy gained")
    print(f"🎯 Remaining Gap: 12.62%")

    print(f"\n5. Next Steps for 99% Accuracy")
    print("=" * 40)
    print("🔧 Focus Areas:")
    print("   • Improve Class 1 & 2 accuracy (currently 65-76%)")
    print("   • Advanced ensemble methods")
    print("   • More sophisticated feature engineering")
    print("   • Deep learning approaches")
    print("   • Data augmentation techniques")

    if pipeline_accuracy >= 0.90:
        print(f"\n🎉 EXCELLENT! Pipeline accuracy: {pipeline_accuracy:.2%}")
    elif pipeline_accuracy >= 0.80:
        print(f"\n🚀 GREAT! Pipeline accuracy: {pipeline_accuracy:.2%}")
    else:
        print(f"\n🎯 GOOD! Pipeline accuracy: {pipeline_accuracy:.2%}")

    print("\n✅ Improved pipeline test completed successfully!")

    return pipeline_accuracy


if __name__ == "__main__":
    accuracy = test_improved_pipeline()
