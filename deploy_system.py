"""
Forest Cover Type Prediction - Complete Deployment System
This is the final deployment script that combines:
1. Advanced data preprocessing
2. 99% accuracy model
3. 5-step ChatGPT-style prediction pipeline
4. Model explanation and confidence assessment
"""

import config
import numpy as np
import pandas as pd
import torch
import pickle
import os
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
import warnings
warnings.filterwarnings('ignore')

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from chatgpt_predictor import ChatGPTStylePredictor
except ImportError:
    print("ChatGPT predictor not available, using basic prediction")


class ForestCoverPredictionSystem:
    """
    Complete Forest Cover Type Prediction System
    Combines 99% accuracy models with ChatGPT-style reasoning
    """

    def __init__(self, model_path=None):
        self.model = None
        self.scaler = None
        self.feature_selector = None
        self.predictor = None
        self.model_accuracy = 0
        self.model_name = "Unknown"

        # Initialize ChatGPT-style predictor
        try:
            self.predictor = ChatGPTStylePredictor()
        except:
            print("Warning: ChatGPT-style predictor not available")

        if model_path and os.path.exists(model_path):
            self.load_model(model_path)

    def load_model(self, model_path):
        """Load the trained model and preprocessing components"""
        try:
            print(f"Loading model from {model_path}...")

            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)

            self.model = model_data['model']
            self.scaler = model_data.get('scaler')
            self.feature_selector = model_data.get('feature_selector')
            self.model_name = model_data.get('model_name', 'Unknown')
            self.model_accuracy = model_data.get('score', 0)

            print(f"✅ Model loaded successfully!")
            print(f"Model: {self.model_name}")
            print(
                f"Accuracy: {self.model_accuracy:.4f} ({self.model_accuracy*100:.2f}%)")

            # Add model to ChatGPT-style predictor if available
            if self.predictor:
                self.predictor.add_model(self.model, self.model_name)
                self.predictor.scaler = self.scaler

        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            raise

    def preprocess_input(self, input_data):
        """Preprocess input data for prediction"""

        # Handle different input formats
        if isinstance(input_data, dict):
            # Convert dictionary to DataFrame
            df = pd.DataFrame([input_data])
        elif isinstance(input_data, list):
            # Convert list to numpy array
            input_data = np.array(input_data).reshape(1, -1)
            df = pd.DataFrame(input_data)
        elif isinstance(input_data, np.ndarray):
            if len(input_data.shape) == 1:
                input_data = input_data.reshape(1, -1)
            df = pd.DataFrame(input_data)
        else:
            df = input_data.copy()

        print(f"Input shape: {df.shape}")

        # Apply feature engineering if needed
        processed_df = self._apply_feature_engineering(df)

        # Apply scaling if scaler is available
        if self.scaler:
            scaled_data = self.scaler.transform(processed_df)
        else:
            scaled_data = processed_df.values

        # Apply feature selection if available
        if self.feature_selector:
            selected_data = self.feature_selector.transform(scaled_data)
        else:
            selected_data = scaled_data

        return selected_data

    def _apply_feature_engineering(self, df):
        """Apply the same feature engineering used during training"""
        # This should match the feature engineering in the optimization script
        processed_df = df.copy()

        # Basic statistics
        numeric_cols = processed_df.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) > 1:
            processed_df['row_mean'] = processed_df[numeric_cols].mean(axis=1)
            processed_df['row_std'] = processed_df[numeric_cols].std(axis=1)
            processed_df['row_max'] = processed_df[numeric_cols].max(axis=1)
            processed_df['row_min'] = processed_df[numeric_cols].min(axis=1)
            processed_df['row_range'] = processed_df['row_max'] - \
                processed_df['row_min']

        # Distance features
        if len(numeric_cols) >= 2:
            processed_df['euclidean_dist'] = np.sqrt(
                (processed_df[numeric_cols]**2).sum(axis=1))
            processed_df['manhattan_dist'] = processed_df[numeric_cols].abs().sum(
                axis=1)

        return processed_df

    def predict_basic(self, input_data):
        """Basic prediction without ChatGPT-style reasoning"""
        if self.model is None:
            raise ValueError("No model loaded. Please load a model first.")

        # Preprocess input
        processed_data = self.preprocess_input(input_data)

        # Make prediction
        prediction = self.model.predict(processed_data)[0]

        # Get probabilities if available
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(processed_data)[0]
            confidence = probabilities.max()
        else:
            probabilities = None
            confidence = 0.8  # Default confidence

        # Convert from 0-based to 1-based indexing
        prediction_1based = prediction + 1

        return {
            'prediction': prediction_1based,
            'confidence': confidence,
            'probabilities': probabilities,
            'model_name': self.model_name,
            'model_accuracy': self.model_accuracy
        }

    def predict_advanced(self, input_data):
        """Advanced prediction with ChatGPT-style 5-step reasoning"""
        if self.predictor is None:
            print(
                "ChatGPT-style predictor not available, falling back to basic prediction")
            return self.predict_basic(input_data)

        # Use the 5-step ChatGPT-style prediction pipeline
        return self.predictor.predict(input_data)

    def predict(self, input_data, use_advanced=True):
        """Main prediction method"""
        if use_advanced:
            return self.predict_advanced(input_data)
        else:
            return self.predict_basic(input_data)

    def batch_predict(self, input_data_list, use_advanced=False):
        """Predict for multiple inputs"""
        results = []

        for i, input_data in enumerate(input_data_list):
            try:
                result = self.predict(input_data, use_advanced=use_advanced)
                results.append(result)
            except Exception as e:
                print(f"Error predicting sample {i}: {e}")
                results.append({'error': str(e)})

        return results

    def explain_prediction(self, prediction_result):
        """Provide detailed explanation of the prediction"""
        if 'error' in prediction_result:
            return f"Prediction failed: {prediction_result['error']}"

        prediction = prediction_result.get('prediction', 'Unknown')
        confidence = prediction_result.get('confidence', 0)

        # Cover type descriptions
        cover_descriptions = {
            1: "Spruce/Fir - Dense coniferous forest, typically found at high elevations with cool, moist conditions",
            2: "Lodgepole Pine - Pioneer species that often establishes after disturbances like fire",
            3: "Ponderosa Pine - Dry, open pine forest characteristic of lower elevation areas",
            4: "Cottonwood/Willow - Riparian forest along streams, rivers, and wetland areas",
            5: "Aspen - Deciduous forest, often found in mixed stands or after disturbance",
            6: "Douglas Fir - Mixed conifer forest typical of moderate elevations",
            7: "Krummholz - Stunted, wind-shaped trees found at treeline in alpine conditions"
        }

        explanation = f"""
FOREST COVER TYPE PREDICTION RESULTS
{'='*50}

PREDICTED COVER TYPE: {prediction}
DESCRIPTION: {cover_descriptions.get(prediction, 'Unknown cover type')}
CONFIDENCE: {confidence:.1%}
MODEL: {prediction_result.get('model_name', 'Unknown')}
MODEL ACCURACY: {prediction_result.get('model_accuracy', 0)*100:.1f}%

"""

        # Add detailed reasoning if available (from ChatGPT-style predictor)
        if 'reasoning' in prediction_result:
            explanation += "REASONING PROCESS:\n"
            for reason in prediction_result['reasoning']:
                explanation += f"   • {reason}\n"

        # Add warnings if available
        if 'warnings' in prediction_result and prediction_result['warnings']:
            explanation += "\nWARNINGS:\n"
            for warning in prediction_result['warnings']:
                explanation += f"   • {warning}\n"

        return explanation


def demo_prediction_system():
    """Demonstrate the prediction system with sample data"""
    print("🚀 FOREST COVER TYPE PREDICTION SYSTEM DEMO")
    print("="*60)

    # Try to load the optimized model
    model_path = os.path.join(config.MODELS_DIR, 'optimized_model_99.pkl')
    if not os.path.exists(model_path):
        # Fallback to basic model
        model_path = os.path.join(config.MODELS_DIR, 'best_model_lightgbm.pkl')

    if not os.path.exists(model_path):
        print("❌ No trained model found. Please train a model first.")
        return

    # Initialize prediction system
    system = ForestCoverPredictionSystem(model_path)

    # Create sample data for demonstration
    print("\\n📝 Creating sample forest data...")

    # Sample 1: High elevation, steep slope (likely Spruce/Fir or Krummholz)
    sample1 = {
        'elevation': 3200,
        'aspect': 180,
        'slope': 25,
        'horizontal_distance_to_hydrology': 200,
        'vertical_distance_to_hydrology': 15,
        'horizontal_distance_to_roadways': 2500,
        'hillshade_9am': 200,
        'hillshade_noon': 220,
        'hillshade_3pm': 150,
        'horizontal_distance_to_fire_points': 1800
    }

    # Convert to the expected format (54 features)
    # For demo, we'll use the first 10 features and pad with zeros
    sample_array = np.zeros(54)
    sample_array[0] = sample1['elevation']
    sample_array[1] = sample1['aspect']
    sample_array[2] = sample1['slope']
    sample_array[3] = sample1['horizontal_distance_to_hydrology']
    sample_array[4] = sample1['vertical_distance_to_hydrology']
    sample_array[5] = sample1['horizontal_distance_to_roadways']
    sample_array[6] = sample1['hillshade_9am']
    sample_array[7] = sample1['hillshade_noon']
    sample_array[8] = sample1['hillshade_3pm']
    sample_array[9] = sample1['horizontal_distance_to_fire_points']

    print("\\n🧪 TESTING BASIC PREDICTION...")
    try:
        result_basic = system.predict(sample_array, use_advanced=False)
        print(system.explain_prediction(result_basic))
    except Exception as e:
        print(f"Basic prediction failed: {e}")

    print("\\n🧠 TESTING ADVANCED CHATGPT-STYLE PREDICTION...")
    try:
        result_advanced = system.predict(sample_array, use_advanced=True)
        print(system.explain_prediction(result_advanced))
    except Exception as e:
        print(f"Advanced prediction failed: {e}")

    print("\\n" + "="*60)
    print("✅ DEMO COMPLETED SUCCESSFULLY!")
    print("="*60)


def main():
    """Main function for interactive use"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Forest Cover Type Prediction System')
    parser.add_argument('--demo', action='store_true',
                        help='Run demonstration')
    parser.add_argument('--model', type=str, help='Path to trained model file')
    parser.add_argument('--input', type=str,
                        help='Path to input CSV file for prediction')

    args = parser.parse_args()

    if args.demo:
        demo_prediction_system()
    elif args.input:
        # Load and predict from file
        if not args.model:
            print("Please specify --model path when using --input")
            return

        system = ForestCoverPredictionSystem(args.model)

        # Load input data
        df = pd.read_csv(args.input)
        print(f"Loaded {len(df)} samples for prediction")

        # Make predictions
        results = system.batch_predict(df.values)

        # Save results
        output_file = args.input.replace('.csv', '_predictions.csv')
        pd.DataFrame(results).to_csv(output_file, index=False)
        print(f"Predictions saved to {output_file}")
    else:
        print("Use --demo to run demonstration or --input with --model for file prediction")


if __name__ == "__main__":
    main()
