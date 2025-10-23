"""
Performance benchmark tests for the ML system
"""
import pytest
import pandas as pd
import numpy as np
import time
from pathlib import Path
import joblib


class TestPerformanceBenchmarks:
    """Performance and load testing suite"""

    @pytest.fixture
    def sample_data(self):
        """Create sample dataset"""
        np.random.seed(42)
        n_samples = 1000

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

    @pytest.fixture
    def trained_model(self, sample_data):
        """Create and train a model for testing"""
        from sklearn.ensemble import RandomForestClassifier
        from src.data_preprocessing import DataPreprocessor

        X = sample_data.drop('cover_type', axis=1)
        y = sample_data['cover_type']

        preprocessor = DataPreprocessor()
        X_processed = preprocessor.fit_transform(X, y)

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_processed, y)

        return model, preprocessor

    def test_single_prediction_latency(self, sample_data, trained_model):
        """Test single prediction latency"""
        model, preprocessor = trained_model
        X = sample_data.drop('cover_type', axis=1).iloc[0:1]

        X_processed = preprocessor.transform(X)

        # Warm up
        for _ in range(10):
            model.predict(X_processed)

        # Measure
        latencies = []
        for _ in range(100):
            start = time.perf_counter()
            model.predict(X_processed)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # ms

        avg_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)

        print(f"\nSingle Prediction Latency:")
        print(f"  Average: {avg_latency:.2f}ms")
        print(f"  P95: {p95_latency:.2f}ms")
        print(f"  P99: {p99_latency:.2f}ms")

        # Assert reasonable latency (< 50ms average)
        assert avg_latency < 50
        assert p99_latency < 100

    def test_batch_prediction_throughput(self, sample_data, trained_model):
        """Test batch prediction throughput"""
        model, preprocessor = trained_model

        batch_sizes = [10, 50, 100, 500]
        results = {}

        for batch_size in batch_sizes:
            X = sample_data.drop('cover_type', axis=1).iloc[:batch_size]
            X_processed = preprocessor.transform(X)

            start = time.perf_counter()
            model.predict(X_processed)
            end = time.perf_counter()

            elapsed = end - start
            throughput = batch_size / elapsed

            results[batch_size] = {
                'time': elapsed,
                'throughput': throughput
            }

        print(f"\nBatch Prediction Throughput:")
        for batch_size, metrics in results.items():
            print(
                f"  Batch {batch_size}: {metrics['throughput']:.0f} predictions/sec")

        # Should handle at least 100 predictions/sec for small batches
        assert results[100]['throughput'] > 100

    def test_preprocessing_performance(self, sample_data):
        """Test preprocessing performance"""
        from src.data_preprocessing import DataPreprocessor

        X = sample_data.drop('cover_type', axis=1)
        y = sample_data['cover_type']

        preprocessor = DataPreprocessor()

        # Fit transform timing
        start = time.perf_counter()
        preprocessor.fit_transform(X, y)
        fit_time = time.perf_counter() - start

        # Transform timing
        start = time.perf_counter()
        preprocessor.transform(X[:100])
        transform_time = time.perf_counter() - start

        print(f"\nPreprocessing Performance:")
        print(f"  Fit-transform: {fit_time:.3f}s")
        print(f"  Transform (100 samples): {transform_time*1000:.2f}ms")

        # Should be fast
        assert fit_time < 5.0  # 5 seconds
        assert transform_time < 0.1  # 100ms for 100 samples

    def test_memory_usage(self, sample_data):
        """Test memory usage during operations"""
        import sys
        from src.data_preprocessing import DataPreprocessor

        X = sample_data.drop('cover_type', axis=1)
        y = sample_data['cover_type']

        initial_size = sys.getsizeof(X) + sys.getsizeof(y)

        preprocessor = DataPreprocessor()
        X_processed = preprocessor.fit_transform(X, y)

        processed_size = sys.getsizeof(X_processed)

        print(f"\nMemory Usage:")
        print(f"  Initial: {initial_size / 1024:.2f} KB")
        print(f"  Processed: {processed_size / 1024:.2f} KB")
        print(f"  Ratio: {processed_size / initial_size:.2f}x")

        # Memory shouldn't explode
        assert processed_size < initial_size * 10

    def test_concurrent_predictions(self, sample_data, trained_model):
        """Test concurrent prediction handling"""
        import concurrent.futures

        model, preprocessor = trained_model
        X = sample_data.drop('cover_type', axis=1).iloc[:100]
        X_processed = preprocessor.transform(X)

        def make_prediction(idx):
            return model.predict(X_processed[idx:idx+1])

        start = time.perf_counter()

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_prediction, i) for i in range(100)]
            results = [f.result()
                       for f in concurrent.futures.as_completed(futures)]

        elapsed = time.perf_counter() - start

        print(f"\nConcurrent Predictions:")
        print(f"  100 predictions with 10 workers: {elapsed:.3f}s")
        print(f"  Throughput: {100/elapsed:.0f} predictions/sec")

        assert len(results) == 100
        assert elapsed < 5.0  # Should complete in reasonable time

    def test_model_loading_time(self, sample_data, trained_model, tmp_path):
        """Test model loading performance"""
        model, preprocessor = trained_model

        # Save model
        model_path = tmp_path / "benchmark_model.joblib"
        joblib.dump(model, model_path)

        # Measure loading time
        start = time.perf_counter()
        loaded_model = joblib.load(model_path)
        load_time = time.perf_counter() - start

        print(f"\nModel Loading:")
        print(f"  Time: {load_time*1000:.2f}ms")

        # Should load quickly
        assert load_time < 1.0  # 1 second

    def test_data_validation_performance(self, sample_data):
        """Test data validation performance"""
        from pydantic import BaseModel, validator

        class PredictionInput(BaseModel):
            elevation: float
            aspect: float
            slope: float

            class Config:
                extra = 'allow'

        # Create test data
        test_records = sample_data.drop(
            'cover_type', axis=1).to_dict('records')[:100]

        start = time.perf_counter()
        for record in test_records:
            PredictionInput(**record)
        elapsed = time.perf_counter() - start

        print(f"\nData Validation:")
        print(f"  100 records: {elapsed*1000:.2f}ms")
        print(f"  Per record: {elapsed*10:.2f}ms")

        # Should be fast
        assert elapsed < 0.5  # 500ms for 100 records

    def test_feature_engineering_performance(self, sample_data):
        """Test feature engineering performance"""
        from src.data_preprocessing import DataPreprocessor

        X = sample_data.drop('cover_type', axis=1)
        y = sample_data['cover_type']

        preprocessor = DataPreprocessor()

        # Measure individual steps if possible
        start = time.perf_counter()
        X_processed = preprocessor.fit_transform(X, y)
        total_time = time.perf_counter() - start

        throughput = len(X) / total_time

        print(f"\nFeature Engineering:")
        print(f"  Total time: {total_time:.3f}s")
        print(f"  Throughput: {throughput:.0f} samples/sec")
        print(f"  Features: {X.shape[1]} → {X_processed.shape[1]}")

        # Should process at least 100 samples/sec
        assert throughput > 100

    @pytest.mark.slow
    def test_stress_test(self, trained_model):
        """Stress test with large dataset"""
        np.random.seed(42)
        n_samples = 10000

        # Create large dataset
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

        large_df = pd.DataFrame(data)

        model, preprocessor = trained_model

        start = time.perf_counter()
        X_processed = preprocessor.transform(large_df)
        predictions = model.predict(X_processed)
        elapsed = time.perf_counter() - start

        throughput = n_samples / elapsed

        print(f"\nStress Test:")
        print(f"  Samples: {n_samples}")
        print(f"  Time: {elapsed:.3f}s")
        print(f"  Throughput: {throughput:.0f} predictions/sec")

        assert len(predictions) == n_samples
        assert elapsed < 30.0  # Should complete within 30 seconds
