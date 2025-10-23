"""
Data validation pipeline using Great Expectations style validation
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of data validation"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    stats: Dict[str, Any]


class ForestCoverDataValidator:
    """Validator for forest cover prediction data"""

    def __init__(self):
        """Initialize validator with expected schema"""
        self.feature_ranges = {
            'elevation': (1500, 4000),
            'aspect': (0, 360),
            'slope': (0, 90),
            'horizontal_distance_to_hydrology': (0, 10000),
            'vertical_distance_to_hydrology': (-500, 500),
            'horizontal_distance_to_roadways': (0, 10000),
            'hillshade_9am': (0, 255),
            'hillshade_noon': (0, 255),
            'hillshade_3pm': (0, 255),
            'horizontal_distance_to_fire_points': (0, 10000),
        }

        self.required_features = list(self.feature_ranges.keys())

        # Add wilderness areas
        for i in range(4):
            self.required_features.append(f'wilderness_area_{i}')

        # Add soil types
        for i in range(40):
            self.required_features.append(f'soil_type_{i}')

    def validate(self, data: pd.DataFrame) -> ValidationResult:
        """
        Validate input data

        Args:
            data: DataFrame to validate

        Returns:
            ValidationResult with validation status and details
        """
        errors = []
        warnings = []
        stats = {}

        # Check if DataFrame is empty
        if len(data) == 0:
            errors.append("Input data is empty")
            return ValidationResult(False, errors, warnings, stats)

        # Check for required features
        missing_features = []
        for feature in self.required_features:
            if feature not in data.columns:
                missing_features.append(feature)

        if missing_features:
            errors.append(f"Missing required features: {missing_features[:5]}")

        # Validate feature ranges
        for feature, (min_val, max_val) in self.feature_ranges.items():
            if feature in data.columns:
                col_data = data[feature]

                # Check for missing values
                missing_count = col_data.isna().sum()
                if missing_count > 0:
                    warnings.append(
                        f"{feature}: {missing_count} missing values "
                        f"({missing_count/len(data)*100:.1f}%)"
                    )

                # Check range
                valid_data = col_data.dropna()
                if len(valid_data) > 0:
                    if valid_data.min() < min_val or valid_data.max() > max_val:
                        errors.append(
                            f"{feature}: values out of range [{min_val}, {max_val}]. "
                            f"Found [{valid_data.min()}, {valid_data.max()}]"
                        )

                    # Store stats
                    stats[feature] = {
                        'min': float(valid_data.min()),
                        'max': float(valid_data.max()),
                        'mean': float(valid_data.mean()),
                        'std': float(valid_data.std()),
                        'missing': int(missing_count)
                    }

        # Validate binary features (wilderness and soil)
        binary_features = [f'wilderness_area_{i}' for i in range(4)]
        binary_features += [f'soil_type_{i}' for i in range(40)]

        for feature in binary_features:
            if feature in data.columns:
                unique_values = data[feature].dropna().unique()
                if not set(unique_values).issubset({0, 1}):
                    errors.append(f"{feature}: must be binary (0 or 1)")

        # Check for duplicate rows
        duplicate_count = data.duplicated().sum()
        if duplicate_count > 0:
            warnings.append(f"Found {duplicate_count} duplicate rows")

        # Check data types
        for feature in self.feature_ranges.keys():
            if feature in data.columns:
                if not pd.api.types.is_numeric_dtype(data[feature]):
                    errors.append(f"{feature}: must be numeric")

        # Overall stats
        stats['n_samples'] = len(data)
        stats['n_features'] = len(data.columns)
        stats['missing_total'] = int(data.isna().sum().sum())
        stats['duplicates'] = int(duplicate_count)

        is_valid = len(errors) == 0

        return ValidationResult(is_valid, errors, warnings, stats)

    def validate_single(self, sample: Dict[str, Any]) -> ValidationResult:
        """
        Validate a single prediction sample

        Args:
            sample: Dictionary with feature values

        Returns:
            ValidationResult
        """
        df = pd.DataFrame([sample])
        return self.validate(df)

    def validate_batch(self, samples: List[Dict[str, Any]]) -> ValidationResult:
        """
        Validate a batch of prediction samples

        Args:
            samples: List of dictionaries with feature values

        Returns:
            ValidationResult
        """
        df = pd.DataFrame(samples)
        return self.validate(df)

    def get_data_profile(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate comprehensive data profile

        Args:
            data: DataFrame to profile

        Returns:
            Dictionary with profile information
        """
        profile = {
            'overview': {
                'n_samples': len(data),
                'n_features': len(data.columns),
                # MB
                'memory_usage': data.memory_usage(deep=True).sum() / 1024**2,
            },
            'features': {},
            'quality': {
                'missing_cells': int(data.isna().sum().sum()),
                'missing_percentage': float(data.isna().sum().sum() / (len(data) * len(data.columns)) * 100),
                'duplicate_rows': int(data.duplicated().sum()),
            }
        }

        # Feature-level profiles
        for col in data.columns:
            if pd.api.types.is_numeric_dtype(data[col]):
                profile['features'][col] = {
                    'type': 'numeric',
                    'min': float(data[col].min()),
                    'max': float(data[col].max()),
                    'mean': float(data[col].mean()),
                    'median': float(data[col].median()),
                    'std': float(data[col].std()),
                    'missing': int(data[col].isna().sum()),
                    'unique': int(data[col].nunique()),
                }
            else:
                profile['features'][col] = {
                    'type': 'categorical',
                    'unique': int(data[col].nunique()),
                    'missing': int(data[col].isna().sum()),
                    'top_values': data[col].value_counts().head(5).to_dict(),
                }

        return profile


def validate_training_data(data: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Quick validation for training data

    Args:
        data: Training DataFrame with target column

    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []

    # Check for target column
    target_col = None
    for col in ['Cover_Type', 'cover_type', 'target']:
        if col in data.columns:
            target_col = col
            break

    if target_col is None:
        errors.append("No target column found")
        return False, errors

    # Validate target values
    valid_targets = set(range(1, 8))
    unique_targets = set(data[target_col].unique())

    if not unique_targets.issubset(valid_targets):
        errors.append(
            f"Invalid target values: {unique_targets - valid_targets}")

    # Check class balance
    target_counts = data[target_col].value_counts()
    min_samples = target_counts.min()
    max_samples = target_counts.max()

    if max_samples / min_samples > 10:
        errors.append(
            f"Severe class imbalance detected: "
            f"max={max_samples}, min={min_samples}, ratio={max_samples/min_samples:.1f}"
        )

    # Validate features
    validator = ForestCoverDataValidator()
    feature_data = data.drop(columns=[target_col])
    result = validator.validate(feature_data)

    if not result.is_valid:
        errors.extend(result.errors)

    return len(errors) == 0, errors


def validate_model_input(data: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validate data for model prediction

    Args:
        data: DataFrame for prediction (no target column)

    Returns:
        Tuple of (is_valid, error_messages)
    """
    validator = ForestCoverDataValidator()
    result = validator.validate(data)

    return result.is_valid, result.errors
