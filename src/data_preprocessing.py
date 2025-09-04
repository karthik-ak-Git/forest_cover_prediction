"""
Forest Cover Type Prediction - Data Preprocessing Module
This module handles all data preprocessing tasks including:
- Feature engineering
- Data scaling and normalization
- Train/validation/test splits
- Feature selection
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import config


class ForestCoverDataset(Dataset):
    """Custom PyTorch Dataset for Forest Cover data"""

    def __init__(self, features, targets=None):
        self.features = torch.FloatTensor(features)
        self.targets = torch.LongTensor(
            targets) if targets is not None else None

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if self.targets is not None:
            # Convert to 0-based indexing
            return self.features[idx], self.targets[idx] - 1
        return self.features[idx]


class DataPreprocessor:
    """Complete data preprocessing pipeline"""

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scaler = None
        self.feature_selector = None
        self.selected_features = None
        self.feature_names = None

    def load_data(self, file_path):
        """Load and return the dataset"""
        print(f"Loading data from {file_path}...")
        df = pd.read_csv(file_path)
        print(f"Dataset shape: {df.shape}")
        return df

    def engineer_features(self, df):
        """Create new features from existing ones"""
        print("Engineering new features...")
        df_processed = df.copy()

        # Create interaction features
        df_processed['Elevation_x_Slope'] = df_processed['Elevation'] * \
            df_processed['Slope']
        df_processed['Hydrology_Distance_Ratio'] = (
            df_processed['Horizontal_Distance_To_Hydrology'] /
            (df_processed['Vertical_Distance_To_Hydrology'].abs() + 1)
        )

        # Create distance-based features
        df_processed['Total_Distance_To_Infrastructure'] = (
            df_processed['Horizontal_Distance_To_Roadways'] +
            df_processed['Horizontal_Distance_To_Fire_Points']
        )

        # Hillshade features
        df_processed['Hillshade_Variance'] = df_processed[[
            'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm']].var(axis=1)
        df_processed['Hillshade_Mean'] = df_processed[[
            'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm']].mean(axis=1)

        # Aspect-based features (circular feature)
        df_processed['Aspect_Sin'] = np.sin(np.radians(df_processed['Aspect']))
        df_processed['Aspect_Cos'] = np.cos(np.radians(df_processed['Aspect']))

        # Remove original aspect as it's circular
        df_processed = df_processed.drop('Aspect', axis=1)

        print(
            f"New dataset shape after feature engineering: {df_processed.shape}")
        return df_processed

    def prepare_features_and_target(self, df):
        """Separate features and target variable"""
        # Remove Id column if present
        if 'Id' in df.columns:
            df = df.drop('Id', axis=1)

        X = df.drop('Cover_Type', axis=1)
        y = df['Cover_Type']

        self.feature_names = X.columns.tolist()
        print(f"Number of features: {len(self.feature_names)}")
        print(f"Target distribution:\\n{y.value_counts().sort_index()}")

        return X.values, y.values\n    \n def split_data(self, X, y, test_size=0.2, val_size=0.2): \n        \"\"\"Split data into train, validation, and test sets\"\"\"\n print(f\"Splitting data: train={1-test_size-val_size: .1f}, val={val_size: .1f}, test={test_size: .1f}\")\n        \n        # First split: separate test set\n        X_temp, X_test, y_temp, y_test = train_test_split(\n            X, y, test_size=test_size, stratify=y, random_state=self.random_state\n        )\n        \n        # Second split: separate train and validation\n        val_size_adjusted = val_size / (1 - test_size)\n        X_train, X_val, y_train, y_val = train_test_split(\n            X_temp, y_temp, test_size=val_size_adjusted, \n            stratify=y_temp, random_state=self.random_state\n        )\n        \n        print(f\"Train set: {X_train.shape[0]} samples\")\n        print(f\"Validation set: {X_val.shape[0]} samples\")\n        print(f\"Test set: {X_test.shape[0]} samples\")\n        \n        return X_train, X_val, X_test, y_train, y_val, y_test\n    \n    def scale_features(self, X_train, X_val, X_test, scaler_type='standard'):\n        \"\"\"Scale features using the specified scaler\"\"\"\n        print(f\"Scaling features using {scaler_type} scaler...\")\n        \n        if scaler_type == 'standard':\n            self.scaler = StandardScaler()\n        elif scaler_type == 'robust':\n            self.scaler = RobustScaler()\n        elif scaler_type == 'minmax':\n            self.scaler = MinMaxScaler()\n        else:\n            raise ValueError(f\"Unknown scaler type: {scaler_type}\")\n        \n        X_train_scaled = self.scaler.fit_transform(X_train)\n        X_val_scaled = self.scaler.transform(X_val)\n        X_test_scaled = self.scaler.transform(X_test)\n        \n        return X_train_scaled, X_val_scaled, X_test_scaled\n    \n    def select_features(self, X_train, y_train, X_val, X_test, method='f_classif', k=50):\n        \"\"\"Feature selection using various methods\"\"\"\n        if k == 'all' or k >= X_train.shape[1]:\n            print(\"Using all features (no feature selection)\")\n            return X_train, X_val, X_test\n        \n        print(f\"Selecting top {k} features using {method}...\")\n        \n        if method == 'f_classif':\n            score_func = f_classif\n        elif method == 'mutual_info':\n            score_func = mutual_info_classif\n        elif method == 'random_forest':\n            # Use Random Forest feature importance\n            rf = RandomForestClassifier(n_estimators=100, random_state=self.random_state)\n            rf.fit(X_train, y_train)\n            importances = rf.feature_importances_\n            indices = np.argsort(importances)[::-1][:k]\n            \n            self.selected_features = indices\n            selected_feature_names = [self.feature_names[i] for i in indices]\n            print(f\"Selected features: {selected_feature_names[:10]}...\")  # Show first 10\n            \n            return X_train[:, indices], X_val[:, indices], X_test[:, indices]\n        \n        else:\n            raise ValueError(f\"Unknown feature selection method: {method}\")\n        \n        self.feature_selector = SelectKBest(score_func=score_func, k=k)\n        X_train_selected = self.feature_selector.fit_transform(X_train, y_train)\n        X_val_selected = self.feature_selector.transform(X_val)\n        X_test_selected = self.feature_selector.transform(X_test)\n        \n        # Get selected feature names\n        selected_indices = self.feature_selector.get_support(indices=True)\n        selected_feature_names = [self.feature_names[i] for i in selected_indices]\n        print(f\"Selected features: {selected_feature_names[:10]}...\")  # Show first 10\n        \n        return X_train_selected, X_val_selected, X_test_selected\n    \n    def create_pytorch_datasets(self, X_train, y_train, X_val, y_val, X_test, y_test):\n        \"\"\"Create PyTorch datasets and dataloaders\"\"\"\n        print(\"Creating PyTorch datasets...\")\n        \n        train_dataset = ForestCoverDataset(X_train, y_train)\n        val_dataset = ForestCoverDataset(X_val, y_val)\n        test_dataset = ForestCoverDataset(X_test, y_test)\n        \n        train_loader = DataLoader(\n            train_dataset, batch_size=config.BATCH_SIZE, \n            shuffle=True, num_workers=0, pin_memory=True\n        )\n        val_loader = DataLoader(\n            val_dataset, batch_size=config.BATCH_SIZE, \n            shuffle=False, num_workers=0, pin_memory=True\n        )\n        test_loader = DataLoader(\n            test_dataset, batch_size=config.BATCH_SIZE, \n            shuffle=False, num_workers=0, pin_memory=True\n        )\n        \n        print(f\"Train loader: {len(train_loader)} batches\")\n        print(f\"Validation loader: {len(val_loader)} batches\")\n        print(f\"Test loader: {len(test_loader)} batches\")\n        \n        return train_loader, val_loader, test_loader\n    \n    def get_cross_validation_folds(self, X, y, n_folds=5):\n        \"\"\"Generate cross-validation folds\"\"\"\n        print(f\"Creating {n_folds}-fold cross-validation splits...\")\n        \n        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)\n        folds = list(skf.split(X, y))\n        \n        print(f\"Created {len(folds)} folds for cross-validation\")\n        return folds\n    \n    def preprocess_pipeline(self, file_path, scaler_type='standard', \n                          feature_selection_method='f_classif', k_features='all'):\n        \"\"\"Complete preprocessing pipeline\"\"\"\n        print(\"=\" * 60)\n        print(\"STARTING DATA PREPROCESSING PIPELINE\")\n        print(\"=\" * 60)\n        \n        # 1. Load data\n        df = self.load_data(file_path)\n        \n        # 2. Feature engineering\n        df_engineered = self.engineer_features(df)\n        \n        # 3. Prepare features and target\n        X, y = self.prepare_features_and_target(df_engineered)\n        \n        # 4. Split data\n        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(\n            X, y, test_size=config.TEST_SIZE, val_size=config.VAL_SIZE\n        )\n        \n        # 5. Scale features\n        X_train_scaled, X_val_scaled, X_test_scaled = self.scale_features(\n            X_train, X_val, X_test, scaler_type=scaler_type\n        )\n        \n        # 6. Feature selection\n        X_train_final, X_val_final, X_test_final = self.select_features(\n            X_train_scaled, y_train, X_val_scaled, X_test_scaled,\n            method=feature_selection_method, k=k_features\n        )\n        \n        # 7. Create PyTorch datasets\n        train_loader, val_loader, test_loader = self.create_pytorch_datasets(\n            X_train_final, y_train, X_val_final, y_val, X_test_final, y_test\n        )\n        \n        # 8. Generate CV folds\n        cv_folds = self.get_cross_validation_folds(X_train_final, y_train, config.CV_FOLDS)\n        \n        print(\"=\" * 60)\n        print(\"PREPROCESSING PIPELINE COMPLETED SUCCESSFULLY!\")\n        print(\"=\" * 60)\n        \n        return {\n            'train_loader': train_loader,\n            'val_loader': val_loader,\n            'test_loader': test_loader,\n            'X_train': X_train_final,\n            'X_val': X_val_final,\n            'X_test': X_test_final,\n            'y_train': y_train,\n            'y_val': y_val,\n            'y_test': y_test,\n            'cv_folds': cv_folds,\n            'n_features': X_train_final.shape[1],\n            'n_classes': len(np.unique(y))\n        }\n    \n    def save_preprocessor(self, filepath):\n        \"\"\"Save the preprocessor state\"\"\"\n        import pickle\n        with open(filepath, 'wb') as f:\n            pickle.dump({\n                'scaler': self.scaler,\n                'feature_selector': self.feature_selector,\n                'selected_features': self.selected_features,\n                'feature_names': self.feature_names\n            }, f)\n        print(f\"Preprocessor saved to {filepath}\")\n    \n    def load_preprocessor(self, filepath):\n        \"\"\"Load the preprocessor state\"\"\"\n        import pickle\n        with open(filepath, 'rb') as f:\n            state = pickle.load(f)\n        \n        self.scaler = state['scaler']\n        self.feature_selector = state['feature_selector']\n        self.selected_features = state['selected_features']\n        self.feature_names = state['feature_names']\n        print(f\"Preprocessor loaded from {filepath}\")\n\nif __name__ == \"__main__\":\n    # Test the preprocessing pipeline\n    preprocessor = DataPreprocessor()\n    \n    # Run the complete pipeline\n    data = preprocessor.preprocess_pipeline(\n        file_path=config.TRAIN_DATA_PATH,\n        scaler_type='standard',\n        feature_selection_method='f_classif',\n        k_features='all'\n    )\n    \n    print(f\"\\nFinal data shapes:\")\n    print(f\"Training features: {data['X_train'].shape}\")\n    print(f\"Training targets: {data['y_train'].shape}\")\n    print(f\"Number of features: {data['n_features']}\")\n    print(f\"Number of classes: {data['n_classes']}\")\n    \n    # Save preprocessor\n    import os\n    preprocessor.save_preprocessor(os.path.join(config.MODELS_DIR, 'preprocessor.pkl'))
