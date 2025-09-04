"""
Forest Cover Type Prediction - Configuration File
This file contains all the configuration settings for the project.
"""

import torch
import os

# Project Paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')

# Data Files
TRAIN_DATA_PATH = os.path.join(PROJECT_ROOT, 'train.csv')

# Model Configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.2

# Target Accuracy
TARGET_ACCURACY = 0.99

# PyTorch Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 512
LEARNING_RATE = 0.001
EPOCHS = 200
PATIENCE = 20

# Model Parameters
NEURAL_NET_PARAMS = {
    'hidden_layers': [512, 256, 128, 64],
    'dropout_rate': 0.3,
    'activation': 'relu'
}

XGBOOST_PARAMS = {
    'max_depth': 10,
    'learning_rate': 0.1,
    'n_estimators': 1000,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': RANDOM_STATE
}

RANDOM_FOREST_PARAMS = {
    'n_estimators': 500,
    'max_depth': 20,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'random_state': RANDOM_STATE,
    'n_jobs': -1
}

# Cross-Validation
CV_FOLDS = 5

# Feature Engineering
FEATURE_SELECTION_K = 'all'  # Number of top features to select

print(f"Configuration loaded successfully!")
print(f"Device: {DEVICE}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
