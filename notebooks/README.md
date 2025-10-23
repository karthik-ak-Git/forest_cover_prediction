# Forest Cover Type Prediction - Complete Analysis

This directory contains comprehensive Jupyter notebooks for the Forest Cover Type Prediction project.

## 📓 Notebooks

### 1. `complete_forest_cover_analysis.ipynb`
**Comprehensive end-to-end analysis and modeling notebook**

This notebook provides a complete walkthrough of the entire machine learning pipeline:

#### Contents:

1. **Environment Setup** (Section 1)
   - Library imports and configuration
   - Version checks
   - Random seed initialization

2. **Data Loading & Exploration** (Section 2)
   - Dataset loading from CSV
   - Initial data inspection
   - Memory usage analysis
   - Missing values and duplicate checks

3. **Exploratory Data Analysis** (Section 3)
   - Target variable distribution analysis
   - Forest cover type breakdown
   - Class imbalance assessment
   - Feature type categorization
   - Numerical feature distributions
   - Wilderness area analysis
   - Soil type analysis
   - Correlation matrix visualization

4. **Data Preprocessing** (Section 4)
   - Feature-target separation
   - Stratified train/validation/test split (60/20/20)
   - StandardScaler for numerical features
   - Binary feature preservation

5. **Feature Engineering** (Section 5)
   - Distance-based features (Euclidean distance to hydrology)
   - Mean distance to amenities
   - Hillshade aggregations (mean, variance)
   - Elevation categories (high/low)
   - Slope categories (steep/flat)
   - Total: 8 new engineered features

6. **Model Development** (Section 6)
   - **Baseline**: Logistic Regression (~72% accuracy)
   - **Random Forest**: 300 trees, max_depth=25 (~95% accuracy)
   - **XGBoost**: Gradient boosting with early stopping (~96.5% accuracy)
   - **LightGBM**: Fast gradient boosting (~97% accuracy)
   - Feature importance analysis

7. **Model Evaluation & Comparison** (Section 7)
   - Accuracy comparison across models
   - Overfitting analysis
   - Confusion matrix visualization
   - Per-class accuracy breakdown
   - Classification reports

8. **Ensemble Methods** (Section 8)
   - Voting Classifier (soft voting)
   - Combining RF + XGBoost + LightGBM
   - Improved accuracy: ~97.5%

9. **Model Deployment** (Section 9)
   - Final test set evaluation
   - Confusion matrix on test data
   - Per-class test accuracy

10. **Conclusions & Future Work** (Section 10)
    - Key findings summary
    - Project achievements
    - Future improvements
    - Technologies used

### 2. `01_data_exploration.ipynb`
**Initial data exploration notebook** (existing)

Basic exploratory data analysis and visualization.

## 🚀 Getting Started

### Prerequisites

Make sure you have the required packages installed:

```bash
pip install -r ../requirements.txt
```

### Running the Notebooks

1. **Start Jupyter Notebook/Lab**:
   ```bash
   jupyter notebook
   # or
   jupyter lab
   ```

2. **Navigate to the notebooks directory**:
   ```
   cd notebooks
   ```

3. **Open the desired notebook**:
   - `complete_forest_cover_analysis.ipynb` for full analysis
   - `01_data_exploration.ipynb` for basic exploration

### Execution Order

For best results, execute the notebooks in this order:

1. ✅ `01_data_exploration.ipynb` - Understand the data
2. ✅ `complete_forest_cover_analysis.ipynb` - Complete pipeline

## 📊 Key Results

### Model Performance Summary

| Model | Validation Accuracy | Test Accuracy |
|-------|-------------------|---------------|
| Logistic Regression | 72% | - |
| Random Forest | 95% | - |
| XGBoost | 96.5% | - |
| LightGBM | 97% | - |
| **Voting Ensemble** | **97.5%** | **97.5%+** |

### Top Important Features

1. **Elevation** - Most significant predictor
2. **Wilderness_Area_4** - Strong indicator
3. **Soil_Type_10** - High importance
4. **Horizontal_Distance_To_Roadways**
5. **Wilderness_Area_3**
6. **Distance_To_Hydrology** (engineered)
7. **Soil_Type_38**
8. **Hillshade features**

## 🔧 Feature Engineering Details

The notebook includes custom feature engineering:

```python
def create_engineered_features(X_df):
    # Distance calculations
    Distance_To_Hydrology = sqrt(Horizontal² + Vertical²)
    Mean_Distance_To_Amenities = mean(hydrology, roadways, fire_points)
    
    # Hillshade features
    Mean_Hillshade = mean(9am, noon, 3pm)
    Hillshade_Variance = variance across day
    
    # Categories
    Elevation_High = Elevation > 3000
    Elevation_Low = Elevation < 2500
    Slope_Steep = Slope > 20
    Slope_Flat = Slope < 5
```

## 📈 Visualizations Included

The notebook contains comprehensive visualizations:

- 📊 Target distribution (bar chart & pie chart)
- 📉 Numerical feature distributions (histograms)
- 📦 Box plots by cover type
- 🔥 Correlation heatmaps
- 🌲 Wilderness area distributions
- 🏔️ Soil type distributions
- 📊 Model performance comparisons
- 🎯 Confusion matrices
- ⭐ Feature importance plots

## 💡 Best Practices Demonstrated

1. **Stratified Splitting**: Maintains class distribution across splits
2. **Feature Scaling**: StandardScaler for numerical features only
3. **Early Stopping**: Prevents overfitting in gradient boosting
4. **Cross-Validation**: Ensures model robustness
5. **Ensemble Methods**: Combines multiple models for better performance
6. **Error Analysis**: Identifies misclassification patterns
7. **Documentation**: Clear markdown cells explaining each step

## 🎯 Learning Outcomes

By working through these notebooks, you will learn:

- ✅ Complete ML pipeline from data to deployment
- ✅ Effective EDA techniques
- ✅ Feature engineering strategies
- ✅ Multiple ML algorithms implementation
- ✅ Model evaluation and comparison
- ✅ Ensemble method creation
- ✅ Visualization best practices
- ✅ Code organization and documentation

## 📝 Notes

### Data Requirements

- The notebooks expect `train.csv` in the parent directory
- Ensure ~1GB of available RAM for processing
- GPU optional for neural networks (PyTorch)

### Execution Time

- Full notebook execution: ~10-30 minutes (depending on hardware)
- Can skip neural network sections if PyTorch not available
- Early stopping reduces training time significantly

### Customization

Feel free to modify:

- Hyperparameters in model sections
- Train/val/test split ratios
- Feature engineering functions
- Visualization styles
- Model selection for ensemble

## 🔗 Related Files

- `../train.csv` - Training dataset
- `../train_models.py` - Production training script
- `../fastapi_main.py` - API server
- `../frontend/` - Web application

## 📚 References

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

## 🤝 Contributing

To add new notebooks or improve existing ones:

1. Follow the existing structure
2. Add clear markdown documentation
3. Include visualizations
4. Test all cells before committing
5. Update this README

## 📧 Support

For questions or issues:
- Open an issue on GitHub
- Check the main project README
- Review the notebook documentation

---

**Last Updated**: October 2025  
**Author**: Karthik A K  
**Project**: Forest Cover Type Prediction
