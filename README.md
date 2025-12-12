# Localized Snowfall Prediction in the Sierra Nevada using SNOTEL Telemetry

## Abstract
General weather forecasts often fail to capture the microclimates of high-elevation ski resorts. This project aims to predict daily new snowfall at three major Sierra Nevada resorts (Palisades Tahoe, Heavenly, and Mammoth) by training machine learning models on 20 years of historical SNOTEL sensor data. We will utilize a dataset of over 20,000 daily records, engineering features such as lagged temperature, snow density, and rolling precipitation trends. We expect to train Random Forest and XGBoost regressors that outperform baseline persistence models, while also using feature importance analysis to quantify the impact of local thermodynamics on powder days.

## Project Structure

### Core Scripts

#### [data_extraction.py](data_extraction.py)
**Purpose**: Fetch raw SNOTEL data from USDA NRCS API
**What it does**:
- Downloads 20 years of historical weather/snow data (2005-present) for three Sierra Nevada ski resorts
- Fetches 12 different measurements per station: snow depth, SWE, temperature, precipitation, humidity, wind, and soil metrics
- Engineers two key features: `NewSnow_in` (daily snow depth increase) and `Precip_Liquid_in` (daily precipitation increase)
- Outputs consolidated data to `nwcc_snow_data.csv`

**Run**: `python data_extraction.py`

#### [data_inspection.py](data_inspection.py)
**Purpose**: Exploratory data analysis and quality checks
**What it does**:
- Analyzes data patterns, distributions, and temporal trends
- Identifies missing values and data quality issues across all features
- Provides statistical summaries by resort and time period
- Helps inform feature engineering decisions

**Run**: `python data_inspection.py`

#### [feature_engineering.py](feature_engineering.py)
**Purpose**: Transform raw SNOTEL data into ML-ready features
**What it does**:
- Removes features with >10% missing data (RelHumidity, WindSpeed, WindDir, SoilMoisture)
- Creates 68 engineered features across 6 categories:
  - **Temporal** (10): day_of_year, month, season, is_peak_season, days_since_oct1
  - **Lagged** (28): 1, 2, 3, 7-day historical values for temp, snow, precip, SWE
  - **Rolling Windows** (12): 3-day and 7-day moving statistics
  - **Derived** (14): temp_range, freezing_level, thaw cycles, storm indicators
  - **Change/Momentum** (7): Temperature trends, precipitation momentum
  - **Interaction** (4): Temperature × precipitation interactions
- Implements proper data leakage prevention (rolling windows exclude current day)
- Outputs `nwcc_snow_data_engineered_fixed.csv` (22,148 rows × 83 columns)

**Run**: `python feature_engineering.py`

#### [train_baseline.py](train_baseline.py)
**Purpose**: Train and evaluate Linear Regression baseline model
**What it does**:
- Loads engineered features dataset
- Performs time-series train/test split (train: pre-2023, test: 2023-2025)
- Trains Linear Regression model with feature standardization
- Evaluates performance: R² = 0.559, RMSE = 1.67 inches, MAE = 0.57 inches
- Compares against persistence baseline (predicting yesterday's snowfall)
- Generates per-resort performance breakdown

**Run**: `python train_baseline.py`

#### [train_advanced_models.py](train_advanced_models.py)
**Purpose**: Train Random Forest and XGBoost models with optimized hyperparameters, compare against baseline
**What it does**:
- Uses best hyperparameters from extensive RandomizedSearchCV tuning
- Random Forest: Optimized with n_estimators=500, max_depth=25, min_samples_leaf=8, max_features=0.5
- XGBoost: Uses heavily regularized configuration (max_depth=8, min_child_weight=20, learning_rate=0.005, gamma=2.0)
- Performs 3-fold cross-validation during hyperparameter search (CV R² = 0.712 for XGBoost)
- **Best model**: XGBoost (Tuned) achieves Test R² = 0.704, RMSE = 1.37 inches, MAE = 0.38 inches
- **Successfully prevented overfitting**: XGBoost Train R² = 0.869 with strong regularization (gamma=2.0, min_child_weight=20)
- Identifies top predictive features (yesterday's precipitation is #1 at 16.1% importance)
- Evaluates per-resort performance (Mammoth: 0.810, Heavenly: 0.751, Palisades: 0.573)
- Generates comparison visualization saved to `model_comparison.png`

**Run**: `python train_advanced_models.py`

### Data Files

#### [nwcc_snow_data.csv](nwcc_snow_data.csv)
Raw SNOTEL data fetched from USDA API. Contains ~22,000 daily observations with 14 features including snow depth, temperature, precipitation, SWE, and engineered NewSnow_in target variable.

#### [nwcc_snow_data_engineered_fixed.csv](nwcc_snow_data_engineered_fixed.csv)
ML-ready dataset with 68 engineered features. Fixed to prevent data leakage. Used for all model training.

### Documentation

#### [requirements.txt](requirements.txt)
Python package dependencies. Install with: `pip install -r requirements.txt`

#### [model_comparison.png](model_comparison.png)
Visualization comparing R² scores across all three models and resorts.

### Archive Files

#### archive/
Contains deprecated files:
- `check_leakage.py`: Script used to identify data leakage issues (now fixed)
- `nwcc_snow_data_engineered.csv`: Original engineered dataset with leakage (replaced by _fixed version)
- `Snow Water Equivalent, November 19, 2025, end of day (1).csv`: Example SNOTEL snapshot data
- `LEAKAGE_FIX_SUMMARY.md`: Documentation of data leakage fixes applied
- `MODEL_COMPARISON_SUMMARY.md`: Previous model comparison results (before latest hyperparameter tuning)

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Fetch fresh SNOTEL data (optional - data already included)
python data_extraction.py

# 3. Run data inspection (optional)
python data_inspection.py

# 4. Generate engineered features
python feature_engineering.py

# 5. Train baseline model
python train_baseline.py

# 6. Train advanced models and see comparison
python train_advanced_models.py
```

## Results Summary

| Model | Train R² | Test R² | RMSE (inches) | MAE (inches) | Improvement vs Baseline |
|-------|----------|---------|---------------|--------------|------------------------|
| Linear Regression | 0.530 | 0.559 | 1.67 | 0.57 | - |
| Random Forest (Tuned) | 0.819 | 0.687 | 1.41 | 0.38 | +22.9% |
| **XGBoost (Tuned)** | **0.869** | **0.704** | **1.37** | **0.38** | **+26.0%** |

**Key Achievement**: Aggressive hyperparameter tuning with strong regularization (gamma=2.0, min_child_weight=20, very low learning_rate=0.005) successfully prevented overfitting. XGBoost achieves excellent generalization with Train R² = 0.869, Test R² = 0.704, and CV R² = 0.712.

**Best Hyperparameters (XGBoost)**:
- `max_depth=8, min_child_weight=20, learning_rate=0.005, gamma=2.0`
- `subsample=0.6, colsample_bytree=0.8, n_estimators=1000`

**Top Predictive Features** (XGBoost Tuned):
1. Precip_lag1 (yesterday's precipitation) - 16.1%
2. precip_change_1d (daily precipitation change) - 8.5%
3. AvgTemp_lag1 (yesterday's avg temperature) - 4.0%
4. MaxTemp_lag1 (yesterday's max temperature) - 3.5%
5. freezing_level (current freezing conditions) - 3.2%

**Performance by Resort** (XGBoost Tuned):
- Mammoth Mountain: R² = 0.810 (best)
- Heavenly: R² = 0.751
- Palisades Tahoe: R² = 0.573 (most challenging terrain)
