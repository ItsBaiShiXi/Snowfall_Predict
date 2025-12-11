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
**Purpose**: Train Random Forest and XGBoost models, compare against baseline
**What it does**:
- Trains Random Forest (100 trees, max_depth=20) and XGBoost (100 estimators) regressors
- Compares all three models on same train/test split
- **Best model**: XGBoost achieves R² = 0.688, RMSE = 1.41 inches, MAE = 0.39 inches
- Identifies top predictive features (yesterday's precipitation is #1 at 25.8% importance)
- Evaluates per-resort performance (Mammoth: 0.768, Heavenly: 0.759, Palisades: 0.560)
- Generates comparison visualization saved to `model_comparison.png`

**Run**: `python train_advanced_models.py`

### Data Files

#### [nwcc_snow_data.csv](nwcc_snow_data.csv)
Raw SNOTEL data fetched from USDA API. Contains ~22,000 daily observations with 14 features including snow depth, temperature, precipitation, SWE, and engineered NewSnow_in target variable.

#### [nwcc_snow_data_engineered_fixed.csv](nwcc_snow_data_engineered_fixed.csv)
ML-ready dataset with 68 engineered features. Fixed to prevent data leakage. Used for all model training.

### Documentation

#### [MODEL_COMPARISON_SUMMARY.md](MODEL_COMPARISON_SUMMARY.md)
Summary of model performance comparison between Linear Regression, Random Forest, and XGBoost. Includes metrics, feature importance analysis, and recommendations.

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

| Model | Test R² | RMSE (inches) | MAE (inches) |
|-------|---------|---------------|--------------|
| Linear Regression | 0.559 | 1.67 | 0.57 |
| Random Forest | 0.649 | 1.49 | 0.43 |
| **XGBoost** | **0.688** | **1.41** | **0.39** |

**Top Predictive Features** (XGBoost):
1. Precip_lag1 (yesterday's precipitation) - 25.8%
2. PrecipAccum_rolling_7d_sum - 8.5%
3. Precip_Liquid_in - 6.3%
4. SnowDepth_in - 5.9%
5. NewSnow_lag1 - 4.7%
