"""
Feature Engineering for Snowfall Prediction
Transforms raw SNOTEL data into ML-ready features
Removes features with >10% missing data
"""
import pandas as pd
import numpy as np

def load_and_clean_data(filepath='nwcc_snow_data.csv'):
    """Load data and remove high-missing features"""
    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'])

    # Calculate missing percentages
    missing_pct = (df.isnull().sum() / len(df) * 100)

    # Identify features to drop (>10% missing)
    features_to_drop = missing_pct[missing_pct > 10].index.tolist()

    print("=" * 80)
    print("DATA CLEANING")
    print("=" * 80)
    print(f"Features removed (>10% missing): {features_to_drop}")
    print(f"Missing percentages:\n{missing_pct[missing_pct > 0]}\n")

    # Drop high-missing features
    df = df.drop(columns=features_to_drop)

    return df

def create_temporal_features(df):
    """Create date-based features for seasonality"""
    print("Creating temporal features...")

    df['day_of_year'] = df['Date'].dt.dayofyear
    df['month'] = df['Date'].dt.month
    df['week_of_year'] = df['Date'].dt.isocalendar().week
    df['year'] = df['Date'].dt.year

    # Season (meteorological: Dec-Feb=Winter, Mar-May=Spring, etc.)
    df['season'] = df['month'].map({
        12: 'Winter', 1: 'Winter', 2: 'Winter',
        3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Fall', 10: 'Fall', 11: 'Fall'
    })

    # Peak ski season (Dec-Feb)
    df['is_peak_season'] = df['month'].isin([12, 1, 2]).astype(int)

    # Days since October 1 (start of ski season, usually start from november)
    df['days_since_oct1'] = df.apply(
        lambda row: (row['Date'] - pd.Timestamp(f"{row['year'] if row['month'] >= 10 else row['year']-1}-10-01")).days,
        axis=1
    )

    return df

def create_lagged_features(df):
    """Create lagged features by resort (to avoid cross-contamination)"""
    print("Creating lagged features (1, 2, 3, 7 days)...")

    # Sort by resort and date to ensure proper lagging
    df = df.sort_values(['Resort', 'Date']).reset_index(drop=True)

    lag_periods = [1, 2, 3, 7]
    lag_features = {
        'AvgTemp_F': 'AvgTemp',
        'MinTemp_F': 'MinTemp',
        'MaxTemp_F': 'MaxTemp',
        'SnowDepth_in': 'SnowDepth',
        'Precip_Liquid_in': 'Precip',
        'NewSnow_in': 'NewSnow',
        'SWE_in': 'SWE'
    }

    for feature, short_name in lag_features.items():
        for lag in lag_periods:
            df[f'{short_name}_lag{lag}'] = df.groupby('Resort')[feature].shift(lag)

    return df

def create_rolling_features(df):
    """Create rolling window statistics"""
    print("Creating rolling window features (3-day, 7-day windows)...")

    # Sort by resort and date
    df = df.sort_values(['Resort', 'Date']).reset_index(drop=True)

    rolling_configs = [
        # (column, window_size, agg_function, output_name)
        # IMPORTANT: Use shift(1) to exclude current observation (prevent leakage)
        ('AvgTemp_F', 3, 'mean', 'AvgTemp_rolling_mean_3d'),
        ('AvgTemp_F', 7, 'mean', 'AvgTemp_rolling_mean_7d'),
        ('AvgTemp_F', 7, 'std', 'AvgTemp_rolling_std_7d'),
        ('AvgTemp_F', 7, 'max', 'AvgTemp_rolling_max_7d'),  # Did it thaw last week?
        ('AvgTemp_F', 7, 'min', 'AvgTemp_rolling_min_7d'),  # How cold did it stay?
        ('MinTemp_F', 3, 'mean', 'MinTemp_rolling_mean_3d'),
        ('MinTemp_F', 7, 'min', 'MinTemp_rolling_min_7d'),  # Coldest point last week
        ('Precip_Liquid_in', 3, 'sum', 'Precip_rolling_sum_3d'),
        ('Precip_Liquid_in', 7, 'sum', 'Precip_rolling_sum_7d'),
        ('NewSnow_in', 7, 'sum', 'NewSnow_rolling_sum_7d'),
        ('NewSnow_in', 7, 'max', 'NewSnow_rolling_max_7d'),
        ('SnowDepth_in', 7, 'mean', 'SnowDepth_rolling_mean_7d'),
    ]

    for col, window, agg, output_name in rolling_configs:
        # Use shift(1) to look only at past values (exclude current day)
        df[output_name] = df.groupby('Resort')[col].transform(
            lambda x: x.rolling(window=window, min_periods=1).agg(agg).shift(1)
        )

    # Snow depth change over 7 days (using past values only)
    df['SnowDepth_change_7d'] = df.groupby('Resort')['SnowDepth_in'].transform(
        lambda x: x.shift(1) - x.shift(8)
    )

    return df

def create_derived_features(df):
    """Create thermodynamic and derived features"""
    print("Creating derived thermodynamic features...")

    # Temperature indicators
    df['temp_range'] = df['MaxTemp_F'] - df['MinTemp_F']
    df['freezing_level'] = df['AvgTemp_F'] - 32.0
    df['is_freezing'] = (df['AvgTemp_F'] <= 32.0).astype(int)
    df['cold_snap'] = (df['AvgTemp_F'] < 20.0).astype(int)

    # REMOVED: snow_to_liquid_ratio (LEAKAGE - uses target variable!)
    # REMOVED: swe_to_depth_ratio (LEAKAGE - uses current day's SnowDepth)

    # Instead, use yesterday's snow density as a feature
    df = df.sort_values(['Resort', 'Date']).reset_index(drop=True)
    df['SnowDepth_lag1_for_ratio'] = df.groupby('Resort')['SnowDepth_in'].shift(1)
    df['SWE_lag1_for_ratio'] = df.groupby('Resort')['SWE_in'].shift(1)
    df['swe_to_depth_ratio_lag1'] = np.where(
        df['SnowDepth_lag1_for_ratio'] > 0,
        df['SWE_lag1_for_ratio'] / df['SnowDepth_lag1_for_ratio'],
        0
    )
    df = df.drop(columns=['SnowDepth_lag1_for_ratio', 'SWE_lag1_for_ratio'])

    # Storm indicators
    df['is_precip_day'] = (df['Precip_Liquid_in'] > 0).astype(int)
    df['precip_with_freezing'] = ((df['Precip_Liquid_in'] > 0) & (df['AvgTemp_F'] <= 32)).astype(int)
    df['heavy_precip'] = (df['Precip_Liquid_in'] > 0.5).astype(int)

    # Thaw indicators (important for snow melt/preservation)
    # Note: These use rolling features which are already shifted, so no additional shift needed
    df['thaw_last_week'] = (df['AvgTemp_rolling_max_7d'] > 32).astype(int)  # Did it thaw at all?
    df['freeze_thaw_cycles'] = ((df['AvgTemp_rolling_min_7d'] < 32) &
                                 (df['AvgTemp_rolling_max_7d'] > 32)).astype(int)  # Both freeze and thaw

    return df

def create_change_features(df):
    """Create multi-day change/momentum features"""
    print("Creating change and momentum features...")

    df = df.sort_values(['Resort', 'Date']).reset_index(drop=True)

    # Temperature changes (yesterday to day before)
    # These show temperature trends leading up to today
    df['temp_change_1d'] = df.groupby('Resort')['AvgTemp_F'].diff(1)
    df['temp_change_3d'] = df.groupby('Resort')['AvgTemp_F'].diff(3)

    # Temperature acceleration (change in change)
    df['temp_acceleration'] = df.groupby('Resort')['temp_change_1d'].diff(1)

    # Precipitation trend
    df['precip_change_1d'] = df.groupby('Resort')['Precip_Liquid_in'].diff(1)

    # REMOVED: swe_change_1d and swe_change_3d (LEAKAGE!)
    # SWE change IS the snowfall's water content - using it reveals the answer
    # Instead, use lagged SWE changes
    df['swe_change_lag1_to_lag2'] = df.groupby('Resort')['SWE_in'].transform(
        lambda x: x.shift(1) - x.shift(2)
    )
    df['swe_change_lag1_to_lag4'] = df.groupby('Resort')['SWE_in'].transform(
        lambda x: x.shift(1) - x.shift(4)
    )

    return df

def create_interaction_features(df):
    """Create interaction features"""
    print("Creating interaction features...")

    # Temperature × Precipitation
    df['temp_precip_interaction'] = df['AvgTemp_F'] * df['Precip_Liquid_in']

    # Freezing precipitation (only when below freezing)
    df['freezing_precip'] = np.where(
        df['AvgTemp_F'] < 32,
        (32 - df['AvgTemp_F']) * df['Precip_Liquid_in'],
        0
    )

    # Temporal interactions
    df['month_temp'] = df['month'] * df['AvgTemp_F']

    # Season × precipitation (encoded numerically)
    season_encoding = {'Winter': 0, 'Spring': 1, 'Summer': 2, 'Fall': 3}
    df['season_encoded'] = df['season'].map(season_encoding)
    df['season_precip'] = df['season_encoded'] * df['Precip_Liquid_in']

    return df

def create_resort_features(df):
    """Create resort-specific features (one-hot encoding)"""
    print("Creating resort features...")

    # One-hot encode resorts
    resort_dummies = pd.get_dummies(df['Resort'], prefix='resort')
    df = pd.concat([df, resort_dummies], axis=1)

    return df

def engineer_features(input_file='nwcc_snow_data.csv', output_file='nwcc_snow_data_engineered.csv'):
    """Main feature engineering pipeline"""

    print("\n" + "=" * 80)
    print("SNOWFALL PREDICTION - FEATURE ENGINEERING PIPELINE")
    print("=" * 80 + "\n")

    # Load and clean
    df = load_and_clean_data(input_file)
    initial_rows = len(df)
    initial_cols = len(df.columns)

    # Apply transformations
    df = create_temporal_features(df)
    df = create_lagged_features(df)
    df = create_rolling_features(df)
    df = create_derived_features(df)
    df = create_change_features(df)
    df = create_interaction_features(df)
    df = create_resort_features(df)

    # Handle NaN values from lagging/rolling (drop rows with NaN in lag features)
    # Keep first 7 days for reference but they won't be used in training
    print("\n" + "=" * 80)
    print("HANDLING MISSING VALUES FROM LAGS/ROLLING")
    print("=" * 80)
    print(f"Rows before NaN handling: {len(df)}")

    # Count NaN by column
    nan_counts = df.isnull().sum()
    print(f"\nColumns with NaN values (top 10):")
    print(nan_counts[nan_counts > 0].sort_values(ascending=False).head(10))

    # Drop rows where lagged features are NaN (first 7 days per resort)
    # This is expected and necessary
    df_clean = df.dropna()

    print(f"\nRows after dropping NaN: {len(df_clean)}")
    print(f"Rows dropped: {len(df) - len(df_clean)} ({(len(df) - len(df_clean)) / len(df) * 100:.2f}%)")

    # Save engineered dataset
    try:
        df_clean.to_csv(output_file, index=False)
    except PermissionError:
        output_file = 'nwcc_snow_data_engineered_fixed.csv'
        print(f"\nPermission denied. Saving to {output_file} instead...")
        df_clean.to_csv(output_file, index=False)

    # Summary
    print("\n" + "=" * 80)
    print("FEATURE ENGINEERING SUMMARY")
    print("=" * 80)
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print(f"Initial shape: {initial_rows} rows × {initial_cols} columns")
    print(f"Final shape: {len(df_clean)} rows × {len(df_clean.columns)} columns")
    print(f"Features added: {len(df_clean.columns) - initial_cols}")
    print(f"\nTarget variable: NewSnow_in")
    print(f"Features ready for ML modeling!")

    # Show sample of new features
    print("\n" + "=" * 80)
    print("SAMPLE OF ENGINEERED FEATURES (first 5 rows)")
    print("=" * 80)
    feature_cols = [col for col in df_clean.columns if col not in
                    ['Date', 'SWE_in', 'SnowDepth_in', 'PrecipAccum_in',
                     'MaxTemp_F', 'MinTemp_F', 'AvgTemp_F', 'SoilTemp_2in_F',
                     'SoilMoist_2in_pct', 'NewSnow_in', 'Precip_Liquid_in', 'Resort']]
    print(f"\nNew engineered features ({len(feature_cols)} total):")
    print(df_clean[feature_cols].head())

    return df_clean

if __name__ == "__main__":
    # Run feature engineering
    df_engineered = engineer_features()

    print("\n" + "=" * 80)
    print("Feature engineering complete!")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Split data into train/validation/test sets (time-series split)")
    print("2. Train baseline models (Random Forest, XGBoost)")
    print("3. Evaluate feature importance")
    print("4. Tune hyperparameters")
