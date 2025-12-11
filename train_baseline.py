"""
Baseline Model Training - Linear Regression
Time-series train/test split for snowfall prediction
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(filepath='nwcc_snow_data_engineered_fixed.csv'):
    """Load engineered dataset"""
    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def train_test_split_timeseries(df, test_start_date='2023-01-01'):
    """
    Time-series split: Train on data before test_start_date, test on data after
    This prevents data leakage from future to past
    """
    train = df[df['Date'] < test_start_date].copy()
    test = df[df['Date'] >= test_start_date].copy()

    print("=" * 80)
    print("TRAIN/TEST SPLIT (Time-Series)")
    print("=" * 80)
    print(f"Total samples: {len(df)}")
    print(f"Train samples: {len(train)} ({len(train)/len(df)*100:.1f}%)")
    print(f"  Date range: {train['Date'].min()} to {train['Date'].max()}")
    print(f"Test samples: {len(test)} ({len(test)/len(df)*100:.1f}%)")
    print(f"  Date range: {test['Date'].min()} to {test['Date'].max()}")

    return train, test

def prepare_features(train, test, target='NewSnow_in'):
    """
    Prepare features and target for modeling
    Exclude non-feature columns and handle scaling
    """
    # Columns to exclude from features
    exclude_cols = ['Date', 'Resort', 'season', 'NewSnow_in',
                    'SWE_in', 'SnowDepth_in', 'PrecipAccum_in',
                    'MaxTemp_F', 'MinTemp_F', 'AvgTemp_F',
                    'SoilTemp_2in_F', 'Precip_Liquid_in']

    feature_cols = [col for col in train.columns if col not in exclude_cols]

    print("\n" + "=" * 80)
    print("FEATURE PREPARATION")
    print("=" * 80)
    print(f"Total features for modeling: {len(feature_cols)}")
    print(f"Target variable: {target}")

    # Separate features and target
    X_train = train[feature_cols]
    y_train = train[target]
    X_test = test[feature_cols]
    y_test = test[target]

    print(f"\nFeature matrix shape:")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_test: {X_test.shape}")

    # Scale features (important for linear regression)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"\nFeatures scaled using StandardScaler")

    return X_train_scaled, X_test_scaled, y_train, y_test, feature_cols, scaler

def train_linear_regression(X_train, y_train):
    """Train Linear Regression model"""
    print("\n" + "=" * 80)
    print("TRAINING LINEAR REGRESSION")
    print("=" * 80)

    model = LinearRegression()
    model.fit(X_train, y_train)

    print("Model trained successfully!")
    print(f"Intercept: {model.intercept_:.4f}")
    print(f"Number of coefficients: {len(model.coef_)}")

    return model

def evaluate_model(model, X_train, X_test, y_train, y_test, dataset_name="Test"):
    """Evaluate model performance"""
    # Predictions
    y_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)

    # Clip negative predictions to 0 (can't have negative snowfall)
    y_pred_clipped = np.clip(y_pred, 0, None)

    # Metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_clipped))
    mae = mean_absolute_error(y_test, y_pred_clipped)
    r2 = r2_score(y_test, y_pred_clipped)

    # Train metrics
    rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
    mae_train = mean_absolute_error(y_train, y_train_pred)
    r2_train = r2_score(y_train, y_train_pred)

    print("\n" + "=" * 80)
    print(f"MODEL EVALUATION - {dataset_name}")
    print("=" * 80)
    print("\nTrain Set Performance:")
    print(f"  RMSE: {rmse_train:.4f} inches")
    print(f"  MAE:  {mae_train:.4f} inches")
    print(f"  R²:   {r2_train:.4f}")

    print(f"\n{dataset_name} Set Performance:")
    print(f"  RMSE: {rmse:.4f} inches")
    print(f"  MAE:  {mae:.4f} inches")
    print(f"  R²:   {r2:.4f}")

    # Baseline comparison: predict yesterday's snowfall
    print("\n--- Baseline Comparison ---")
    print("Persistence Model: Predict yesterday's snowfall (NewSnow_lag1)")

    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'predictions': y_pred_clipped,
        'actuals': y_test
    }

def analyze_feature_importance(model, feature_names, top_n=20):
    """Analyze and display feature importance based on coefficients"""
    print("\n" + "=" * 80)
    print(f"TOP {top_n} MOST IMPORTANT FEATURES (by absolute coefficient)")
    print("=" * 80)

    # Get absolute coefficients
    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': model.coef_,
        'Abs_Coefficient': np.abs(model.coef_)
    })

    # Sort by absolute value
    coef_df = coef_df.sort_values('Abs_Coefficient', ascending=False)

    print(coef_df.head(top_n).to_string(index=False))

    return coef_df

def plot_predictions(y_test, y_pred, save_path='predictions_plot.png'):
    """Plot actual vs predicted snowfall"""
    plt.figure(figsize=(12, 5))

    # Scatter plot
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, y_pred, alpha=0.3, s=10)
    plt.plot([0, y_test.max()], [0, y_test.max()], 'r--', lw=2, label='Perfect prediction')
    plt.xlabel('Actual Snowfall (inches)')
    plt.ylabel('Predicted Snowfall (inches)')
    plt.title('Actual vs Predicted Snowfall')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Residuals histogram
    plt.subplot(1, 2, 2)
    residuals = y_test - y_pred
    plt.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Residual (Actual - Predicted)')
    plt.ylabel('Frequency')
    plt.title('Residuals Distribution')
    plt.axvline(x=0, color='r', linestyle='--', lw=2)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {save_path}")
    plt.close()

def analyze_by_resort(test, predictions):
    """Analyze performance by resort"""
    print("\n" + "=" * 80)
    print("PERFORMANCE BY RESORT")
    print("=" * 80)

    test_with_pred = test.copy()
    test_with_pred['Prediction'] = predictions

    for resort in test['Resort'].unique():
        resort_data = test_with_pred[test_with_pred['Resort'] == resort]
        y_true = resort_data['NewSnow_in']
        y_pred = resort_data['Prediction']

        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        print(f"\n{resort}:")
        print(f"  Samples: {len(resort_data)}")
        print(f"  RMSE: {rmse:.4f} inches")
        print(f"  MAE:  {mae:.4f} inches")
        print(f"  R²:   {r2:.4f}")

def main():
    """Main training pipeline"""
    print("\n" + "=" * 80)
    print("LINEAR REGRESSION BASELINE - SNOWFALL PREDICTION")
    print("=" * 80 + "\n")

    # Load data
    df = load_data()

    # Train/test split (use 2023+ as test set)
    train, test = train_test_split_timeseries(df, test_start_date='2023-01-01')

    # Prepare features
    X_train, X_test, y_train, y_test, feature_cols, scaler = prepare_features(train, test)

    # Train model
    model = train_linear_regression(X_train, y_train)

    # Evaluate
    results = evaluate_model(model, X_train, X_test, y_train, y_test, dataset_name="Test")

    # Feature importance
    coef_df = analyze_feature_importance(model, feature_cols, top_n=20)

    # Plot predictions
    plot_predictions(results['actuals'], results['predictions'])

    # Performance by resort
    analyze_by_resort(test, results['predictions'])

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print("\nKey Takeaways:")
    print("1. Check R² score - closer to 1.0 is better")
    print("2. RMSE/MAE show average prediction error in inches")
    print("3. Review top features to understand what drives snowfall")
    print("4. Compare against a persistence baseline (predict lag1)")

    return model, scaler, feature_cols, results

if __name__ == "__main__":
    model, scaler, feature_cols, results = main()
