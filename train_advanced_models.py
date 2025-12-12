"""
Train Random Forest and XGBoost models for snowfall prediction
Compare against Linear Regression baseline
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import time

def load_data(filepath='nwcc_snow_data_engineered_fixed.csv'):
    """Load engineered dataset"""
    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def train_test_split_timeseries(df, test_start_date='2023-01-01'):
    """Time-series split"""
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
    """Prepare features and target"""
    exclude_cols = ['Date', 'Resort', 'season', 'NewSnow_in',
                    'SWE_in', 'SnowDepth_in', 'PrecipAccum_in',
                    'MaxTemp_F', 'MinTemp_F', 'AvgTemp_F',
                    'SoilTemp_2in_F', 'Precip_Liquid_in']

    feature_cols = [col for col in train.columns if col not in exclude_cols]

    X_train = train[feature_cols]
    y_train = train[target]
    X_test = test[feature_cols]
    y_test = test[target]

    print("\n" + "=" * 80)
    print("FEATURE PREPARATION")
    print("=" * 80)
    print(f"Total features: {len(feature_cols)}")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")

    return X_train, X_test, y_train, y_test, feature_cols, train, test

def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """Evaluate model performance"""
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Clip negative predictions
    y_test_pred_clipped = np.clip(y_test_pred, 0, None)
    y_train_pred_clipped = np.clip(y_train_pred, 0, None)

    # Metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred_clipped))
    train_mae = mean_absolute_error(y_train, y_train_pred_clipped)
    train_r2 = r2_score(y_train, y_train_pred_clipped)

    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred_clipped))
    test_mae = mean_absolute_error(y_test, y_test_pred_clipped)
    test_r2 = r2_score(y_test, y_test_pred_clipped)

    print(f"\n{model_name} Performance:")
    print(f"  Train - RMSE: {train_rmse:.4f}, MAE: {train_mae:.4f}, R²: {train_r2:.4f}")
    print(f"  Test  - RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}, R²: {test_r2:.4f}")

    return {
        'model_name': model_name,
        'train_rmse': train_rmse,
        'train_mae': train_mae,
        'train_r2': train_r2,
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'test_r2': test_r2,
        'predictions': y_test_pred_clipped,
        'test_data': y_test
    }

def train_linear_regression(X_train, X_test, y_train, y_test):
    """Train Linear Regression (for comparison)"""
    print("\n" + "=" * 80)
    print("TRAINING LINEAR REGRESSION (BASELINE)")
    print("=" * 80)

    # Scale features for Linear Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    start_time = time.time()
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    train_time = time.time() - start_time

    print(f"Training time: {train_time:.2f} seconds")

    # Create wrapper to handle scaled input
    class ScaledLinearRegression:
        def __init__(self, model, scaler):
            self.model = model
            self.scaler = scaler

        def predict(self, X):
            return self.model.predict(self.scaler.transform(X))

    wrapped_model = ScaledLinearRegression(model, scaler)
    return evaluate_model(wrapped_model, X_train, X_test, y_train, y_test, "Linear Regression")

def train_random_forest(X_train, X_test, y_train, y_test, feature_cols, tune_hyperparams=True):
    """Train Random Forest with optional hyperparameter tuning"""
    print("\n" + "=" * 80)
    print("TRAINING RANDOM FOREST")
    print("=" * 80)

    start_time = time.time()

    if tune_hyperparams:
        print("Running hyperparameter tuning with RandomizedSearchCV...")

        # Parameter distribution for RandomizedSearchCV
        param_dist = {
            'n_estimators': [100, 200, 300, 500],
            'max_depth': [10, 15, 20, 25, None],        # Limit depth to reduce overfitting
            'min_samples_split': [5, 10, 20, 30],       # Require more samples to split
            'min_samples_leaf': [2, 4, 8, 12],          # Require more samples per leaf
            'max_features': ['sqrt', 'log2', 0.5],      # Limit features per split
            'max_samples': [0.6, 0.7, 0.8, 0.9]         # Bootstrap sample size
        }

        # Base model
        rf_base = RandomForestRegressor(random_state=42, n_jobs=-1)

        # RandomizedSearchCV
        random_search = RandomizedSearchCV(
            rf_base,
            param_distributions=param_dist,
            n_iter=30,              # Try 30 random combinations
            cv=3,                   # 3-fold cross-validation
            scoring='r2',
            n_jobs=-1,
            verbose=2,
            random_state=42
        )

        random_search.fit(X_train, y_train)
        model = random_search.best_estimator_

        train_time = time.time() - start_time
        print(f"\nBest parameters found:")
        for param, value in random_search.best_params_.items():
            print(f"  {param}: {value}")
        print(f"Best CV R² score: {random_search.best_score_:.4f}")
        print(f"Training time (including tuning): {train_time:.2f} seconds")
    else:
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=20,
            min_samples_split=10,
            min_samples_leaf=4,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1,
            verbose=1
        )
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
        print(f"Training time: {train_time:.2f} seconds")

    # Feature importance
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\nTop 15 Most Important Features:")
    print(importance_df.head(15).to_string(index=False))

    model_name = "Random Forest (Tuned)" if tune_hyperparams else "Random Forest"
    results = evaluate_model(model, X_train, X_test, y_train, y_test, model_name)
    results['feature_importance'] = importance_df
    return results

def train_xgboost(X_train, X_test, y_train, y_test, feature_cols, tune_hyperparams=True):
    """Train XGBoost with optional hyperparameter tuning"""
    print("\n" + "=" * 80)
    print("TRAINING XGBOOST")
    print("=" * 80)

    start_time = time.time()

    if tune_hyperparams:
        print("Using best hyperparameters from previous tuning...")

        # # Parameter distribution with regularization focus
        # param_dist = {
        #     # It hit 6, so we start at 6 and go deeper
        #     'max_depth': [6, 8, 10, 12],
        #
        #     # It hit 7, so we go much higher to see if it wants simpler trees
        #     'min_child_weight': [7, 10, 15, 20],
        #
        #     # Keep 0.01 since it won, but maybe try slightly faster 0.02
        #     'learning_rate': [0.005, 0.01, 0.02],
        #
        #     # It wants less data per tree (high variance reduction)
        #     'subsample': [0.5, 0.6, 0.7],
        #
        #     # It liked 0.8, so center around that
        #     'colsample_bytree': [0.7, 0.8, 0.9],
        #
        #     # Low learning rate needs MORE trees. 1000 was the limit, go higher.
        #     'n_estimators': [1000, 1500, 2000, 3000],
        #
        #     # It hit 0.2 (max), so let's try significantly higher regularization
        #     'gamma': [0.2, 0.5, 1.0, 2.0]
        # }
        #
        # # Base model
        # xgb_base = xgb.XGBRegressor(random_state=42, n_jobs=-1, verbosity=1)
        #
        # # RandomizedSearchCV
        # random_search = RandomizedSearchCV(
        #     xgb_base,
        #     param_distributions=param_dist,
        #     n_iter=30,              # Try 30 random combinations
        #     cv=3,                   # 3-fold cross-validation
        #     scoring='r2',
        #     n_jobs=-1,
        #     verbose=2,
        #     random_state=42
        # )
        #
        # random_search.fit(X_train, y_train)
        # model = random_search.best_estimator_

        # Best parameters from hyperparameter tuning (CV R² = 0.7116)
        best_params = {
            'subsample': 0.6,
            'n_estimators': 1000,
            'min_child_weight': 20,
            'max_depth': 8,
            'learning_rate': 0.005,
            'gamma': 2.0,
            'colsample_bytree': 0.8
        }

        print(f"\nBest parameters:")
        for param, value in best_params.items():
            print(f"  {param}: {value}")

        model = xgb.XGBRegressor(
            **best_params,
            random_state=42,
            n_jobs=-1,
            verbosity=1
        )

        model.fit(X_train, y_train, verbose=False)
        train_time = time.time() - start_time
        print(f"Training time: {train_time:.2f} seconds")
    else:
        model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            gamma=0.1,
            random_state=42,
            n_jobs=-1,
            verbosity=1
        )
        model.fit(X_train, y_train, verbose=False)
        train_time = time.time() - start_time
        print(f"Training time: {train_time:.2f} seconds")

    # Feature importance
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\nTop 15 Most Important Features:")
    print(importance_df.head(15).to_string(index=False))

    model_name = "XGBoost (Tuned)" if tune_hyperparams else "XGBoost"
    results = evaluate_model(model, X_train, X_test, y_train, y_test, model_name)
    results['feature_importance'] = importance_df
    results['model'] = model  # Save model for later use
    return results

def compare_models(results_list):
    """Compare all models"""
    print("\n" + "=" * 80)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 80)

    comparison_df = pd.DataFrame([
        {
            'Model': r['model_name'],
            'Train RMSE': r['train_rmse'],
            'Test RMSE': r['test_rmse'],
            'Train MAE': r['train_mae'],
            'Test MAE': r['test_mae'],
            'Train R²': r['train_r2'],
            'Test R²': r['test_r2']
        }
        for r in results_list
    ])

    print("\n" + comparison_df.to_string(index=False))

    # Find best model
    best_model_idx = comparison_df['Test R²'].idxmax()
    best_model = comparison_df.iloc[best_model_idx]['Model']
    best_r2 = comparison_df.iloc[best_model_idx]['Test R²']

    print(f"\nBest Model: {best_model} (Test R² = {best_r2:.4f})")

    # Improvement over baseline
    baseline_r2 = comparison_df[comparison_df['Model'] == 'Linear Regression']['Test R²'].values[0]
    for _, row in comparison_df.iterrows():
        if row['Model'] != 'Linear Regression':
            improvement = (row['Test R²'] - baseline_r2) / baseline_r2 * 100
            print(f"  {row['Model']}: {improvement:+.1f}% improvement over Linear Regression")

    return comparison_df

def plot_comparison(results_list, save_path='model_comparison.png'):
    """Plot model comparison"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Extract data
    models = [r['model_name'] for r in results_list]
    test_r2 = [r['test_r2'] for r in results_list]
    test_rmse = [r['test_rmse'] for r in results_list]
    test_mae = [r['test_mae'] for r in results_list]

    # Plot 1: R² Score
    axes[0, 0].bar(models, test_r2, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    axes[0, 0].set_ylabel('R² Score')
    axes[0, 0].set_title('Test Set R² Score (Higher is Better)')
    axes[0, 0].set_ylim([0, 1])
    for i, v in enumerate(test_r2):
        axes[0, 0].text(i, v + 0.02, f'{v:.4f}', ha='center', fontweight='bold')

    # Plot 2: RMSE
    axes[0, 1].bar(models, test_rmse, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    axes[0, 1].set_ylabel('RMSE (inches)')
    axes[0, 1].set_title('Test Set RMSE (Lower is Better)')
    for i, v in enumerate(test_rmse):
        axes[0, 1].text(i, v + 0.05, f'{v:.4f}', ha='center', fontweight='bold')

    # Plot 3: MAE
    axes[1, 0].bar(models, test_mae, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    axes[1, 0].set_ylabel('MAE (inches)')
    axes[1, 0].set_title('Test Set MAE (Lower is Better)')
    for i, v in enumerate(test_mae):
        axes[1, 0].text(i, v + 0.02, f'{v:.4f}', ha='center', fontweight='bold')

    # Plot 4: Predictions scatter (best model)
    best_idx = np.argmax(test_r2)
    best_results = results_list[best_idx]
    axes[1, 1].scatter(best_results['test_data'], best_results['predictions'],
                      alpha=0.3, s=20, color=['#1f77b4', '#ff7f0e', '#2ca02c'][best_idx])
    max_val = max(best_results['test_data'].max(), best_results['predictions'].max())
    axes[1, 1].plot([0, max_val], [0, max_val], 'r--', lw=2, label='Perfect prediction')
    axes[1, 1].set_xlabel('Actual Snowfall (inches)')
    axes[1, 1].set_ylabel('Predicted Snowfall (inches)')
    axes[1, 1].set_title(f'{best_results["model_name"]} - Predictions')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nComparison plot saved to: {save_path}")
    plt.close()

def analyze_by_resort(test_df, results_list):
    """Analyze performance by resort for all models"""
    print("\n" + "=" * 80)
    print("PERFORMANCE BY RESORT")
    print("=" * 80)

    for results in results_list:
        print(f"\n{results['model_name']}:")
        test_with_pred = test_df.copy()
        test_with_pred['Prediction'] = results['predictions']

        for resort in test_df['Resort'].unique():
            resort_data = test_with_pred[test_with_pred['Resort'] == resort]
            y_true = resort_data['NewSnow_in']
            y_pred = resort_data['Prediction']

            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)

            print(f"  {resort:20s} - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")

def main():
    """Main training pipeline"""
    print("\n" + "=" * 80)
    print("ADVANCED MODEL TRAINING - SNOWFALL PREDICTION")
    print("=" * 80 + "\n")

    # Load data
    df = load_data()

    # Train/test split
    train, test = train_test_split_timeseries(df)

    # Prepare features
    X_train, X_test, y_train, y_test, feature_cols, train_df, test_df = prepare_features(train, test)

    # Train all models
    results = []

    # 1. Linear Regression (baseline)
    lr_results = train_linear_regression(X_train, X_test, y_train, y_test)
    results.append(lr_results)

    # 2. Random Forest
    rf_results = train_random_forest(X_train, X_test, y_train, y_test, feature_cols)
    results.append(rf_results)

    # 3. XGBoost
    xgb_results = train_xgboost(X_train, X_test, y_train, y_test, feature_cols)
    results.append(xgb_results)

    # Compare models
    comparison_df = compare_models(results)

    # Plot comparison
    plot_comparison(results)

    # Analyze by resort
    analyze_by_resort(test_df, results)

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)

    return results, comparison_df

if __name__ == "__main__":
    results, comparison_df = main()
