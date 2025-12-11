"""
Check for data leakage in engineered features
Identify features that contain information from the current day (target)
"""
import pandas as pd

# Load engineered data
df = pd.read_csv('nwcc_snow_data_engineered.csv')
df['Date'] = pd.to_datetime(df['Date'])

print("=" * 80)
print("DATA LEAKAGE ANALYSIS")
print("=" * 80)

# Check specific suspicious features
test_date = '2023-01-15'
test_resort = 'Heavenly'

sample = df[(df['Date'] == test_date) & (df['Resort'] == test_resort)].iloc[0]

print(f"\nSample Date: {test_date}, Resort: {test_resort}")
print(f"\nTarget: NewSnow_in = {sample['NewSnow_in']:.4f}")

# Get the original raw values (if still in dataset)
print("\n--- Checking Rolling Windows ---")
print("Rolling windows should NOT include current day's value")
print("They should only look backward (1 to 7 days ago)")

# Check if rolling mean includes current value
# For a proper backward-looking 7-day rolling mean, it should average days t-7 to t-1
# If it includes day t, it will be too close to the current SnowDepth

print("\n--- LEAKAGE SUSPECTS ---")

print("\n1. snow_to_liquid_ratio:")
print(f"   Formula: NewSnow_in / Precip_Liquid_in")
print(f"   Value: {sample['snow_to_liquid_ratio']:.4f}")
print("   PROBLEM: Uses TARGET VARIABLE (NewSnow_in) to create feature!")
print("   This is DIRECT LEAKAGE - model sees the answer!")

print("\n2. swe_to_depth_ratio:")
print(f"   Formula: SWE_in / SnowDepth_in")
print(f"   Value: {sample['swe_to_depth_ratio']:.4f}")
print("   PROBLEM: Uses current day's SnowDepth_in")
print("   Since NewSnow = SnowDepth(t) - SnowDepth(t-1), this leaks info!")

print("\n3. Rolling window features (e.g., SnowDepth_rolling_mean_7d):")
print(f"   Value: {sample['SnowDepth_rolling_mean_7d']:.4f}")
print("   PROBLEM: If min_periods=1 and includes current observation,")
print("   this contains today's snow depth -> direct leakage!")

print("\n4. Derived features using current day values:")
derived_current = ['temp_range', 'freezing_level', 'is_freezing', 'cold_snap',
                  'is_precip_day', 'precip_with_freezing', 'heavy_precip',
                  'temp_precip_interaction', 'freezing_precip', 'month_temp']
print(f"   Features using current day temp/precip: {len(derived_current)}")
print("   These MAY be okay if we're predicting based on current conditions")
print("   BUT only if available before the snowfall measurement!")

print("\n" + "=" * 80)
print("LEAKAGE CONFIRMATION TEST")
print("=" * 80)

# Calculate correlation between suspicious features and target
from sklearn.metrics import r2_score
import numpy as np

correlations = []
for col in df.columns:
    if col not in ['Date', 'Resort', 'NewSnow_in', 'season'] and df[col].dtype != 'object':
        corr = df[['NewSnow_in', col]].corr().iloc[0, 1]
        if abs(corr) > 0.5:
            correlations.append((col, abs(corr)))

correlations.sort(key=lambda x: x[1], reverse=True)

print("\nFeatures with |correlation| > 0.5 with target (TOP 15):")
print("High correlation suggests potential leakage\n")
for feat, corr in correlations[:15]:
    print(f"{feat:40s} {corr:.4f}")

print("\n" + "=" * 80)
print("RECOMMENDED FIXES")
print("=" * 80)
print("""
1. CRITICAL - Remove snow_to_liquid_ratio (uses target variable!)
2. CRITICAL - Remove swe_to_depth_ratio (uses current SnowDepth)
3. FIX - Rolling windows: Change from rolling(window=N, min_periods=1)
         to rolling(window=N).shift(1) to exclude current observation
4. REVIEW - Derived features: Only use if the data is available BEFORE
            the daily snow measurement (typically measured at end of day)

The model's R² = 0.969 is artificially high due to these leakage issues.
After fixing, expect R² to drop significantly (0.3-0.6 is more realistic for weather).
""")
