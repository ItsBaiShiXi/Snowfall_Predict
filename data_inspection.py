"""
Data inspection script for snowfall prediction project
Analyzes patterns, missing values, and provides feature engineering insights
"""
import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('nwcc_snow_data.csv')
df['Date'] = pd.to_datetime(df['Date'])

print("=" * 80)
print("DATASET OVERVIEW")
print("=" * 80)
print(f"Shape: {df.shape}")
print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
print(f"Resorts: {df['Resort'].unique()}")
print(f"Rows per resort: {df.groupby('Resort').size()}")

print("\n" + "=" * 80)
print("MISSING VALUES ANALYSIS")
print("=" * 80)
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({
    'Missing Count': missing,
    'Missing %': missing_pct
}).sort_values('Missing %', ascending=False)
print(missing_df[missing_df['Missing Count'] > 0])

print("\n" + "=" * 80)
print("TARGET VARIABLE (NewSnow_in) STATISTICS")
print("=" * 80)
print(df['NewSnow_in'].describe())
print(f"\nDays with snowfall > 0: {(df['NewSnow_in'] > 0).sum()} ({(df['NewSnow_in'] > 0).mean()*100:.2f}%)")
print(f"Days with heavy snow (>6in): {(df['NewSnow_in'] > 6).sum()} ({(df['NewSnow_in'] > 6).mean()*100:.2f}%)")

print("\n" + "=" * 80)
print("BASIC STATISTICS BY RESORT")
print("=" * 80)
for resort in df['Resort'].unique():
    resort_df = df[df['Resort'] == resort]
    print(f"\n{resort}:")
    print(f"  Avg NewSnow: {resort_df['NewSnow_in'].mean():.2f} in")
    print(f"  Max NewSnow: {resort_df['NewSnow_in'].max():.2f} in")
    print(f"  Avg Temp: {resort_df['AvgTemp_F'].mean():.2f}°F")
    print(f"  Snowfall days: {(resort_df['NewSnow_in'] > 0).mean()*100:.2f}%")

print("\n" + "=" * 80)
print("SEASONAL PATTERNS")
print("=" * 80)
df['Month'] = df['Date'].dt.month
monthly_snow = df.groupby('Month')['NewSnow_in'].agg(['mean', 'sum', 'count'])
monthly_snow.columns = ['Avg Daily', 'Total', 'Days']
print(monthly_snow)

print("\n" + "=" * 80)
print("TEMPERATURE VS SNOWFALL INSIGHTS")
print("=" * 80)
# Snow days vs no-snow days temperature comparison
snow_days = df[df['NewSnow_in'] > 0]
no_snow_days = df[df['NewSnow_in'] == 0]
print(f"Avg temp on snow days: {snow_days['AvgTemp_F'].mean():.2f}°F")
print(f"Avg temp on no-snow days: {no_snow_days['AvgTemp_F'].mean():.2f}°F")
print(f"Avg precip on snow days: {snow_days['Precip_Liquid_in'].mean():.3f} in")
print(f"Avg precip on no-snow days: {no_snow_days['Precip_Liquid_in'].mean():.3f} in")

print("\n" + "=" * 80)
print("CORRELATION WITH TARGET (NewSnow_in)")
print("=" * 80)
numeric_cols = df.select_dtypes(include=[np.number]).columns
correlations = df[numeric_cols].corr()['NewSnow_in'].sort_values(ascending=False)
print(correlations)

print("\n" + "=" * 80)
print("DATA AVAILABILITY OVER TIME (for optional features)")
print("=" * 80)
df['Year'] = df['Date'].dt.year
availability = df.groupby('Year')[['RelHumidity_pct', 'WindSpeedMax_mph', 'WindDir_deg']].apply(
    lambda x: (~x.isnull()).mean() * 100
)
print(availability.describe())
print(f"\nFirst year with >50% humidity data: {availability[availability['RelHumidity_pct'] > 50].index.min()}")
print(f"First year with >50% wind data: {availability[availability['WindSpeedMax_mph'] > 50].index.min()}")
