import pandas as pd
import requests
import io
from datetime import datetime
import time

# --- CONFIGURATION ---
START_DATE = "2005-01-01"
END_DATE = datetime.now().strftime("%Y-%m-%d")

# UPDATED Station IDs
# Sometimes the State/Network format is picky.
# We will use the standard format expected by the Report Generator.
stations = {
    "Palisades Tahoe": "784:CA:SNTL",
    "Heavenly": "518:CA:SNTL",
    "Mammoth": "846:CA:SNTL"
}

# The URL construction
base_url = (
    "https://wcc.sc.egov.usda.gov/reportGenerator/view_csv/"
    "customSingleStationReport/daily/{station_id}|id=%22%22|name/"
    "{start},{end}/WTEQ::value,SNWD::value,PRCP::value,TMAX::value,TMIN::value"
)

# Headers to look like a real browser (CRITICAL FIX)
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}


def fetch_and_clean_station(resort_name, station_id):
    print(f"--- Processing {resort_name} ({station_id}) ---")

    # 1. Construct URL
    url = base_url.format(station_id=station_id, start=START_DATE, end=END_DATE)
    print(f"Requesting URL: {url}")

    try:
        # 2. Fetch Data with Headers
        response = requests.get(url, headers=headers, timeout=10)

        # Check for server errors (404, 500)
        if response.status_code != 200:
            print(f"Error: Server returned status code {response.status_code}")
            return None

        # Check if content is empty
        if not response.text.strip():
            print("Error: Server returned empty text.")
            return None

        # Debug: Print first few lines to ensure it's CSV
        # print(f"First 200 chars of response:\n{response.text[:200]}\n")

        # 3. Parse CSV
        # 'error_bad_lines' is deprecated in newer pandas, use 'on_bad_lines'
        df = pd.read_csv(io.StringIO(response.text), comment='#')

        if df.empty:
            print("Error: DataFrame is empty after parsing.")
            return None

        # 4. Clean Header Names
        # SNOTEL column names can change slightly, so we rely on position or keyword matching if strict naming fails
        # But usually, this mapping works if the URL request was correct
        df.columns = ['Date', 'SWE_in', 'SnowDepth_in', 'PrecipAccum_in', 'MaxTemp_F', 'MinTemp_F']

        # 5. Date Parsing
        df['Date'] = pd.to_datetime(df['Date'])

        # 6. Feature Engineering
        df['NewSnow_in'] = df['SnowDepth_in'].diff().clip(lower=0)
        df['Precip_Liquid_in'] = df['PrecipAccum_in'].diff().clip(lower=0)
        df['Resort'] = resort_name

        print(f" Success! Retrieved {len(df)} rows.")
        return df

    except Exception as e:
        print(f"Critical Error fetching {resort_name}: {e}")
        return None


# --- MAIN EXECUTION ---
all_data = []

for resort, snotel_id in stations.items():
    station_df = fetch_and_clean_station(resort, snotel_id)
    if station_df is not None:
        all_data.append(station_df)
    # Be polite to the server
    time.sleep(1)

if all_data:
    # Combine all into one dataset
    final_df = pd.concat(all_data, ignore_index=True)

    # Final Cleanup
    final_df.dropna(subset=['NewSnow_in'], inplace=True)

    # Save
    filename = "stage1_ski_data.csv"
    final_df.to_csv(filename, index=False)

    print("\n" + "=" * 40)
    print(f"DONE! Saved {len(final_df)} rows to {filename}")
    print("=" * 40)
    print("\nSample Data:")
    print(final_df.head())
else:
    print("\nNo data collected. Please check your internet connection or the URLs.")