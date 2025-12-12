import pandas as pd
import requests
import io
from datetime import datetime
import time

# --- CONFIGURATION ---
START_DATE = "2005-01-01"
END_DATE = datetime.now().strftime("%Y-%m-%d")

# UPDATED Station IDs
stations = {
    "Palisades Tahoe": "784:CA:SNTL",
    "Heavenly": "518:CA:SNTL",
    "Mammoth": "846:CA:SNTL"
}

# We keep Snow/Temp, add Humidity (snow quality), Wind (drifting), and Soil (early season base).
# We exclude the "trash" (Salinity, Turbidity, Battery Voltage).
part_list = [
    "WTEQ::value", "SNWD::value", "PRCP::value",  # Snow & Precip
    "TMAX::value", "TMIN::value", "TAVG::value",  # Temps
    "RHUM::value",                                # Relative Humidity (Wet bulb/Snow quality)
    "WSPDX::value", "WDIRV::value",               # Wind (Speed Max & Direction)
    "STO:-2:value", "SMS:-2:value"                # Soil Temp & Moisture (-2in depth)
]
elements = ",".join(part_list)

base_url = (
    "https://wcc.sc.egov.usda.gov/reportGenerator/view_csv/"
    "customSingleStationReport/daily/{station_id}|id=%22%22|name/"
    "{start},{end}/{elements}"
)

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

def fetch_and_clean_station(resort_name, station_id):
    print(f"--- Processing {resort_name} ({station_id}) ---")

    # 1. Construct URL with the elements variable
    url = base_url.format(station_id=station_id, start=START_DATE, end=END_DATE, elements=elements)

    try:
        # 2. Fetch Data
        response = requests.get(url, headers=headers, timeout=15)

        if response.status_code != 200:
            print(f"Error: Server returned status code {response.status_code}")
            return None

        if not response.text.strip():
            print("Error: Server returned empty text.")
            return None

        # 3. Parse CSV
        df = pd.read_csv(io.StringIO(response.text), comment='#')

        if df.empty:
            print("Error: DataFrame is empty after parsing.")
            return None

        # The order here MUST match the order in 'part_list' above + Date at the start.
        df.columns = [
            'Date', 
            'SWE_in', 'SnowDepth_in', 'PrecipAccum_in', 
            'MaxTemp_F', 'MinTemp_F', 'AvgTemp_F',
            'RelHumidity_pct', 
            'WindSpeedMax_mph', 'WindDir_deg',
            'SoilTemp_2in_F', 'SoilMoist_2in_pct'
        ]

        # 4. Date Parsing
        df['Date'] = pd.to_datetime(df['Date'])

        # 5. Feature Engineering
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
    time.sleep(1)

if all_data:
    final_df = pd.concat(all_data, ignore_index=True)

    # Only drop if the essential Snow Depth is missing
    final_df.dropna(subset=['NewSnow_in'], inplace=True)

    filename = "nwcc_snow_data.csv"
    final_df.to_csv(filename, index=False)

    print("\n" + "=" * 40)
    print(f"DONE! Saved {len(final_df)} rows to {filename}")
    print("=" * 40)
    print("\nSample Data:")
    print(final_df.head())
else:
    print("\nNo data collected.")