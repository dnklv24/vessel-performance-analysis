"""
===============================================================================
SINTEF OCEAN: DATA PROCESSING & CLEANING (FINAL ML-COMPATIBLE VERSION)
===============================================================================
"""

import os
import glob
import pandas as pd
import numpy as np

data_path = r"D:\APPLIED DATA SCIENCE\data"

OUT_NOON_CLEAN = os.path.join(data_path, "clean_vessel_data_noon.csv")
OUT_CLEANING_EVENTS = os.path.join(data_path, "cleaning_events.csv")
OUT_HIGH_FREQ = os.path.join(data_path, "clean_sensors_1min.csv")

# =============================================================================
# Step 1: noon reports processing
# =============================================================================
def process_noon_reports(data_path):
    print("\n" + "="*60)
    print("Step 1: Processing NOON REPORTS...")
    print("="*60)

    pattern = os.path.join(data_path, "N*.csv") #take all files starting with letter
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"[WARNING] Files N*.csv not found in {data_path}")
        return

    dfs = [] #create a list to be filled with monthly tables 
    for f in files:
        df_month = pd.read_csv(f)
        df_month['Source_File'] = os.path.basename(f) #create new column and put file names there to detect potential anomalies
        dfs.append(df_month)

    raw = pd.concat(dfs, ignore_index=True) #merge all tables from the list
    clean = pd.DataFrame() #box for clean columns

    # 1. time synchronization
    clean['Date'] = pd.to_datetime(raw['Date'], format='%d/%m/%Y')
    clean = clean.sort_values('Date').reset_index(drop=True)

    # 2. draft, trim & loading
    clean['Fwd_Draft_m'] = raw['Fwd Draft'].astype(float) #how deep goes the "nose"
    clean['Aft_Draft_m'] = raw['Aft Draft'].astype(float) #how deep goes the stern
    clean['Mean_Draft_m'] = ((clean['Fwd_Draft_m'] + clean['Aft_Draft_m']) / 2.0).round(3) #avg depth
    clean['Trim_m'] = (clean['Aft_Draft_m'] - clean['Fwd_Draft_m']).round(3) #longitudinal angle
    clean['Displacement_t'] = raw['Displacement'].astype(float) #actual ship's weight

    # sorting ships into 4 load classes based on weight & depth
    def classify_condition(row):
        disp = row['Displacement_t']
        draft = row['Mean_Draft_m']
        if disp > 75000 or draft > 12.5: return 'Scantling_Draft' #full ship load
        elif disp > 50000 or draft > 8.5: return 'Design_Draft' #standard ship load
        elif disp > 30000 or draft > 4.0: return 'Ballast_Draft' #non-commercial load (water)
        else: return 'Light_Ballast_or_Port' # no load (during stop)
    
    # assign the label to each row
    clean['Loading_Condition'] = clean.apply(classify_condition, axis=1)

    # 3. Fuel consumption & propeller slip
    # goal - to detect long stops when the ship is cleaned
    clean['Fuel_Consumption_mt_day'] = raw['Consumption for Propulsion'].fillna(0.0)
    clean['Slip_pct'] = raw['Slip'].replace('%:', '', regex=True).astype(float).fillna(0.0)
    #detect that the ship is moving if it consumes fuel & the propeller turns
    clean['Is_Underway'] = (clean['Fuel_Consumption_mt_day'] > 0) | (clean['Slip_pct'].abs() > 0.01)

    # 4. search of cleaning events
    idle = (~clean['Is_Underway']).astype(int) #if the ship isn't underway, its idle value = 1
    idle_groups = (idle != idle.shift()).cumsum() * idle #fixating the change of movement status; * idle making ID null for moving days, leaving only those with stopped
    group_lengths = clean.groupby(idle_groups)['Date'].transform('count') * idle
    clean['Potential_Cleaning_Window'] = group_lengths >= 3 #group data by ID and see the length of the stops

    clean.to_csv(OUT_NOON_CLEAN, index=False)

    events = clean[clean['Potential_Cleaning_Window']].copy()
    if not events.empty: #we check if there is at least one cleaning event 
        events_summary = events.groupby(idle_groups[clean['Potential_Cleaning_Window']]).agg(
            Start_Date=('Date', 'min'), #earliest date - beginning of stop
            End_Date=('Date', 'max'), # latest date - end of stop
            Duration_Days=('Date', 'count') #duration of stop
        ).reset_index(drop=True)
        events_summary.to_csv(OUT_CLEANING_EVENTS, index=False)
        print(f"Saved cleaning_events.csv ({len(events_summary)} windows)")


# =============================================================================
# Step 2: High-frequency logs processing
# =============================================================================
def process_high_frequency_logs(data_path):
    print("\n" + "="*60)
    print("Step 2: Processing MONTHLY HIGH-FREQUENCY LOGS...")
    print("="*60)

    monthly_wide_dfs = []
    
    for i in range(1, 13): #find all 12 files by name starting with a number
        possible_paths = [os.path.join(data_path, f"{i}.csv"), os.path.join(data_path, f"{i}.txt")]
        file_found = next((p for p in possible_paths if os.path.exists(p)), None)

        if file_found:
            print(f"--> Parsing month {i}...", end="", flush=True)
            records = []
            with open(file_found, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if line.startswith("2024") and "," in line:
                        parts = line.strip().split(",")
                        if len(parts) == 3:
                            records.append((parts[0].strip(), parts[1].strip(), parts[2].strip()))
                        #the line is split into 3 parts: time, sensor code, sensor value

            if records:
                df_m = pd.DataFrame(records, columns=["timestamp", "sensor_code", "value"])
                df_m["timestamp"] = pd.to_datetime(df_m["timestamp"], errors="coerce").dt.floor("1min") #round timestamps to minutes
                df_m["value"] = pd.to_numeric(df_m["value"], errors="coerce")
                df_m = df_m.dropna(subset=["timestamp", "value"])
                
                # wide format table: rows - minutes, columns - sensors 
                df_wide_month = df_m.groupby(["timestamp", "sensor_code"])["value"].mean().unstack("sensor_code")
                monthly_wide_dfs.append(df_wide_month)
                print(" Done")

    if not monthly_wide_dfs:
        return

    full_wide_df = pd.concat(monthly_wide_dfs, axis=0) #put all 12 months into one df
    df_base = full_wide_df.resample("1min").mean() #the step is exactly 1 minute

    rpm_col    = '2::0::11::0_1::1::0::3::0_1::0::6::0_8'     # Shaft RPM
    torque_col = '2::0::11::0_1::1::0::3::0_12::0::2::0_8'    # Shaft Torque
    power_col  = '2::0::11::0_1::1::0::3::0_14::0::1::0_8'    # Shaft Power (kW)
    thrust_col = '2::0::11::0_1::1::0::3::0_17::0::1::0_8'    # Shaft Thrust (kN)
    me_rpm_col = '3::0::1::0_1::2::0::8::0_1::0::6::0_8'      # Main Engine RPM
    sog_col    = '2::0::6::1_1::1::0::2::0_1::0::1::0_8'      # Speed Over Ground (knots)
    stw_col    = '2::0::7::0_1::1::0::2::0_1::0::5::11_8'     # Speed Through Water (knots)
    turn_col   = '2::0::5::0_1::1::0::2::0_6::0::1::0_8'      # Heading Turn Rate (deg/min)

    weather_cols = [c for c in df_base.columns if c.startswith('4::')]
    ship_cols = [c for c in df_base.columns if not c.startswith('4::')]

    for limit_val in [5, 10, 15]:
        print(f"\n--- Generating dataset with interpolation limit: {limit_val} ---")
        
        df_1min = df_base.copy()
        df_1min[ship_cols] = df_1min[ship_cols].interpolate(method="linear", limit=limit_val)
        df_1min[weather_cols] = df_1min[weather_cols].interpolate(method="linear", limit=65)
        df_1min = df_1min.reset_index()

        # =====================================================================
        # 6. Filtering of miscalibrated torque & port stays (Version 3 + ISO 19030)
        # =====================================================================
        # 6.1. First, apply Version 3 miscalibration filter BEFORE .abs()
        if rpm_col in df_1min.columns and torque_col in df_1min.columns:
            miscalibrated = (df_1min[rpm_col] > 20) & (df_1min[torque_col] < 1000)
            df_1min = df_1min[~miscalibrated].copy()
            print(f"--> Version 3 Torque Filter: deleted {miscalibrated.sum()} miscalibrated rows")

        # 6.2. Fix inverted sensor signs (.abs()) on the remaining healthy rows
        for col in [rpm_col, torque_col, power_col, thrust_col, me_rpm_col]:
            if col in df_1min.columns:
                df_1min[col] = df_1min[col].abs()

        # 6.3. Strictly remove Port Stays, Zero-Power Outages & Maneuvers ("Datatrash")
        # This keeps ONLY the ~7,000-9,000 pure ocean transit rows where all sensors work!
        valid_transit = (
            (df_1min[stw_col] > 3.0) & (df_1min[stw_col] <= 18.0) &
            (df_1min[sog_col] > 3.0) & (df_1min[sog_col] <= 18.0) &
            (df_1min[power_col] > 100.0) &
            (df_1min[rpm_col] > 20.0) &
            (df_1min[torque_col] > 10.0) &
            (df_1min[turn_col].abs() <= 5.0)
        )
        df_1min = df_1min[valid_transit].copy()
        print(f"--> Removed port stays & zero-power outages. Clean transit rows kept: {len(df_1min)}")

        # =====================================================================
        # 7. Merge cleaning events & real Draft from Noon Reports
        # =====================================================================
        df_1min['Is_Cleaning_Window'] = False
        timestamp_naive = df_1min['timestamp'].dt.tz_localize(None)
        
        if os.path.exists(OUT_CLEANING_EVENTS):
            events_df = pd.read_csv(OUT_CLEANING_EVENTS)
            events_df['Start_Date'] = pd.to_datetime(events_df['Start_Date'])
            events_df['End_Date'] = pd.to_datetime(events_df['End_Date']) + pd.Timedelta(days=1, seconds=-1)
            
            for _, row in events_df.iterrows():
                mask = (timestamp_naive >= row['Start_Date']) & (timestamp_naive <= row['End_Date'])
                df_1min.loc[mask, 'Is_Cleaning_Window'] = True

        # Replace broken 0.0 high-frequency draft columns with REAL Noon Report drafts
        # so Sina's ML script automatically computes true Draft and Trim!
        if os.path.exists(OUT_NOON_CLEAN):
            noon_df = pd.read_csv(OUT_NOON_CLEAN)
            noon_df['Date_Key'] = pd.to_datetime(noon_df['Date']).dt.date
            df_1min['Date_Key'] = timestamp_naive.dt.date
            
            noon_features = noon_df[['Date_Key', 'Fwd_Draft_m', 'Aft_Draft_m', 'Mean_Draft_m', 'Trim_m', 'Displacement_t']]
            df_1min = pd.merge(df_1min, noon_features, on='Date_Key', how='left')
            df_1min = df_1min.drop(columns=['Date_Key'])

            df_1min['3::0::1::0_1::1::0::2::0_11::0::1::0_8'] = df_1min['Fwd_Draft_m']
            df_1min['3::0::1::0_1::1::0::2::0_11::0::2::0_8'] = df_1min['Aft_Draft_m']
            df_1min['3::0::1::0_1::1::0::2::0_11::0::3::0_8'] = df_1min['Mean_Draft_m']
            df_1min['3::0::1::0_1::1::0::2::0_11::0::4::0_8'] = df_1min['Mean_Draft_m']
            print("--> Replaced broken 0.0 draft sensors with real Noon Report Draft & Trim.")

        # =====================================================================
        # 8. Column mapping & float conversion (No duplicate column names!)
        # =====================================================================
        sensor_cols = df_1min.columns.difference(['timestamp', 'Is_Cleaning_Window'])
        df_1min[sensor_cols] = df_1min[sensor_cols].astype(float)

        mapping_file = os.path.join(data_path, "Metrics registration.xlsx")
        if os.path.exists(mapping_file):
            mapping_df = pd.read_excel(mapping_file)
            name_counts = mapping_df['quantity_name'].value_counts()
            
            mapping_dict = {}
            seen_names = set()
            for _, row in mapping_df.iterrows():
                qid = row['qid_mapping']
                qname = str(row['quantity_name']).strip()
                sname = str(row['source_name']).strip()
                # Keep the first occurrence with its exact name (for Sina's script),
                # and append source_name to the second duplicate so Pandas doesn't create a 2D DataFrame!
                if qname in seen_names:
                    mapping_dict[qid] = f"{qname} ({sname})"
                else:
                    mapping_dict[qid] = qname
                    seen_names.add(qname)
                    
            df_1min = df_1min.rename(columns=mapping_dict)
            print("--> Applied unique readable column names compatible with ML script.")

        out_file = OUT_HIGH_FREQ.replace(".csv", f"_limit{limit_val}.csv")
        df_1min.to_csv(out_file, index=False)
        print(f"Saved {os.path.basename(out_file)} ({len(df_1min)} rows) to: {out_file}")


if __name__ == "__main__":
    process_noon_reports(data_path)
    process_high_frequency_logs(data_path)

