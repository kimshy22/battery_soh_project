import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# =========================================================
# 1. USER SETTINGS
# =========================================================
Q_RATED_AH = 2.5
NOMINAL_VOLTAGE_V = 3.3

project_root = Path(__file__).resolve().parent.parent
file_path = project_root / "output" / "LFP_k1_1C_25degC_all_discharge_rows.csv"

USE_PARTIAL_SOC_CORRECTION = False
SOC_START = 90.0
SOC_END = 20.0

# =========================================================
# 2. LOAD DISCHARGE-ONLY CSV
# =========================================================
print("Loading discharge-only CSV...")
df = pd.read_csv(file_path)

print("\nFirst 5 rows:")
print(df.head())

print("\nOriginal columns:")
print(df.columns.tolist())

print("\nDataset shape:")
print(df.shape)

# =========================================================
# 3. RENAME COLUMNS TO STANDARD NAMES
# =========================================================
df = df.rename(columns={
    'Test_Time(s)': 'time_s',
    'Voltage(V)': 'voltage_v',
    'Current(A)': 'current_a',
    'Surface_Temp(degC)': 'temp_c'
})

print("\nColumns after renaming:")
print(df.columns.tolist())

# =========================================================
# 4. KEEP ONLY REQUIRED COLUMNS
# =========================================================
required_cols = ['time_s', 'voltage_v', 'current_a', 'temp_c']
df = df[required_cols].copy()

# Convert to numeric just in case
for col in required_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Clean the dataset
df = df.dropna()
df = df.sort_values('time_s').reset_index(drop=True)

print("\nCleaned data preview:")
print(df.head())

print("\nMissing values after cleaning:")
print(df.isnull().sum())

# =========================================================
# 5. COMPUTE TIME DIFFERENCE
# =========================================================
df['dt_s'] = df['time_s'].diff()
df.loc[0, 'dt_s'] = 0

print("\nData with time difference:")
print(df.head())

# =========================================================
# 6. CHECK CURRENT VALUES
# =========================================================
print("\nCurrent statistics:")
print(df['current_a'].describe())

print("\nFirst 10 current values:")
print(df['current_a'].head(10).tolist())

# Because this CSV already contains discharge-only data,
# we do NOT need to filter discharge again.
discharge_df = df.copy()

# =========================================================
# 7. COMPUTE CAPACITY
# =========================================================
discharge_df['delta_q_ah'] = (
    discharge_df['current_a'].abs() * discharge_df['dt_s'] / 3600.0
)

discharge_df['cum_capacity_ah'] = discharge_df['delta_q_ah'].cumsum()

Q_window_ah = discharge_df['delta_q_ah'].sum()

print(f"\nMeasured discharge capacity in this window: {Q_window_ah:.6f} Ah")

# =========================================================
# 8. DETERMINE CURRENT CAPACITY
# =========================================================
if USE_PARTIAL_SOC_CORRECTION:
    soc_fraction = (SOC_START - SOC_END) / 100.0

    if soc_fraction <= 0:
        raise ValueError(
            "SOC_START must be greater than SOC_END for discharge.")

    Q_current_ah = Q_window_ah / soc_fraction
    method_used = "Partial SOC window correction"
else:
    Q_current_ah = Q_window_ah
    method_used = "Full discharge assumption"

print(f"Capacity method used: {method_used}")
print(f"Estimated current capacity: {Q_current_ah:.6f} Ah")

# =========================================================
# 9. COMPUTE SOH
# =========================================================
SOH_percent = (Q_current_ah / Q_RATED_AH) * 100.0

print("\n===== SOH RESULTS =====")
print(f"Nominal voltage: {NOMINAL_VOLTAGE_V:.2f} V")
print(f"Rated capacity: {Q_RATED_AH:.3f} Ah")
print(f"Measured discharge capacity: {Q_window_ah:.6f} Ah")
print(f"Estimated current capacity: {Q_current_ah:.6f} Ah")
print(f"SOH: {SOH_percent:.2f} %")

# =========================================================
# 10. VISUALIZATIONS
# =========================================================

# Voltage vs Time
plt.figure(figsize=(8, 5))
plt.plot(discharge_df['time_s'], discharge_df['voltage_v'])
plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')
plt.title('Voltage vs Time (Discharge Only)')
plt.grid(True)
plt.tight_layout()
plt.show()

# Current vs Time
plt.figure(figsize=(8, 5))
plt.plot(discharge_df['time_s'], discharge_df['current_a'])
plt.xlabel('Time (s)')
plt.ylabel('Current (A)')
plt.title('Current vs Time (Discharge Only)')
plt.grid(True)
plt.tight_layout()
plt.show()

# Temperature vs Time
plt.figure(figsize=(8, 5))
plt.plot(discharge_df['time_s'], discharge_df['temp_c'])
plt.xlabel('Time (s)')
plt.ylabel('Temperature (°C)')
plt.title('Temperature vs Time (Discharge Only)')
plt.grid(True)
plt.tight_layout()
plt.show()

# Voltage vs Discharged Capacity
plt.figure(figsize=(8, 5))
plt.plot(discharge_df['cum_capacity_ah'], discharge_df['voltage_v'])
plt.xlabel('Discharged Capacity (Ah)')
plt.ylabel('Voltage (V)')
plt.title('Voltage vs Discharged Capacity')
plt.grid(True)
plt.tight_layout()
plt.show()
