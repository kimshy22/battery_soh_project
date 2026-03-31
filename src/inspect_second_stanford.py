import pandas as pd
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
file_path = project_root / "output" / "LFP_k1_1C_25degC_all_discharge_rows.csv"

df = pd.read_csv(file_path)

# Keep useful columns
df = df[['Test_Time(s)', 'Step_Time(s)', 'Step_Index',
         'Voltage(V)', 'Current(A)', 'Surface_Temp(degC)']].copy()

# Convert to numeric where needed
for col in ['Test_Time(s)', 'Step_Time(s)', 'Step_Index', 'Voltage(V)', 'Current(A)', 'Surface_Temp(degC)']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.dropna().reset_index(drop=True)

# Group by step and summarize behavior
summary = df.groupby('Step_Index').agg(
    rows=('Step_Index', 'size'),
    test_time_start_s=('Test_Time(s)', 'min'),
    test_time_end_s=('Test_Time(s)', 'max'),
    step_time_end_s=('Step_Time(s)', 'max'),
    voltage_start_v=('Voltage(V)', 'first'),
    voltage_end_v=('Voltage(V)', 'last'),
    current_mean_a=('Current(A)', 'mean'),
    current_min_a=('Current(A)', 'min'),
    current_max_a=('Current(A)', 'max'),
    temp_mean_c=('Surface_Temp(degC)', 'mean')
).reset_index()

print("\n===== STEP SUMMARY =====")
print(summary.to_string(index=False))
