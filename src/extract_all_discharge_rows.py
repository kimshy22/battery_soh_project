import pandas as pd
from pathlib import Path

# =========================================================
# PATHS
# =========================================================
project_root = Path(__file__).resolve().parent.parent
input_file = project_root / "data" / "standford" / "LFP_k1_1C_25degC (1).xlsx"
output_dir = project_root / "output"
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / "LFP_k1_1C_25degC_all_discharge_rows.csv"

# =========================================================
# LOAD DATA
# =========================================================
df = pd.read_excel(input_file)

print("Original dataset shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

# =========================================================
# CHECK THAT CURRENT COLUMN EXISTS
# =========================================================
if "Current(A)" not in df.columns:
    raise KeyError("Column 'Current(A)' not found in the dataset.")

# =========================================================
# EXTRACT ALL DISCHARGE ROWS
# =========================================================
# For this Stanford file, discharge is indicated by negative current
discharge_df = df[df["Current(A)"] < 0].copy()

print("\nDischarge-only dataset shape:", discharge_df.shape)
print("\nFirst 5 discharge rows:")
print(discharge_df.head())

print("\nUnique Step_Index values in discharge data:")
if "Step_Index" in discharge_df.columns:
    print(discharge_df["Step_Index"].unique())
else:
    print("No Step_Index column found.")

# =========================================================
# SAVE OUTPUT
# =========================================================
discharge_df.to_csv(output_file, index=False)

print(f"\nSaved discharge-only data to:\n{output_file}")
