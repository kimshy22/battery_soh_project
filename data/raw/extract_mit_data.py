import h5py
import pandas as pd
import numpy as np


def extract_all_mit_discharge_data(mat_filename, output_csv, cell_index=0):
    print(f" Opening {mat_filename}... (This might take a moment)")

    try:
        f = h5py.File(mat_filename, 'r')
    except Exception as e:
        print(f"Error opening file: {e}")
        return

    batch = f['batch']
    cycles_ref = batch['cycles'][cell_index, 0]
    cycles = f[cycles_ref]

    # Get the absolute total number of cycles recorded for this cell
    total_recorded_cycles = cycles['I'].shape[0]

    print(
        f" Found {total_recorded_cycles} total cycles for Cell #{cell_index + 1}.")
    print("Extracting time-series columns for ALL discharge phases. This may take a few minutes...\n")

    all_cycle_data = []

    # REMOVED the mismatched linear arrays ('Qdlin', 'Tdlin', 'discharge_dQdV')
    columns_to_extract = ['I', 'Qc', 'Qd', 'T', 'V', 't']

    # Start from cycle 1 to avoid cycle 0 (which is a lab calibration cycle)
    for cycle_num in range(1, total_recorded_cycles):
        cycle_dict = {}

        # Extract the same-length time-series columns dynamically
        for col in columns_to_extract:
            try:
                cycle_dict[col] = f[cycles[col][cycle_num, 0]][:].flatten()
            except Exception as e:
                print(f" Skipping {col} in cycle {cycle_num}: {e}")

        # Convert this cycle's data into a Pandas DataFrame
        try:
            df_cycle = pd.DataFrame(cycle_dict)
        except ValueError as ve:
            print(
                f" Cycle {cycle_num} length mismatch. Skipping. Error: {ve}")
            continue

        # Tag the cycle number so your LSTM knows the sequence
        df_cycle['Cycle_Number'] = cycle_num

        # Convert time to seconds and rename columns to match our BMS pipeline
        df_cycle['Time_s'] = df_cycle['t'] * 60.0
        df_cycle = df_cycle.rename(columns={
            'I': 'Current_A',
            'V': 'Voltage_V',
            'T': 'Temp_C'
        })

        # FILTER: Keep only Discharge and Rest phases (Current <= 0.05A)
        df_discharge_only = df_cycle[df_cycle['Current_A'] <= 0.05].copy()

        # Reset the time so each discharge cycle starts at Time = 0
        if not df_discharge_only.empty:
            first_time = df_discharge_only['Time_s'].iloc[0]
            df_discharge_only['Time_s'] = df_discharge_only['Time_s'] - first_time

            all_cycle_data.append(df_discharge_only)

        # Print progress every 100 cycles so you know the script hasn't frozen
        if cycle_num % 100 == 0:
            print(
                f"  ...Processed {cycle_num} / {total_recorded_cycles} cycles")

    if not all_cycle_data:
        print(" No valid discharge data found to save.")
        return

    # Combine all individual cycle DataFrames into one massive dataset
    print("\nConcatenating all data into memory... (Requires decent RAM)")
    final_df = pd.concat(all_cycle_data, ignore_index=True)

    # Reorder columns to put the most important ones first
    cols_order = ['Cycle_Number', 'Time_s', 'Voltage_V',
                  'Current_A', 'Temp_C', 'Qd', 'Qc', 't']
    final_df = final_df[cols_order]

    # Save to CSV
    print(f"Saving to {output_csv}...")
    final_df.to_csv(output_csv, index=False)
    print(f"\n SUCCESS! Massive ML training dataset saved to {output_csv}")
    print(f"Total rows extracted: {len(final_df):,}")


# --- EXECUTE EXTRACTION ---
MAT_FILE = "2017-05-12_batchdata_updated_struct_errorcorrect.mat"
CSV_FILE = "mit_all_discharge_ml_data.csv"

if __name__ == "__main__":
    extract_all_mit_discharge_data(MAT_FILE, CSV_FILE, cell_index=0)
