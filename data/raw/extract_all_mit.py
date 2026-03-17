import h5py
import pandas as pd
import numpy as np


def extract_mit_data_with_qd(mat_filename, output_csv, cell_index=0):
    print(f"Opening {mat_filename}... Please wait.")
    try:
        f = h5py.File(mat_filename, 'r')
    except Exception as e:
        print(f"Error opening file: {e}")
        return

    batch = f['batch']

    # 1. Access the raw sensor data
    cycles_ref = batch['cycles'][cell_index, 0]
    cycles = f[cycles_ref]

    # 2. Access the TRUE laboratory capacity data
    summary_ref = batch['summary'][cell_index, 0]
    summary = f[summary_ref]

    # Extract the true capacity array (Qd)
    true_qd_array = summary['QDischarge'][0, :]
    total_recorded_cycles = cycles['I'].shape[0]

    print(f"Found {total_recorded_cycles} cycles.")
    print("Extracting raw sensors AND true laboratory capacity (Qd)...")

    all_cycle_data = []

    for cycle_num in range(1, total_recorded_cycles):
        try:
            # Extract raw sensor arrays
            i_data = f[cycles['I'][cycle_num, 0]][:].flatten()
            v_data = f[cycles['V'][cycle_num, 0]][:].flatten()
            t_data = f[cycles['T'][cycle_num, 0]][:].flatten()
            time_data = f[cycles['t'][cycle_num, 0]][:].flatten()

            # Map the true capacity to this specific cycle
            try:
                true_qd = true_qd_array[cycle_num]
            except IndexError:
                true_qd = 0.0

            df_cycle = pd.DataFrame({
                'Time_s': time_data * 60.0,  # Convert to seconds
                'Voltage_V': v_data,
                'Current_A': i_data,
                'Temp_C': t_data
            })

            df_cycle['Cycle_Number'] = cycle_num
            df_cycle['Qd'] = true_qd

            #  Keep everything except charging. We NEED the 0.0A rest periods!
            df_discharge = df_cycle[df_cycle['Current_A'] <= 0.05].copy()
            if not df_discharge.empty:
                # Reset time to start at 0 for each cycle
                df_discharge['Time_s'] = df_discharge['Time_s'] - \
                    df_discharge['Time_s'].iloc[0]
                all_cycle_data.append(df_discharge)

            if cycle_num % 100 == 0:
                print(
                    f"  ...Processed {cycle_num} / {total_recorded_cycles} cycles")

        except Exception as e:
            # Skip any corrupted MIT records
            continue

    if not all_cycle_data:
        print("No valid data found.")
        return

    print("\nCombining all cycles...")
    final_df = pd.concat(all_cycle_data, ignore_index=True)

    cols_order = ['Cycle_Number', 'Time_s',
                  'Voltage_V', 'Current_A', 'Temp_C', 'Qd']
    final_df = final_df[cols_order]

    print(f"Saving perfect dataset to {output_csv}...")
    final_df.to_csv(output_csv, index=False)
    print("✅ EXTRACTION COMPLETE. The dataset is ready for HIL and LSTM.")


# --- EXECUTE EXTRACTION ---
# Ensure this matches your actual .mat file name exactly
MAT_FILE = "2017-05-12_batchdata_updated_struct_errorcorrect.mat"
CSV_FILE = "mit_all_discharge_ml_data.csv"

if __name__ == "__main__":
    extract_mit_data_with_qd(MAT_FILE, CSV_FILE, cell_index=0)
