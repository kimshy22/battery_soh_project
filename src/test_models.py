from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
import gzip
import joblib
import numpy as np
import pandas as pd
import os
import warnings

# --- SILENCE ALL WARNINGS ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')


def load_estimator():
    print("\n[LOADING ESTIMATOR - Real-Time Diagnostics]")
    with gzip.open('models/feature_scaler.gz', 'rb') as f:
        feature_scaler = joblib.load(f)
    with gzip.open('models/target_scaler.gz', 'rb') as f:
        target_scaler = joblib.load(f)

    model = Sequential()
    model.add(Input(shape=(50, 4)))
    model.add(LSTM(units=64, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=32, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=16))
    model.add(Dense(units=1))

    model.load_weights('models/lfp_soh_weights.weights.h5')
    print(" Estimator successfully loaded!")
    return model, feature_scaler, target_scaler


def load_predictor():
    print("\n[LOADING PREDICTOR - Remaining Useful Life (RUL)]")
    with gzip.open('models/rul_scaler.gz', 'rb') as f:
        rul_scaler = joblib.load(f)

    model = Sequential()
    model.add(Input(shape=(20, 1)))
    model.add(LSTM(units=32, return_sequences=False))
    model.add(Dense(units=8))
    model.add(Dense(units=1))

    model.load_weights('models/rul_predictor.weights.h5')
    print(" Predictor successfully loaded!")
    return model, rul_scaler


def main_menu():
    print("========================================")
    print("   LFP BATTERY AI TROUBLESHOOTING TOOL  ")
    print("========================================")
    print("1. Test Estimator Model (Real CSV Data)")
    print("2. Test Predictor Model (Real CSV Data)")
    print("3. Exit")

    choice = input("\nEnter the number of the model to test: ")

    if choice == '1':
        import time  # Added for the live dashboard delay
        import csv
        import os
        est_model, f_scaler, t_scaler = load_estimator()
        print("\nLoading real sensor data from data/input/ folder...")

        csv_path = 'data/input/mit_all_discharge_ml_data.csv'
        # NEW: The path for our real-time diagnostic log
        log_path = 'data/logs/estimator_log.csv'
        try:
            df = pd.read_csv(csv_path)
            features = ['Voltage_V', 'Current_A', 'Temp_C', 'Time_s']

            # Grab the last 150 rows of the CSV so we have room to "slide"
            chunk_of_data = df[features].values[-150:]

            print("\n========================================")
            print("  STARTING LIVE SLIDING WINDOW SIMULATOR")
            print("========================================")
            # Check if the log file exists. If not, write the headers.
            file_exists = os.path.isfile(log_path)
            with open(log_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                if not file_exists:
                    # Writing our column headers
                    writer.writerow(
                        ['Simulation_Step', 'Estimated_SOH_Percentage'])
            # Slide the window forward 100 times
            for step in range(100):
                # The Sliding Math: Grab 50 rows, moving forward by 1 each loop
                current_window = chunk_of_data[step: step + 50]

                # Scale the 50 rows
                scaled_readings = f_scaler.transform(current_window)
                input_data = np.expand_dims(
                    scaled_readings, axis=0).astype(np.float32)

                # AI makes a fresh prediction
                prediction_scaled = est_model.predict(input_data, verbose=0)
                real_soh = t_scaler.inverse_transform(prediction_scaled)[0][0]

                # Print the live dashboard output
                print(
                    f"Step {step+1:03d}/100 | Rows [{step:03d} to {step+50:03d}] | 🔋 SOH: {real_soh:.2f} %")
                # 3. ATOMIC LOGGING! Open, write, and close instantly for every step
                with open(log_path, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([step + 1, round(float(real_soh), 2)])
                # Wait 0.5 seconds before the next loop so you can watch it!
                time.sleep(0.5)

            print("========================================")
            print(" Simulation Complete.")

        except FileNotFoundError:
            print(f"\n ERROR: Could not find '{csv_path}'.")

    elif choice == '2':
        import time
        import csv
        import os

        pred_model, r_scaler = load_predictor()
        print("\nLoading raw MIT data from data/input/ folder...")
        csv_path = 'data/input/mit_all_discharge_ml_data.csv'
        log_path = 'data/logs/prediction_log.csv'

        try:
            df = pd.read_csv(csv_path)
            if 'SOH' not in df.columns and 'Qd' in df.columns:
                df['SOH'] = (df['Qd'] / 1.1) * 100

            # ---> THE FIX: Changed to Cycle_Number <---
            if 'Cycle_Number' in df.columns:

                # 1. Find the last actual cycle number
                last_known_cycle = int(df['Cycle_Number'].max())

                # Group by Cycle_Number instead of Cycle_Index
                cycle_data = df.groupby('Cycle_Number')[
                    'SOH'].max().values.reshape(-1, 1)
                raw_cycles = cycle_data[-20:]

                if len(raw_cycles) < 20:
                    print(f" ERROR: Found {len(raw_cycles)} cycles.")
                else:
                    scaled_cycles = r_scaler.transform(raw_cycles)
                    current_window = np.reshape(
                        scaled_cycles, (1, 20, 1)).astype(np.float32)

                    print(
                        f"Last known physical cycle is Cycle {last_known_cycle}.")
                    print("\n========================================")
                    future_steps = int(
                        input("How many future cycles do you want to predict? (e.g., 10, 50): "))
                    print("========================================")
                    print(f"  FORECASTING & LOGGING TO: {log_path}")
                    print("========================================")

                    file_exists = os.path.isfile(log_path)
                    with open(log_path, mode='a', newline='') as file:
                        writer = csv.writer(file)
                        if not file_exists:
                            writer.writerow(
                                ['Cycle_Number', 'Predicted_SOH_Percentage'])

                        for step in range(future_steps):
                            target_cycle = last_known_cycle + step + 1

                            prediction_scaled = pred_model.predict(
                                current_window, verbose=0)
                            real_future_soh = r_scaler.inverse_transform(prediction_scaled)[
                                0][0]

                            print(
                                f"Cycle {target_cycle:04d} | 🔋 Predicted SOH: {real_future_soh:.2f} %")
                            writer.writerow(
                                [target_cycle, round(float(real_future_soh), 2)])

                            pred_reshaped = np.reshape(
                                prediction_scaled, (1, 1, 1))
                            current_window = np.append(
                                current_window[:, 1:, :], pred_reshaped, axis=1)
                            time.sleep(0.1)

                    print("========================================")
                    print(" Forecasting Complete and Log Saved.")
            else:
                print(
                    "\n ERROR: Could not find 'Cycle_Number' column. Please check your CSV headers!")
        except FileNotFoundError:
            print(f"\n ERROR: Could not find '{csv_path}'.")
    elif choice == '3':
        print("Exiting...")
    else:
        print("Invalid choice. Please run the script again.")


if __name__ == "__main__":
    main_menu()
