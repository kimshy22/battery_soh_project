import pandas as pd
import numpy as np
import os

# =====================================================================
# 1. Core BMS Functions
# =====================================================================


def get_lfp_soc_from_ocv(voltage):
    """Returns the State of Charge (%) based on a rested LFP Open-Circuit Voltage."""
    ocv_curve = [2.50, 3.00, 3.10, 3.20, 3.28, 3.30, 3.32, 3.33, 3.40]
    soc_curve = [0.0,  10.0, 15.0, 20.0, 50.0, 70.0, 80.0, 90.0, 100.0]
    return np.interp(voltage, ocv_curve, soc_curve)


def normalize_resistance_to_25c(r_measured, temp_c):
    """Normalizes the measured internal resistance to 25C using the Arrhenius equation."""
    E_a = 40000
    R_g = 8.314
    t_ref_k = 25.0 + 273.15
    t_meas_k = temp_c + 273.15
    multiplier = np.exp((E_a / R_g) * ((1 / t_ref_k) - (1 / t_meas_k)))
    return r_measured * multiplier

# =====================================================================
# 2. Data Logging & Display Functions
# =====================================================================


def log_raw_sensor_data(df, filename="raw_sensor_log.csv"):
    """Saves the raw sensor data to a CSV. Appends if file exists."""
    write_header = not os.path.exists(filename)
    df.to_csv(filename, mode='a', header=write_header, index=False)


def log_cycle_soh(cycle_number, metrics, filename="soh_history_log.csv"):
    """Appends the final SOH and the physical proof to the history CSV."""
    new_soh = pd.DataFrame({
        'Cycle': [cycle_number],
        'Step_dV_V': [round(metrics['dv'], 3)],
        'Step_dI_A': [round(metrics['di'], 3)],
        'Calc_R_Ohms': [round(metrics['r_measured'], 5)],
        'Temp_C': [round(metrics['temp_c'], 1)],
        'SOH_Resistance_%': [round(metrics['soh_r'], 2)],
        'SOH_Capacity_%': [round(metrics['soh_c'], 2)],
        'Final_Blended_SOH_%': [round(metrics['final_soh'], 2)]
    })

    write_header = not os.path.exists(filename)
    new_soh.to_csv(filename, mode='a', header=write_header, index=False)
    print(f"[LOG] Cycle {cycle_number} detailed results saved to {filename}")


def display_recent_history(filename="soh_history_log.csv", rows=3):
    """Fetches and displays the most recent SOH estimations."""
    if os.path.exists(filename):
        df_history = pd.read_csv(filename)
        print(
            f"\n--- LAST {min(rows, len(df_history))} CYCLES SOH HISTORY ---")
        # .to_string() neatly formats the table for the terminal
        print(df_history.tail(rows).to_string(index=False))
        print(
            "---------------------------------------------------------------------------\n")
    else:
        print("No history found yet.")

# =====================================================================
# 3. Main SOH/SOC Pipeline
# =====================================================================


def run_bms_pipeline(df, q_rated, r_initial_25c):
    """Processes a full charge/discharge sequence to estimate SOC and SOH."""
    print("\n--- INITIATING RASPBERRY PI BMS PIPELINE ---")

    df = df.sort_values(by='Time_s').reset_index(drop=True)
    df['dt'] = df['Time_s'].diff().fillna(0)

    # Defaults in case the active phase is missing or too clean
    dV_step = 0.0
    dI_step = 0.0
    r_measured = r_initial_25c
    temp_at_step = 25.0
    soh_r = 100.0

    # Phase 1: Initialize SOC from Rested OCV (Using head(2) for manual input ease)
    rest_initial = df[df['Current_A'].abs() < 0.05].head(2)
    if rest_initial.empty:
        print("ERROR: No initial rest period found (Requires Current = 0).")
        # Return fallback metrics dictionary
        return {'dv': 0, 'di': 0, 'r_measured': r_initial_25c, 'temp_c': 25, 'soh_r': 100, 'soh_c': 100, 'final_soh': 100}

    v_start_rested = rest_initial['Voltage_V'].mean()
    soc_start = get_lfp_soc_from_ocv(v_start_rested)
    print(
        f"[SOC] Initial Rested Voltage: {v_start_rested:.3f}V -> Starting SOC: {soc_start:.1f}%")

    # Phase 2: Dynamic SOC & Resistance Calculation
    active_mask = df['Current_A'] < -0.1
    df_active = df[active_mask].copy()

    df['Ah_Discharged'] = (df['Current_A'] * df['dt']) / 3600
    cumulative_ah = df['Ah_Discharged'].cumsum().abs()

    df_active['dI'] = df_active['Current_A'].diff().fillna(0)

    if not df_active.empty:
        step_idx = df_active['dI'].abs().idxmax()
        dI_step = df_active.loc[step_idx, 'dI']
        # Need to check if step_idx - 1 exists in the main dataframe
        if step_idx > 0:
            dV_step = df.loc[step_idx, 'Voltage_V'] - \
                df.loc[step_idx - 1, 'Voltage_V']
        else:
            dV_step = 0.0

        temp_at_step = df.loc[step_idx, 'Temp_C']

        if abs(dI_step) > 0.5:
            r_measured = abs(
                dV_step / dI_step) if dI_step != 0 else r_initial_25c
            r_normalized = normalize_resistance_to_25c(
                r_measured, temp_at_step)
            soh_r = (r_initial_25c / r_normalized) * 100
        else:
            print("[WARNING] No significant load step found. Using default R.")
            r_measured = r_initial_25c
    else:
        print("[WARNING] No active discharge found.")

    soh_r = min(soh_r, 100.0)

    # Phase 3: Capacity-Based SOH (Using tail(2) for manual input ease)
    rest_final = df[df['Current_A'].abs() < 0.05].tail(2)
    if rest_final.empty:
        print("ERROR: No final rest period found (Requires Current = 0).")
        soh_c = soh_r  # Fallback

    v_end_rested = rest_final['Voltage_V'].mean()
    soc_end_ocv = get_lfp_soc_from_ocv(v_end_rested)
    print(
        f"[SOC] Final Rested Voltage:   {v_end_rested:.3f}V -> Ending OCV SOC: {soc_end_ocv:.1f}%")

    total_ah_discharged = cumulative_ah.iloc[-1]
    delta_soc = soc_start - soc_end_ocv

    if delta_soc >= 5.0:  # Threshold lowered for manual testing
        q_actual = total_ah_discharged / (delta_soc / 100.0)
        soh_c = (q_actual / q_rated) * 100
        soh_c = min(soh_c, 100.0)
    else:
        print(
            f"[WARNING] Delta SOC ({delta_soc:.1f}%) is too small for accurate SOH_C calculation.")
        soh_c = soh_r
        q_actual = q_rated

    # Phase 4: Neutralized SOH Blending
    # Weights maintained at 0.65 and 0.35 to equal 1.0 (100%)
    final_soh = (0.65 * soh_r) + (0.35 * soh_c)

    print("\n==================================================")
    print("             CYCLE SUMMARY & SOH REPORT           ")
    print("==================================================")
    print(f"Total Discharged:      {total_ah_discharged:.3f} Ah")
    print(f"Delta SOC (OCV):       {delta_soc:.1f}%")
    print("--------------------------------------------------")
    print(f"Step Voltage Drop:     {dV_step:.3f} V")
    print(f"Step Current Drop:     {dI_step:.3f} A")
    print(f"Measured Temp @ Step:  {temp_at_step:.1f} °C")
    print(f"Calculated Resistance: {r_measured:.5f} Ohms")
    print("--------------------------------------------------")
    print(f"SOH (Resistance-based): {soh_r:.2f}% (Weight: 65%)")
    print(f"SOH (Capacity-based):   {soh_c:.2f}% (Weight: 35%)")
    print(f"FINAL BLENDED SOH:      {final_soh:.2f}%")
    print("==================================================")

    # Package all metrics to send back to the logging function
    metrics = {
        'dv': dV_step,
        'di': dI_step,
        'r_measured': r_measured,
        'temp_c': temp_at_step,
        'soh_r': soh_r,
        'soh_c': soh_c,
        'final_soh': final_soh
    }

    return metrics

# =====================================================================
# 4. Interactive Simulator (User acts as the Sensors)
# =====================================================================


def interactive_sensor_loop(q_rated, r_initial_25c):
    print("\n" + "="*50)
    print(" 🔋 RASPBERRY PI SENSOR SIMULATOR TERMINAL 🔋 ")
    print("="*50)
    print("Instructions to simulate a full cycle:")
    print(" 1. Start with Current = 0 to get initial rested voltage.")
    print(" 2. Enter negative Current to simulate discharging the battery.")
    print(" 3. End with Current = 0 to get final rested voltage.")
    print("\nType 'c' at any prompt to finish the cycle and run the SOH math.")
    print("Type 'q' at any prompt to exit the program entirely.")
    print("-" * 50)

    session_data = []
    time_counter = 0  # Simulated time

    while True:
        print(f"\n[ Simulated Time: {time_counter} seconds ]")

        # Get Voltage
        v_input = input("Enter Voltage (V): ")
        if v_input.lower() == 'c':
            break
        if v_input.lower() == 'q':
            return

        # Get Current
        i_input = input("Enter Current (A) [- discharge, 0 rest]: ")
        if i_input.lower() == 'c':
            break
        if i_input.lower() == 'q':
            return

        # Get Temperature
        t_input = input("Enter Temperature (°C): ")
        if t_input.lower() == 'c':
            break
        if t_input.lower() == 'q':
            return

        try:
            v = float(v_input)
            i = float(i_input)
            t = float(t_input)

            # 1. Log to the raw CSV immediately
            reading_df = pd.DataFrame({'Time_s': [time_counter], 'Voltage_V': [
                                      v], 'Current_A': [i], 'Temp_C': [t]})
            log_raw_sensor_data(reading_df)

            # 2. Keep it in memory so we can run the math at the end
            session_data.append(
                {'Time_s': time_counter, 'Voltage_V': v, 'Current_A': i, 'Temp_C': t})
            print(
                f"--> Logged: {v}V, {i}A, {t}°C. (Saved to raw_sensor_log.csv)")

            # Advance time by 30 seconds for the next reading
            time_counter += 30

        except ValueError:
            print(" Invalid input! Please enter numbers only (e.g., 3.3).")

    # When the user types 'c', break the loop and process the data!
    if len(session_data) > 0:
        df_session = pd.DataFrame(session_data)

        # Run the pipeline and capture the dictionary of metrics
        cycle_metrics = run_bms_pipeline(df_session, q_rated, r_initial_25c)

        # Ask what cycle number this is so we can log it correctly
        cycle_input = input(
            "\nEnter Cycle Number to save this history (e.g., 1): ")
        try:
            cycle_num = int(cycle_input)
        except ValueError:
            cycle_num = 1  # Default if they type a letter

        # Pass the whole dictionary to the logger!
        log_cycle_soh(cycle_num, cycle_metrics)
        display_recent_history()
    else:
        print("No sensor data was entered!")


# --- EXECUTE THE SIMULATOR ---
# Using standard 2.5Ah LFP with 20 mOhm internal resistance
Q_RATED_SPEC = 2.5
R_INITIAL_SPEC = 0.02

if __name__ == "__main__":
    interactive_sensor_loop(Q_RATED_SPEC, R_INITIAL_SPEC)
