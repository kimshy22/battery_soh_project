import time
import pandas as pd
import numpy as np
import os  # Added os module to check if CSV files already exist

# 1. Core BMS Functions


def get_lfp_soc_from_ocv(voltage):
    """
    Returns the State of Charge (%) based on a rested LFP Open-Circuit Voltage.
    Uses linear interpolation between known curve points.
    """
    # Standard LFP rested OCV curve (adjust based on your specific cell datasheet)
    ocv_curve = [2.50, 3.00, 3.10, 3.20, 3.28, 3.30, 3.32, 3.33, 3.40]
    soc_curve = [0.0,  10.0, 15.0, 20.0, 50.0, 70.0, 80.0, 90.0, 100.0]

    return np.interp(voltage, ocv_curve, soc_curve)


def normalize_resistance_to_25c(r_measured, temp_c):
    """
    Normalizes the measured internal resistance to 25C using the Arrhenius equation.
    LFP resistance spikes significantly at lower temperatures.
    """
    E_a = 40000  # Activation energy for LFP charge transfer (J/mol) - Approximate
    R_g = 8.314  # Universal gas constant (J/(mol*K))

    t_ref_k = 25.0 + 273.15
    t_meas_k = temp_c + 273.15

    # Arrhenius multiplier
    multiplier = np.exp((E_a / R_g) * ((1 / t_ref_k) - (1 / t_meas_k)))
    r_normalized = r_measured * multiplier

    return r_normalized


# 2. Data Logging & Display Functions

def log_raw_sensor_data(df, filename="raw_sensor_log.csv"):
    """Saves the raw sensor data to a CSV. Appends if file exists."""
    write_header = not os.path.exists(filename)
    df.to_csv(filename, mode='a', header=write_header, index=False)
    print(f"[LOG] Saved {len(df)} raw sensor readings to {filename}")


def log_cycle_soh(cycle_number, soh_r, soh_c, final_soh, filename="soh_history_log.csv"):
    """Appends the final calculated cycle SOH to the history CSV."""
    new_soh = pd.DataFrame({
        'Cycle': [cycle_number],
        'SOH_Resistance_%': [round(soh_r, 2)],
        'SOH_Capacity_%': [round(soh_c, 2)],
        'Final_Blended_SOH_%': [round(final_soh, 2)]
    })
    write_header = not os.path.exists(filename)
    new_soh.to_csv(filename, mode='a', header=write_header, index=False)
    print(f"[LOG] Cycle {cycle_number} SOH results saved to {filename}")


def display_recent_history(filename="soh_history_log.csv", rows=3):
    """Fetches and displays the most recent SOH estimations."""
    if os.path.exists(filename):
        df_history = pd.read_csv(filename)
        print(
            f"\n--- LAST {min(rows, len(df_history))} CYCLES SOH HISTORY ---")
        print(df_history.tail(rows).to_string(index=False))
        print("--------------------------------------\n")
    else:
        print("No history found yet.")


# 3. Main SOH/SOC Pipeline

def run_bms_pipeline(df, q_rated, r_initial_25c):
    """
    Processes a full charge/discharge sequence to estimate SOC and SOH.
    """
    print("\n--- INITIATING RASPBERRY PI BMS PIPELINE ---")

    # Ensure sequential time
    df = df.sort_values(by='Time_s').reset_index(drop=True)
    df['dt'] = df['Time_s'].diff().fillna(0)

    # Phase 1: Initialize SOC from Rested OCV
    rest_initial = df[df['Current_A'].abs() < 0.05].head(30)
    if rest_initial.empty:
        print("ERROR: No initial rest period found to establish starting OCV.")
        return

    v_start_rested = rest_initial['Voltage_V'].mean()
    soc_start = get_lfp_soc_from_ocv(v_start_rested)
    print(
        f"[SOC] Initial Rested Voltage: {v_start_rested:.3f}V -> Starting SOC: {soc_start:.1f}%")

    # Phase 2: Dynamic SOC & Resistance Calculation (Active)
    active_mask = df['Current_A'] < -0.1
    df_active = df[active_mask].copy()

    # Coulomb Counting for dynamic SOC tracking
    df['Ah_Discharged'] = (df['Current_A'] * df['dt']) / 3600
    cumulative_ah = df['Ah_Discharged'].cumsum().abs()

    # Track SOC throughout the cycle
    df['Dynamic_SOC'] = soc_start - ((cumulative_ah / q_rated) * 100)

    # --- Resistance Calculation (DCIR) ---
    df_active['dI'] = df_active['Current_A'].diff().fillna(0)

    # Find the largest load step
    step_idx = df_active['dI'].abs().idxmax()
    dI_step = df_active.loc[step_idx, 'dI']
    dV_step = df.loc[step_idx, 'Voltage_V'] - df.loc[step_idx - 1, 'Voltage_V']
    temp_at_step = df.loc[step_idx, 'Temp_C']

    if abs(dI_step) > 0.5:  # Ensure the step is significant enough
        r_measured = abs(dV_step / dI_step)
        r_normalized = normalize_resistance_to_25c(r_measured, temp_at_step)
        soh_r = (r_initial_25c / r_normalized) * 100
    else:
        print("[WARNING] No significant load step found. Using default R.")
        r_normalized = r_initial_25c
        soh_r = 100.0

    # Cap at 100%
    soh_r = min(soh_r, 100.0)

    # Phase 3: Capacity-Based SOH (Post-Rest)
    rest_final = df[df['Current_A'].abs() < 0.05].tail(30)
    v_end_rested = rest_final['Voltage_V'].mean()
    soc_end_ocv = get_lfp_soc_from_ocv(v_end_rested)

    print(
        f"[SOC] Final Rested Voltage:   {v_end_rested:.3f}V -> Ending OCV SOC: {soc_end_ocv:.1f}%")

    # Total capacity discharged during the cycle
    total_ah_discharged = cumulative_ah.iloc[-1]

    # The change in SOC based purely on the rested OCV lookups
    delta_soc = soc_start - soc_end_ocv

    if delta_soc >= 10.0:
        # Extrapolate full capacity
        q_actual = total_ah_discharged / (delta_soc / 100.0)
        soh_c = (q_actual / q_rated) * 100
        soh_c = min(soh_c, 100.0)
    else:
        print(
            f"[WARNING] Delta SOC ({delta_soc:.1f}%) is too small for accurate SOH_C calculation.")
        soh_c = soh_r  # Fallback
        q_actual = q_rated

    # Phase 4: Neutralized SOH Blending
    final_soh = (0.65 * soh_r) + (0.35 * soh_c)

    print("  CYCLE SUMMARY & SOH REPORT  ")
    print(f"Total Discharged:      {total_ah_discharged:.3f} Ah")
    print(f"Delta SOC (OCV):       {delta_soc:.1f}%")
    print(f"Extrapolated Capacity: {q_actual:.3f} Ah")
    print(f"Measured Temp @ Step:  {temp_at_step:.1f} °C")
    print(f"Measured Resistance:   {r_measured:.5f} Ohms")
    print(f"Normalized R (25°C):   {r_normalized:.5f} Ohms")
    print(f"SOH (Resistance-based): {soh_r:.2f}% (Weight: 65%)")
    print(f"SOH (Capacity-based):   {soh_c:.2f}% (Weight: 35%)")
    print(f"FINAL BLENDED SOH:      {final_soh:.2f}%")

    # =========================================================
    # THE FIX: Returning the metrics as a dictionary
    # =========================================================
    metrics = {
        'soh_r': soh_r,
        'soh_c': soh_c,
        'final_soh': final_soh
    }
    return metrics


# =====================================================================
# 4. Hardware-in-the-Loop (HIL) CSV Simulator
# =====================================================================

def run_pi_dataset_simulation(csv_filename, q_rated, r_initial_25c):
    print("\n" + "="*60)
    print("  RASPBERRY PI HARDWARE-IN-THE-LOOP (HIL) SIMULATOR  ")
    print("="*60)
    print(f"Loading massive dataset: {csv_filename}...")

    try:
        mit_data = pd.read_csv(csv_filename)
    except FileNotFoundError:
        print(
            f" ERROR: Could not find {csv_filename}. Make sure it is in the same folder!")
        return

    grouped_cycles = mit_data.groupby('Cycle_Number')

    cycles_to_run = 10
    cycles_processed = 0

    print(f" Dataset loaded! Found {len(grouped_cycles)} total records.")
    print(
        f"Searching for the first {cycles_to_run} valid laboratory cycles...\n")
    time.sleep(1)

    for cycle_num, cycle_df in grouped_cycles:
        if cycles_processed >= cycles_to_run:
            break

        true_q = cycle_df['Qd'].iloc[0]

        # Skip invalid lab data, but print a tiny dot so we know it's searching
        if true_q <= 0.1:
            print(f"Skipping Cycle {cycle_num} (No MIT Capacity Data)")
            continue

        cycles_processed += 1

        print(f"\n{'='*40}")
        print(f"   PROCESSING CYCLE #{cycle_num}")
        print(f"{'='*40}")

        log_raw_sensor_data(cycle_df.head(5))

        cycle_metrics = run_bms_pipeline(cycle_df, q_rated, r_initial_25c)

        estimated_q = (cycle_metrics['soh_c'] / 100.0) * q_rated
        accuracy_error = abs(true_q - estimated_q) / true_q * 100.0

        print("\n---  MIT LABORATORY VALIDATION  ---")
        print(f"MIT True Capacity (Qd): {true_q:.4f} Ah")
        print(f"Raspberry Pi Estimated: {estimated_q:.4f} Ah")
        print(f"Algorithm Error Margin: {accuracy_error:.2f}%")
        print("---------------------------------------")

        log_cycle_soh(
            cycle_number=cycle_num,
            soh_r=cycle_metrics['soh_r'],
            soh_c=cycle_metrics['soh_c'],
            final_soh=cycle_metrics['final_soh']
        )
        time.sleep(1)

    print("\n HIL SIMULATION COMPLETE! ")
    print(f"Successfully validated {cycles_processed} cycles.")


# --- EXECUTE THE HIL SIMULATION ---
Q_RATED_SPEC = 1.1      # Rated capacity is 1.1 Ah
R_INITIAL_SPEC = 0.0055  # Initial resistance 5.5 mOhm

CSV_FILE = "mit_all_discharge_ml_data.csv"

if __name__ == "__main__":
    # Ensure the simulation only runs once
    run_pi_dataset_simulation(CSV_FILE, Q_RATED_SPEC, R_INITIAL_SPEC)

    # Force the script to stop here
    print("\n[SYSTEM] Simulation session finished. Standing by.")
    import sys
    sys.exit()
