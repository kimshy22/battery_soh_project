import pandas as pd
from pathlib import Path

# =========================================================
# USER SETTINGS
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
STANFORD_DIR = PROJECT_ROOT / "data" / "standford"

# Battery assumptions for your Stanford LFP files
Q_RATED_AH = 2.5
HIGH_START_VOLTAGE = 3.30   # likely near full if starting above this
LOW_END_VOLTAGE = 2.60      # likely near cutoff if ending below this

ALLOWED_EXTENSIONS = {".xlsx", ".xls", ".csv"}


# =========================================================
# HELPER FUNCTION: FIND COLUMN BY KEYWORDS
# =========================================================
def find_column(columns, keywords):
    """
    Return the first column whose lowercase name contains
    one of the given keywords.
    """
    for col in columns:
        col_lower = str(col).strip().lower()
        for kw in keywords:
            if kw in col_lower:
                return col
    return None


# =========================================================
# HELPER FUNCTION: LOAD FILE
# =========================================================
def load_file(file_path):
    suffix = file_path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(file_path)
    elif suffix in {".xlsx", ".xls"}:
        return pd.read_excel(file_path)
    else:
        return None


# =========================================================
# CORE ANALYSIS FOR ONE FILE
# =========================================================
def analyze_stanford_file(file_path):
    try:
        df = load_file(file_path)

        if df is None or df.empty:
            return {
                "file": file_path.name,
                "status": "skipped",
                "reason": "Empty or unsupported file"
            }

        # Try to detect important columns automatically
        time_col = find_column(df.columns, ["time"])
        voltage_col = find_column(df.columns, ["voltage", "volt"])
        current_col = find_column(df.columns, ["current", "curr"])

        if not time_col or not voltage_col or not current_col:
            return {
                "file": file_path.name,
                "status": "skipped",
                "reason": f"Missing required columns. time={time_col}, voltage={voltage_col}, current={current_col}"
            }

        # Keep only required columns
        work = df[[time_col, voltage_col, current_col]].copy()
        work = work.rename(columns={
            time_col: "time_s",
            voltage_col: "voltage_v",
            current_col: "current_a"
        })

        # Convert to numeric
        work["time_s"] = pd.to_numeric(work["time_s"], errors="coerce")
        work["voltage_v"] = pd.to_numeric(work["voltage_v"], errors="coerce")
        work["current_a"] = pd.to_numeric(work["current_a"], errors="coerce")

        # Drop bad rows
        work = work.dropna(subset=["time_s", "voltage_v", "current_a"])
        work = work.sort_values("time_s").reset_index(drop=True)

        if len(work) < 5:
            return {
                "file": file_path.name,
                "status": "uncertain",
                "reason": "Too few valid rows"
            }

        # Compute time difference
        work["dt_s"] = work["time_s"].diff()
        work.loc[0, "dt_s"] = 0

        # Detect discharge sign automatically
        neg_count = (work["current_a"] < 0).sum()
        pos_count = (work["current_a"] > 0).sum()

        if neg_count == 0 and pos_count == 0:
            return {
                "file": file_path.name,
                "status": "uncertain",
                "reason": "No nonzero current values"
            }

        if neg_count >= pos_count:
            discharge = work[work["current_a"] < 0].copy()
            discharge_sign = "negative"
        else:
            discharge = work[work["current_a"] > 0].copy()
            discharge_sign = "positive"

        if discharge.empty:
            return {
                "file": file_path.name,
                "status": "uncertain",
                "reason": "No discharge rows found"
            }

        # Integrate current to get capacity
        discharge["delta_q_ah"] = discharge["current_a"].abs() * \
            discharge["dt_s"] / 3600.0
        q_window_ah = discharge["delta_q_ah"].sum()

        # Get start and end voltages of discharge
        start_v = float(discharge["voltage_v"].iloc[0])
        end_v = float(discharge["voltage_v"].iloc[-1])

        # Ratio to rated capacity
        capacity_ratio = q_window_ah / Q_RATED_AH

        # Heuristic decision
        score_full = 0
        score_partial = 0
        reasons = []

        # Check start voltage
        if start_v >= HIGH_START_VOLTAGE:
            score_full += 1
            reasons.append(f"starts high at {start_v:.3f} V")
        else:
            score_partial += 1
            reasons.append(f"does not start high ({start_v:.3f} V)")

        # Check end voltage
        if end_v <= LOW_END_VOLTAGE:
            score_full += 1
            reasons.append(f"ends near cutoff at {end_v:.3f} V")
        else:
            score_partial += 1
            reasons.append(f"does not end near cutoff ({end_v:.3f} V)")

        # Check capacity
        if 0.70 <= capacity_ratio <= 1.20:
            score_full += 1
            reasons.append(f"capacity near rated ({q_window_ah:.3f} Ah)")
        elif capacity_ratio < 0.50:
            score_partial += 1
            reasons.append(
                f"capacity much smaller than rated ({q_window_ah:.3f} Ah)")
        else:
            reasons.append(f"capacity intermediate ({q_window_ah:.3f} Ah)")

        # Final label
        if score_full >= 3:
            status = "likely full discharge"
        elif score_partial >= 2:
            status = "likely partial discharge"
        else:
            status = "uncertain"

        return {
            "file": file_path.name,
            "status": status,
            "reason": "; ".join(reasons),
            "rows": len(work),
            "discharge_rows": len(discharge),
            "discharge_sign": discharge_sign,
            "start_voltage_v": round(start_v, 4),
            "end_voltage_v": round(end_v, 4),
            "measured_capacity_ah": round(q_window_ah, 4),
            "capacity_ratio_to_rated": round(capacity_ratio, 4)
        }

    except Exception as e:
        return {
            "file": file_path.name,
            "status": "error",
            "reason": str(e)
        }


# =========================================================
# MAIN
# =========================================================
def main():
    if not STANFORD_DIR.exists():
        print(f"Stanford folder not found: {STANFORD_DIR}")
        return

    files = [
        f for f in STANFORD_DIR.rglob("*")
        if f.is_file() and f.suffix.lower() in ALLOWED_EXTENSIONS
    ]

    if not files:
        print("No Stanford data files found.")
        return

    results = []
    for file_path in files:
        result = analyze_stanford_file(file_path)
        results.append(result)

    results_df = pd.DataFrame(results)

    print("\n===== STANFORD DISCHARGE TYPE CHECK =====")
    print(results_df.to_string(index=False))

    output_dir = PROJECT_ROOT / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "stanford_discharge_check.csv"
    results_df.to_csv(output_file, index=False)

    print(f"\nSaved results to: {output_file}")


if __name__ == "__main__":
    main()
