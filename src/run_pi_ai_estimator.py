import pandas as pd
import os


def extract_test_data():
    print("--- Starting Data Extraction ---")

    # 1. Define where the big file is, and where the small file will go
    big_file_path = "../src/mit_all_discharge_ml_data.csv"
    small_file_path = "../data/mit_100_cycles.csv"

    # 2. Load the massive dataset
    print(f"Loading massive dataset from: {big_file_path}")
    df = pd.read_csv(big_file_path)

    # 3. Filter for only the first 100 cycles
    print("Slicing out the first 100 cycles...")
    df_100 = df[df['Cycle_Number'] <= 100]

    # 4. Save the new, small dataset
    df_100.to_csv(small_file_path, index=False)

    print(f"✅ Success! Small dataset saved to: {small_file_path}")
    print(f"Rows extracted: {len(df_100)}")


if __name__ == "__main__":
    extract_test_data()
