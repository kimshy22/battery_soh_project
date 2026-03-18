import pandas as pd


def extract_single_test():
    # 1. Load the big file
    big_file_path = r"C:\Users\SUZZIE\battery_soh_project\src\mit_all_discharge_ml_data.csv"
    df = pd.read_csv(big_file_path)

    # 2. Grab exactly the first 50 rows (This equals ONE test for the AI)
    single_test_df = df.head(50)

    # 3. Save it as a tiny test file
    small_file_path = r"C:\Users\SUZZIE\battery_soh_project\data\single_validation_test.csv"
    single_test_df.to_csv(small_file_path, index=False)

    print(f"✅ Created single validation test file!")
    print(f"File saved to: {small_file_path}")


if __name__ == "__main__":
    extract_single_test()
