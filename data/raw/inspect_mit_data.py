import h5py
import pandas as pd


def inspect_mit_data(mat_filename, cell_index=0, cycle_num=1):
    print(f"🔍 Opening {mat_filename} for inspection...\n")

    try:
        f = h5py.File(mat_filename, 'r')
    except Exception as e:
        print(f"Error opening file: {e}")
        return

    # Navigate to the requested cell
    batch = f['batch']
    cycles_ref = batch['cycles'][cell_index, 0]
    cycles = f[cycles_ref]

    # Get all the available column names (fields) recorded by MIT
    available_columns = list(cycles.keys())
    print(
        f"✅ Found the following columns in the dataset: {available_columns}\n")

    data_dict = {}

    # Loop through every available column and grab the first 10 data points
    for col in available_columns:
        try:
            # Extract the data array using the HDF5 reference
            data_array = f[cycles[col][cycle_num, 0]][:].flatten()

            # Grab only the first 10 rows
            data_dict[col] = data_array[:10]
        except Exception as e:
            print(
                f"⚠️ Could not extract column '{col}' (Might not be a standard array)")

    # Convert to a Pandas DataFrame to print a beautiful table
    if data_dict:
        df_preview = pd.DataFrame(data_dict)
        print(
            f"--- First 10 Rows of Data for Cell #{cell_index + 1}, Cycle #{cycle_num} ---")

        # We use .to_string() so Pandas doesn't hide any columns
        print(df_preview.to_string())
    else:
        print("No data could be extracted for preview.")


# --- EXECUTE INSPECTION ---
# Replace this with the exact name of your downloaded MIT .mat file!
MAT_FILE = "2017-05-12_batchdata_updated_struct_errorcorrect.mat"

if __name__ == "__main__":
    inspect_mit_data(MAT_FILE)
