import pandas as pd
import numpy as np
import joblib
import gzip
import tflite_runtime.interpreter as tflite


def run_validation():
    print("--- RASPBERRY PI AI VALIDATION TEST ---")

    # 1. Load Scaler
    print("Loading scaler...")
    with gzip.open('models/scaler.gz', 'rb') as f:
        scaler = joblib.load(f)

    # 2. Load AI Model
    print("Loading TFLite model...")
    interpreter = tflite.Interpreter(model_path='models/soh_estimator.tflite')
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # 3. Load the exact 50 rows of test data
    print("Loading the 50-row validation data...")
    df = pd.read_csv('data/single_validation_test.csv')
    true_capacity = df['Qd'].iloc[0]

    # 4. Scale and reshape the data for the LSTM
    features = ['Voltage_V', 'Current_A', 'Temp_C']
    raw_readings = df[features].values
    scaled_readings = scaler.transform(raw_readings)

    # The AI needs the shape to be (1 sequence, 50 rows, 3 features)
    input_data = np.expand_dims(scaled_readings, axis=0).astype(np.float32)

    # 5. Run the Prediction
    print("AI is calculating...")
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    estimated_capacity = interpreter.get_tensor(
        output_details[0]['index'])[0][0]

    # 6. Print Results
    print("\n========================================")
    print(f"True MIT Capacity : {true_capacity:.4f} Ah")
    print(f"Pi AI Estimate    : {estimated_capacity:.4f} Ah")

    error = abs(true_capacity - estimated_capacity) / true_capacity * 100
    print(f"Error Margin      : {error:.2f}%")
    print("========================================")


if __name__ == "__main__":
    run_validation()
