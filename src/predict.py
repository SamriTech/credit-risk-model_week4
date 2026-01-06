import mlflow.pyfunc
import pandas as pd

def run_prediction(data_dict):
    # 1. Configuration - same as your API
    MODEL_NAME = "High_Risk_Predictor_Model_Final"
    MODEL_URI = f"models:/{MODEL_NAME}/latest"

    # 2. Load the model
    print(f"Loading model from: {MODEL_URI}...")
    model = mlflow.pyfunc.load_model(MODEL_URI)

    # 3. Convert input to DataFrame
    input_df = pd.DataFrame([data_dict])

    # 4. Predict
    prediction = model.predict(input_df)
    
    return prediction[0]

if __name__ == "__main__":
    # Test sample
    sample_data = {
        "CountryCode": 254,
        "PricingStrategy": 2,
        "TransactionHour": 14,
        "TransactionDay": 31,
        "TransactionMonth": 12,
        "TransactionYear": 2025
    }
    
    result = run_prediction(sample_data)
    print(f"\nPrediction Result: {'HIGH RISK' if result > 0.5 else 'LOW RISK'}")
    print(f"Risk Score: {result}")