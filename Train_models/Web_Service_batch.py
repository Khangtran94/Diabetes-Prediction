# web_service_batch.py
from fastapi import FastAPI
from typing import Dict, Any, List, Union
import uvicorn
import pickle
import pandas as pd
from pathlib import Path

app = FastAPI(title="Diabetes Prediction API (Batch Version)")

# ----------------------
# Load CatBoost model
# ----------------------
model_path = Path(__file__).resolve().parent / "catboost_model.pkl"
print("Loading model from:", model_path)

with open(model_path, "rb") as f:
    saved = pickle.load(f)

loaded_model = saved["model"]
feature_columns = saved["feature_columns"]
categorical_cols = saved["categorical_cols"]

# ----------------------
# Utility Functions
# ----------------------
def prepare_input(
    json_input: Union[Dict[str, Any], List[Dict[str, Any]]],
    feature_columns: list,
    categorical_cols: list,
    fill_numeric: float = 0.0,
) -> pd.DataFrame:
    """
    Converts JSON input (single dict or list of dicts) into a DataFrame suitable for the CatBoost model.
    Fills missing columns with default values (0 for numeric, 'Unknown' for categorical).
    """
    if isinstance(json_input, dict):
        df = pd.DataFrame([json_input])
    else:
        df = pd.DataFrame(json_input)

    # Fill missing columns
    for col in feature_columns:
        if col not in df.columns:
            if col in categorical_cols:
                df[col] = "Unknown"
            else:
                df[col] = fill_numeric

    # Ensure correct column order
    df = df[feature_columns]

    # Convert numeric columns to floats
    for col in df.columns:
        if col not in categorical_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(fill_numeric)

    return df

def predict_patients(patients: Union[Dict[str, Any], List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """
    Predicts diabetes probabilities for single or multiple patients.
    Returns a list of dictionaries with predicted probability and risk.
    """
    input_df = prepare_input(patients, feature_columns, categorical_cols)
    probs = loaded_model.predict_proba(input_df)[:, 1]

    results = []
    for i, prob in enumerate(probs):
        patient_data = patients[i] if isinstance(patients, list) else patients
        results.append({
            "input_values": patient_data,
            "predicted_probability": round(float(prob), 3),
            "diabetes_risk": bool(prob >= 0.5)
        })
    return results

# ----------------------
# API Endpoints
# ----------------------
@app.post("/predict")
def predict_endpoint(patients: Union[Dict[str, Any], List[Dict[str, Any]]]):
    """
    POST endpoint for predicting diabetes probability.
    Accepts either a single patient JSON or a list of patient JSONs.
    """
    results = predict_patients(patients)
    # If single patient, return only the first result (not a list)
    if isinstance(patients, dict):
        return results[0]
    return results

@app.get("/")
def read_root():
    """
    Root endpoint to check if API is running.
    """
    return {"message": "Hello! The Diabetes Prediction API (Batch Version) is running."}

@app.get("/predict")
def predict_get():
    """
    Optional GET route reminding users to use POST.
    """
    return {"message": "Please use POST with JSON data to get predictions."}

# ----------------------
# Run app
# ----------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)


### uvicorn Web_Service_batch:app --reload