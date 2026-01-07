# web_service.py
from fastapi import FastAPI
from typing import Dict, Any
import uvicorn
import pickle
import pandas as pd
from pathlib import Path

app = FastAPI(title="Diabetes Prediction API")

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
    json_input: Dict[str, Any],
    feature_columns: list,
    categorical_cols: list,
    fill_numeric: float = 0.0,
) -> pd.DataFrame:
    """
    Converts JSON input into a DataFrame suitable for the CatBoost model.
    Fills missing columns with default values (0 for numeric, 'Unknown' for categorical).
    """
    # Convert JSON to DataFrame
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


def predict_single(patient: Dict[str, Any]) -> float:
    """
    Predicts the probability of diabetes for a single input.
    """
    input_df = prepare_input(patient, feature_columns, categorical_cols)
    prob = loaded_model.predict_proba(input_df)[:, 1][0]
    return round(float(prob), 3)

# ----------------------
# API Endpoints
# ----------------------
@app.post("/predict")
def predict(patient: Dict[str, Any]):
    """
    POST endpoint for predicting diabetes probability.
    Expects JSON with patient feature values.
    """
    prob = predict_single(patient)
    return {
        "predicted_probability": prob,
        "diabetes_risk": bool(prob >= 0.5),
    }

@app.get("/")
def read_root():
    """
    Root endpoint to check if API is running.
    """
    return {"message": "Hello! The Diabetes Prediction API is running."}

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

#### run for FastAPI please run in terminal
#### uvicorn Web_Service:app --reload