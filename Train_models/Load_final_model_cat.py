import pickle
import pandas as pd
from pathlib import Path

output_path = Path(__file__).resolve().parent / "catboost_model.pkl"

### Load the model:
with open(output_path, 'rb') as f:
    saved = pickle.load(f)

loaded_model = saved['model']
feature_columns = saved['feature_columns']
categorical_cols = saved['categorical_cols']

def prepare_input(json_input, feature_columns, categorical_cols, fill_numeric=0):
    # Convert JSON to DataFrame
    if isinstance(json_input, dict):
        df = pd.DataFrame([json_input])
    else:
        df = pd.DataFrame(json_input)

    # Fill missing columns
    for col in feature_columns:
        if col not in df.columns:
            if col in categorical_cols:
                df[col] = 'Unknown'  # or first category if known
            else:
                df[col] = fill_numeric

    # Ensure correct column order
    df = df[feature_columns]
    return df

### Example
json_input = {
    "gender": "Male",
    "family_history_diabetes": "1",
    "physical_activity_minutes_per_week" : "80",
    "age": "35",
    "bmi": "25",
    "triglycerides":"100",
    "cholesterol_total":"170",
    "hdl_cholesterol":"80",}

input_df = prepare_input(json_input, feature_columns, categorical_cols)
y_pred_prob = loaded_model.predict_proba(input_df)[:, 1]

print("Input values and predicted outcome:")
print('-'*100)

# Only print features provided in the input JSON
for col in json_input.keys():
    print(f"{col}: {input_df[col].iloc[0]}")

print('-'*100)
print(f"Predicted probability of Diabetes = 1: {y_pred_prob[0]:.3f}")