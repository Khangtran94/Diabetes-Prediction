import os
import numpy as np 
import pandas as pd
import pickle

# Get the script's folder
script_dir = os.path.dirname(os.path.abspath(__file__))

# Build path relative to script
csv_path = os.path.join(script_dir, '..', 'Data', 'train.csv')
df = pd.read_csv(csv_path, index_col='id')

target = 'diagnosed_diabetes'

def overview(df, nunique_threshold=15):
    """
    Prints an overview of the DataFrame.
    - Ignores the last column (assumed target).
    - Counts numerical and categorical features.
    - Converts low-cardinality numeric features to categorical.
    """

    import pandas as pd

    # Drop last column (assumed target)
    features = df.iloc[:, :-1]

    # Initial type detection
    num_cols = features.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = features.select_dtypes(include=['object', 'category']).columns.tolist()

    # Convert low-cardinality numeric columns to categorical
    converted_cols = []
    for col in num_cols.copy():
        if features[col].nunique() <= nunique_threshold:
            df[col] = df[col].astype('category')
            num_cols.remove(col)
            cat_cols.append(col)
            converted_cols.append(col)

    print("===== DATA OVERVIEW =====")
    print(f"Total features (excluding last column): {features.shape[1]}")
    print(f"Numeric features: {len(num_cols)}")
    print(f"Categorical features: {len(cat_cols)}")

    if converted_cols:
        print(f"\nConverted to categorical (nunique ≤ {nunique_threshold}):")
        for col in converted_cols:
            print(f"  - {col}")

    print("\n--- Categorical Columns Detail ---")
    for col in cat_cols:
        print(f"{col}: {df[col].nunique()} unique values")

    return num_cols, cat_cols

num_cols, cat_cols = overview(df)

for col in df.select_dtypes(include='object').columns:
    df[col] = pd.Categorical(df[col])
    
    
### Split data into train and validation sets    
neg = np.sum(df[target] == 0)
pos = np.sum(df[target] == 1)
scale_pos_weight = neg / pos
scale_pos_weight
X = df.drop('diagnosed_diabetes',axis=1)
y = df['diagnosed_diabetes']

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, roc_curve, auc, roc_auc_score

X_train, X_val, y_train, y_val = train_test_split(X,y, stratify = y, random_state=42)

### train final models with best parameters found from hyperparameter tuning
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

cat_params = {
    'iterations': 1162,                    # updated
    'depth': 7,                           # updated
    'learning_rate': 0.07750133623087592, # updated
    'l2_leaf_reg': 6.040419312733866,    # updated
    'bagging_temperature': 0.5698939842266584, # updated
    'border_count': 177,                   # updated
    'verbose': 0
}

# Identify categorical columns

cat_model = CatBoostClassifier(**cat_params, eval_metric="AUC", random_seed=42,
                               loss_function='Logloss', allow_writing_files=False)

cat_model.fit(
    X_train, y_train,
    cat_features=cat_cols,
    eval_set=(X_val, y_val),
    use_best_model=True, early_stopping_rounds=50)

# y_pred_cat = cat_model.predict_proba(X_val)[:, 1]
# auc = roc_auc_score(y_val, y_pred_cat)
# print(f"Final Validation ROC-AUC CatBoost: {auc:.6f}")

### Save model
feature_columns = X_train.columns.tolist()  # list of all feature names
categorical_cols = X_train.select_dtypes('category').columns.tolist()

from pathlib import Path

output_path = Path(__file__).resolve().parent / "catboost_model.pkl"

# Save schema together with model
with open(output_path, 'wb') as f:
    pickle.dump({
        'model': cat_model,
        'feature_columns': feature_columns,
        'categorical_cols': categorical_cols
    }, f)

print(f"Model saved at: {output_path}")