# Diabetes-Prediction

<img width="764" height="379" alt="Screenshot 2026-01-07 at 22 03 41" src="https://github.com/user-attachments/assets/2686b8e1-0219-47cc-83ff-3512759f7b38" />

Predicting Diabetes Probability with Machine Learning

This project builds a machine learning model to predict the probability (0-1) that a patient will be diagnosed with diabetes based on health data. It uses supervised learning techniques, compares multiple models (XGBoost, LightGBM, CatBoost), and deploys the best model (CatBoost) via FastAPI web services and Docker for easy use in healthcare scenarios.

The project originated from the Kaggle Playground Series - S5E12 competition dataset, which includes features like age, BMI, cholesterol, physical activity, family history, and more.

## Key Features

Exploratory Data Analysis (EDA) with visualizations (correlation heatmaps, feature importance, distributions)
Hyperparameter tuning using Optuna
Model comparison and selection based on AUC-ROC
FastAPI-based web service for single and batch predictions
Docker support for containerized deployment
Probabilistic predictions for risk assessment

## Project Structure

```text
.
├── Data/
│   ├── test.csv
│   └── train.csv
├── Dockerfile
├── Images/
│   └── [screenshots and diagrams]
├── Notebooks/
│   ├── 1_Import_vs_Check_Data.ipynb
│   ├── 2_EDA_Notebook.ipynb
│   └── 3_Optuna_Hyperparameter_tuning.ipynb
├── README.md
├── Train_models/
│   ├── Load_final_model_cat.py
│   ├── Training_Models.ipynb
│   ├── Web_Service.py
│   ├── Web_Service_batch.py
│   ├── catboost_model.pkl
│   └── final_model_Catboost.py
├── requirements.txt
└── testcase_API.txt 
```

## Business Perspective:
Early detection of diabetes is critical for preventive healthcare. By predicting the probability that a patient will be diagnosed with diabetes, healthcare providers can:

* Identify high-risk patients early.
* Provide targeted lifestyle and medical interventions.
* Reduce long-term healthcare costs associated with untreated diabetes.

The goal is to turn patient health data into actionable insights, enabling data-driven decisions in clinical settings.

## Data Science Perspective:

From a data science standpoint, the challenge involves:
* Analyzing patient data (age, BMI, cholesterol, physical activity, family history, etc.).
* Feature engineering to improve model predictions.
* Modeling the probability of diabetes using supervised machine learning.

The key goal is to build accurate, interpretable predictive models that can generalize to new patients.

## Target Goal:

Predict the probability that a patient will be diagnosed with diabetes using the available dataset.

Predictions are probabilities (values between 0 and 1) rather than just a binary classification.

This allows healthcare providers to assess risk levels, not just yes/no outcomes.

## Machine Learning Models:

ML models help us:

* Identify complex patterns in patient health data.
* Make personalized predictions for each patient.
* Support clinical decision-making with probabilistic risk scores.
* Compare different algorithms to select the best performing model.

## Evaluation Criteria:

Submissions are evaluated using Area Under the ROC Curve (AUC-ROC):

* ROC Curve: Plots the true positive rate (sensitivity) vs the false positive rate (1 - specificity) for different probability thresholds.

* AUC (Area Under the Curve): Measures how well the model distinguishes between patients with and without diabetes.

Why AUC-ROC?

* It evaluates the quality of probability predictions, not just binary accuracy.

* A higher AUC indicates the model correctly ranks high-risk patients above low-risk patients.

* This is especially useful in healthcare where false positives and false negatives have different consequences.

## Dataset Overview:
The data comes from the [Playground Series S5E12 Kaggle competition.](https://www.kaggle.com/competitions/playground-series-s5e12)
![alt text](Images/Background.png)

Overview of the dataset. 
700000 rows and 25 columns
![alt text](Images/Overview.png)

Column Description:

![alt text](Images/Column_Description.png)

## EDA highlights:
More detail you can visit the 
[EDA Notebook](https://github.com/Khangtran94/Diabetes-Prediction/blob/main/Notebooks/2_EDA_Notebook.ipynb).

1. Mutual Information:
![alt text](Images/Mutual_Information.png)

1. Heatmap Correlation:
![alt text](Images/Heatmap.png)

1. Age vs Diabetes:
![alt text](Images/Age.png)
![alt text](Images/Age_Risk.png)

1. Physical Activity Minutes per Week:
![alt text](Images/Physical_Activity.png)

1. Diet Score 
![alt text](Images/Diet_Score.png)

1. BMI
![alt text](Images/BMI.png)
![alt text](Images/BMI_advance.png)

1. Blood Pressure
![alt text](Images/Systolic_bp.png)
![alt text](Images/Diastolic.png)
![alt text](Images/BP_2.png)
![alt text](Images/BP_1.png)

1. Family History
![alt text](Images/Family_History.png)
![alt text](Images/Weight_vs_History.png)

## Model Training vs Compare performance
![alt text](Images/Target_Distribution.png)

I trained 3 models: XGBoost, LightGBM vs Catboost.
I use Optuna for hyperparameter tuning, please refer to [Optuna_Hyperparameter_tuning](https://github.com/Khangtran94/Diabetes-Prediction/blob/main/Notebooks/3_Optuna_Hyperparameter_tuning.ipynb).

* Confusion Matrix
![alt text](Images/Confusion_Matrix.png)

* Classification Report

![alt text](Images/Classfication_Report.png)

* ROC Curve

![alt text](Images/ROC_Curve.png)

* Feature Importance:
![alt text](Images/Feature_Importance.png)

* Cross-validation

![alt text](Images/Cross_Validation.png)

=> Select CATBOOST model as final model

## Dependency and Environment Management:
1. Save final model as catboost_model.pkl
1. Create requirements.txt 
1. Create Dockerfile

## Usage

### Prerequisites
* Python 3.8 or higher
* Git installed for cloning the repository
* Docker (optional, for containerization)

---

### Step 1: Clone the Repository

Clone the repository to your local machine:

```bash
git clone https://github.com/Khangtran94/Diabetes-Prediction.git
cd Diabetes-Prediction
```

### Step 2: Install Dependencies

Install the required Python packages using the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### Step 3: Explore and Prepare Data

The dataset is located in the `Data/` directory (`train.csv` for training and `test.csv` for testing).

Use the Jupyter notebooks in the `Notebooks/` directory:

#### Import and Check Data
Run `Notebooks/1_Import_vs_Check_Data.ipynb` to load the data, inspect its structure, check for missing values, and perform initial cleaning.

#### Exploratory Data Analysis (EDA)
Run `Notebooks/2_EDA_Notebook.ipynb` to analyze patterns, correlations (e.g., heatmap), and visualizations such as Age vs Diabetes, BMI distribution, and feature importance via mutual information.

#### Hyperparameter Tuning
Run `Notebooks/3_Optuna_Hyperparameter_tuning.ipynb` to optimize hyperparameters for models like XGBoost, LightGBM, and CatBoost using Optuna.

---

### Step 4: Train Models

Navigate to the `Train_models/` directory and run the training notebook:

```bash
cd Train_models
jupyter notebook Training_Models.ipynb
```
This notebook trains and compares XGBoost, LightGBM, and CatBoost models, evaluates them using AUC-ROC, confusion matrix, classification report, ROC curve, feature importance, and cross-validation. The best model (CatBoost) is saved as catboost_model.pkl.

### Step 5: Make Predictions
You can use the trained model for predictions in several ways:

#### Option 1: Single or Batch Predictions via FastAPI Web Service
- For **single predictions**: Run `Web_Service.py`: ```termianl uvicorn Web_Service:app --reload```
This starts a FastAPI server. Access the interactive UI at `http://localhost:8000/docs` to input patient data and get diabetes probability predictions.

- For **batch predictions**: Run `Web_Service_batch.py`: ```termianl uvicorn Web_Service_batch:app --reload```
Use the API endpoint to submit multiple patient records (e.g., via JSON payload) for bulk predictions.

Sample test cases are provided in `testcase_API.txt` for validating the API.

#### Option 2: Terminal-Based Predictions
Edit the input JSON in `Load_final_model_cat.py` with patient data, then run: ```python Load_final_model_cat.py```

## FastAPI Web Services
* Option 1: via FastAPI UI.
![alt text](Images/FastAPI.png)
  * Run predict with Diabetes Prediction API (Batch Version) (this time I updated to not only single but also batch prediction).
    * Single patient:
![alt text](Images/FastAPI_Single.png)
    * Batch / Multiple patients:
    ![alt text](Images/FastAPI_Batch.png)

* Option 2: Edit the JSON file in Load_final_model_cat.py then run via Terminal

Example 1

![alt text](Images/CLI_Example_1.png)

Example 2

![alt text](Images/CLI_Example_2.png)

## Build Docker Images
![alt text](Images/Built_Docker_Images.png)
![alt text](Images/Check_Docker_Image.png)
![alt text](Images/Run_Docker_Image.png)

## Docker Hub:
Push Docker Image to Docker Hub:
![alt text](Images/Push_Docker_Hub.png)

You can follow this link to access to Diabetes Image on Docker hub: [![Diabetes API](https://img.shields.io/badge/Docker-Diabetes_API-blue)](https://hub.docker.com/repository/docker/khangtranvn/diabetes-api/general)

![alt text](Images/Docker_Hub.png)
