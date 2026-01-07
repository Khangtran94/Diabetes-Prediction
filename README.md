# Diabetes-Prediction

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

## Dataset Overview:
The data comes from the [Playground Series S5E12 Kaggle competition.](https://www.kaggle.com/competitions/playground-series-s5e12)
![alt text](Images/Background.png)

Overview of the dataset. 
700000 rows and 25 columns
![alt text](Images/Overview.png)

Column Description:

![alt text](Images/Column_Description.png)

## EDA highlights
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
I trained 3 models: XGBoost, LightGBM vs Catboost.
I use Optuna for hyperparameter tuning, please refer to Notebook 3_Optuna_Hyperparameter_tuning.ipynb

1. Confusion Matrix
![alt text](Images/Confusion_Matrix.png)

1. Classification Report

![alt text](Images/Classfication_Report.png)

1. ROC Curve

![alt text](Images/ROC_Curve.png)

1. Feature Importance:
![alt text](Images/Feature_Importance.png)

1. Cross-validation
![alt text](Images/Cross_Validation.png)

=> Select CATBOOST model as final model

## Dependency and Environment Management:
1. Save final model as catboost_model.pkl
1. Create requirements.txt 
1. Create Dockerfile

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