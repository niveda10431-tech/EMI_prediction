
EMIPredict AI - Intelligent Financial Risk Assessment Platform

Project Overview
----------------
EMIPredict AI is a machine learning-based financial risk assessment platform designed to help financial institutions
and fintech companies evaluate EMI eligibility and determine the maximum EMI amount a customer can safely afford.

The system uses machine learning models along with MLflow experiment tracking and a Streamlit web application
to provide real-time financial insights.

Machine Learning Problems
-------------------------
1. Classification
   Target: emi_eligibility
   Classes:
   - Eligible
   - High_Risk
   - Not_Eligible

2. Regression
   Target: max_monthly_emi
   Predicts the maximum safe EMI amount a customer can afford.

Dataset
-------
- 400,000 financial records
- 22 financial and demographic features
- 5 EMI scenarios including shopping, appliances, vehicle, personal loan, and education EMI

Project Workflow
----------------
1. Data Cleaning
2. Exploratory Data Analysis (EDA)
3. Feature Engineering
4. Encoding and Scaling
5. Train-Test Split
6. Model Training
7. Cross Validation
8. Model Evaluation
9. Feature Importance
10. Model Training using Top Important Features
11. Model Comparison
12. MLflow Experiment Tracking
13. Final Model Selection

Machine Learning Models
-----------------------
Classification:
- Logistic Regression
- Random Forest Classifier
- XGBoost Classifier

Regression:
- Linear Regression
- Random Forest Regressor
- XGBoost Regressor

Evaluation Metrics
------------------
Classification:
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

Regression:
- MAE
- RMSE
- R2 Score

Technologies Used
-----------------
Python
Pandas
NumPy
Scikit-learn
XGBoost
Matplotlib
Seaborn
MLflow
Streamlit

Application
-----------
A Streamlit web application is used to:
- Predict EMI eligibility
- Predict maximum EMI amount
- Provide financial risk insights

Deployment
----------
The application can be deployed using Streamlit Cloud with integration to a GitHub repository.
