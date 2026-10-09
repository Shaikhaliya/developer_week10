# Week 10: Customer Churn Prediction — Data Preprocessing & Feature Engineering

## Project Overview

This project focuses on preparing customer data for machine learning by applying data preprocessing and feature engineering techniques. The goal is to build a machine learning pipeline that predicts whether a customer is likely to leave a service.

## Dataset

* **Dataset:** Customer Churn Dataset
* **Rows:** 500
* **Original Columns:** 9
* **Target Column:** `Churn`
* `0` = Customer stays
* `1` = Customer leaves

## Tasks Completed

* Inspected the dataset using Pandas.
* Checked missing values and duplicate records.
* Applied Label Encoding and One-Hot Encoding.
* Scaled numerical features using MinMaxScaler and StandardScaler.
* Detected outliers using the IQR method and Z-score.
* Created new features such as `YearlyCharges`, `TenureYears`, and `LongTermCustomer`.
* Selected relevant features using correlation and Random Forest feature importance.
* Built a machine learning pipeline using Scikit-learn.
* Evaluated the model using accuracy, precision, recall, F1-score, and a confusion matrix.

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Jupyter Notebook

## Model Results

The Random Forest model achieved **97% accuracy** on the test dataset.

* **Precision (Churn Class):** 1.00
* **Recall (Churn Class):** 0.73
* **F1-score (Churn Class):** 0.84

The model correctly identified 8 out of 11 customers who actually churned. These results are based on a single train-test split, so further validation is recommended.

## Project Structure

* `churn_data.csv` — Dataset
* `churn_prediction_pipeline.ipynb` — Jupyter Notebook containing the code and results
* `preprocessing_report.md` — Data preprocessing report
* `feature_engineering_documentation.md` — Feature engineering documentation
* `requirements.txt` — Required Python libraries
* `README.md` — Project overview and documentation

## Conclusion

This project demonstrates how data preprocessing and feature engineering can prepare customer data for machine learning. A Random Forest pipeline was developed and evaluated using multiple classification metrics. The results show promising performance, but further testing with cross-validation and more churn examples would help assess how well the model generalizes to new customers.


