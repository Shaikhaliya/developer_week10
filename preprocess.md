# Customer Churn Prediction — Preprocessing Report

## 1. Project Overview

This project focuses on preparing customer data for machine learning and predicting whether a customer is likely to leave a service. The project demonstrates categorical encoding, feature scaling, outlier detection, feature engineering, feature selection, and an end-to-end preprocessing pipeline.

## 2. Project Objectives

* Explore and understand customer churn data.
* Convert categorical data into numerical representations.
* Apply Min-Max Scaling and Standard Scaling.
* Detect potential outliers using IQR and Z-score methods.
* Create new features from existing customer information.
* Identify useful features using correlation and Random Forest feature importance.
* Build and evaluate a complete preprocessing and prediction pipeline.

## 3. Dataset Description

The dataset contains 500 customer records and 9 original columns.

| Column           | Description                      |
| ---------------- | -------------------------------- |
| CustomerID       | Unique customer identifier       |
| Tenure           | Customer tenure in months        |
| MonthlyCharges   | Monthly customer charges         |
| TotalCharges     | Total customer charges           |
| Contract         | Customer contract type           |
| PaymentMethod    | Customer payment method          |
| PaperlessBilling | Paperless billing status         |
| SeniorCitizen    | Senior citizen indicator         |
| Churn            | Target variable: 0 = No, 1 = Yes |

The dataset contained no missing values in the original columns.

## 4. Initial Data Analysis

The original churn distribution was:

| Churn status | Customers | Percentage |
| ------------ | --------: | ---------: |
| Stayed (0)   |       447 |      89.4% |
| Churned (1)  |        53 |      10.6% |
| Total        |       500 |       100% |

The target variable is imbalanced because the number of customers who stayed is considerably higher than the number who churned.

## 5. Categorical Encoding

Three encoding methods were demonstrated.

### 5.1 Label Encoding

Label Encoding converted `PaperlessBilling` into numerical values:

* No = 0
* Yes = 1

This made the binary categorical variable suitable for numerical processing.

### 5.2 One-Hot Encoding

One-Hot Encoding was applied to the `Contract` column. It generated separate binary columns for the contract categories:

* Contract_Month-to-month
* Contract_One year
* Contract_Two year

One-Hot Encoding represents categories without imposing an artificial numerical ranking.

### 5.3 Ordinal Encoding

Ordinal Encoding was demonstrated using an example education dataset:

* School = 0
* Graduate = 1
* Postgraduate = 2

This was a demonstration of the method, not a transformation of the original customer dataset. Ordinal Encoding is appropriate when categories have a meaningful order.

## 6. Feature Scaling

Two scaling methods were applied to `Tenure` and `MonthlyCharges`.

### 6.1 Min-Max Scaling

Min-Max Scaling transformed the selected features to a range of 0 to 1.

### 6.2 Standard Scaling

Standard Scaling transformed the selected features using the mean and standard deviation. The transformed training data has approximately zero mean and unit standard deviation.

For the final pipeline, Standard Scaling was included for numerical features. Scaling is not generally required by Random Forest, but it demonstrates a standard preprocessing technique.

## 7. Outlier Detection

Potential outliers in `MonthlyCharges` were checked using two methods.

### 7.1 IQR Method

* Q1 = 67
* Q3 = 158
* IQR = 91
* Lower limit = -69.5
* Upper limit = 294.5
* Outliers detected = 0

### 7.2 Z-Score Method

A Z-score threshold of absolute value greater than 3 was used.

* Outliers detected = 0

Neither method detected outliers in `MonthlyCharges`. Therefore, no values were removed from this column.

This result applies only to `MonthlyCharges`; other columns may have different distributions.

## 8. Feature Engineering

Five additional features were created.

| Feature            | Calculation                                    | Purpose                                              |
| ------------------ | ---------------------------------------------- | ---------------------------------------------------- |
| AvgMonthlySpending | TotalCharges / Tenure                          | Estimate average monthly spending from total charges |
| YearlyCharges      | MonthlyCharges × 12                            | Estimate annual charges at the current monthly rate  |
| TenureYears        | Tenure / 12                                    | Express tenure in years                              |
| HighMonthlyCharges | 1 if charges exceed the median, otherwise 0    | Identify relatively high monthly charges             |
| LongTermCustomer   | 1 if tenure is at least 24 months, otherwise 0 | Identify long-tenure customers                       |

The dataset contained zero customers with zero tenure, and the engineered `AvgMonthlySpending` column had no missing values.

These engineered features were treated as candidate predictors. Their usefulness should be validated with model evaluation.

## 9. Feature Selection

Two techniques were used.

### 9.1 Correlation Analysis

Correlation with churn showed the following results:

| Feature            | Correlation with Churn |
| ------------------ | ---------------------: |
| LongTermCustomer   |                -0.5161 |
| TenureYears        |                -0.5092 |
| Tenure             |                -0.5092 |
| AvgMonthlySpending |                 0.4047 |
| Contract_One year  |                -0.1575 |
| MonthlyCharges     |                 0.1074 |

Tenure and LongTermCustomer showed negative relationships with churn. AvgMonthlySpending showed a positive relationship in this dataset. Correlation indicates association, not causation.

### 9.2 Random Forest Feature Importance

The highest feature importance values from the initial Random Forest model were:

| Feature            | Importance |
| ------------------ | ---------: |
| Tenure             |     0.2571 |
| TenureYears        |     0.2454 |
| AvgMonthlySpending |     0.1395 |
| LongTermCustomer   |     0.0754 |
| TotalCharges       |     0.0563 |

Tenure was the most important feature in this model. TenureYears and LongTermCustomer are derived from Tenure, so their importance should not be interpreted as independent contributions.

CustomerID and helper columns were excluded from the final model inputs.

## 10. Final Preprocessing Pipeline

The final pipeline used:

1. A train-test split with 80% training data and 20% testing data.
2. Stratification to preserve the churn proportion across the splits.
3. Median imputation for numerical features.
4. Standard Scaling for numerical features.
5. Most-frequent imputation for categorical features.
6. One-Hot Encoding with `handle_unknown="ignore"`.
7. A Random Forest Classifier with 100 trees.

All preprocessing transformations were fitted on training data through a scikit-learn Pipeline and ColumnTransformer. The same fitted transformations were then applied to the test data, reducing the risk of data leakage.

## 11. Model Evaluation

The model was evaluated on 100 test customers.

| Metric                    |  Result |
| ------------------------- | ------: |
| Accuracy                  |  97.00% |
| Precision for churn class | 100.00% |
| Recall for churn class    |  72.73% |
| F1-score for churn class  |  84.21% |

### Confusion Matrix

| Actual / Predicted | Stayed (0) | Churned (1) |
| ------------------ | ---------: | ----------: |
| Stayed (0)         |         89 |           0 |
| Churned (1)        |          3 |           8 |

The model correctly identified 8 of the 11 customers who churned. It missed 3 churn cases.

Although the model achieved 97% accuracy, the recall of 72.73% indicates that some customers who churned were not identified. Recall is important for customer retention because missed churn cases represent customers the business may fail to reach in time.

## 12. Limitations

* The dataset contains only 500 records.
* The test set contains 11 churn cases, so the metrics may vary across different splits.
* The dataset is imbalanced.
* Some engineered features overlap with existing features.
* The results do not establish causal relationships.
* Feature importance can be affected by correlated features.

## 13. Future Improvements

* Evaluate the pipeline using stratified cross-validation.
* Compare Random Forest with Logistic Regression and other classifiers.
* Test class weighting or resampling methods if appropriate.
* Tune model hyperparameters.
* Evaluate the precision-recall trade-off for churn detection.
* Validate the model on additional customer data.

## 14. Conclusion

This project demonstrated a complete customer churn preprocessing workflow. Categorical encoding, feature scaling, outlier detection, feature engineering, feature selection, and a scikit-learn pipeline were implemented.

The Random Forest pipeline achieved 97% accuracy, 100% precision, 72.73% recall, and an 84.21% F1-score for the churn class on the test set. Further validation is recommended before using the model for real business decisions.
