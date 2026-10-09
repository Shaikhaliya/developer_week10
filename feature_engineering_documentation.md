# Feature Engineering Documentation — Customer Churn Prediction

## 1. Introduction

Feature engineering is the process of creating useful input variables from existing data. It can help a machine learning model identify patterns that are not directly represented by the original columns.

In this project, five features were engineered from customer tenure and billing information.

## 2. Original Features Used

* `Tenure`: Customer tenure in months.
* `MonthlyCharges`: Current monthly charges.
* `TotalCharges`: Total customer charges.

These columns were used to create additional features that describe customer spending, tenure, and relative charge levels.

## 3. Engineered Features

### 3.1 AvgMonthlySpending

**Formula:**

`AvgMonthlySpending = TotalCharges / Tenure`

**Purpose:** Estimates average monthly spending using the customer's total charges and tenure.

**Example:** If total charges are 1,540 and tenure is 6 months:

`AvgMonthlySpending = 1540 / 6 = 256.67`

**Rationale:** This feature can help represent a customer's historical average spending.

**Limitation:** It may differ from the current `MonthlyCharges` because charges can change over time.

### 3.2 YearlyCharges

**Formula:**

`YearlyCharges = MonthlyCharges × 12`

**Purpose:** Estimates the cost over 12 months at the current monthly rate.

**Rationale:** Annualized charges make it easier to compare the approximate yearly cost of customers.

**Limitation:** This is an estimate, not necessarily the actual amount billed during a year.

### 3.3 TenureYears

**Formula:**

`TenureYears = Tenure / 12`

**Purpose:** Converts tenure from months to years.

**Example:** A tenure of 6 months becomes 0.5 years.

**Rationale:** Expressing tenure in years can make the feature easier to interpret.

**Limitation:** It represents the same underlying information as `Tenure`, so both features may be redundant.

### 3.4 HighMonthlyCharges

**Formula:**

`HighMonthlyCharges = 1 if MonthlyCharges > median(MonthlyCharges), otherwise 0`

**Purpose:** Identifies customers whose monthly charges are above the dataset median.

**Interpretation:**

* 1 = Above the median
* 0 = At or below the median

**Rationale:** This feature creates a simple indicator for relatively high monthly charges.

**Limitation:** The threshold depends on the dataset distribution. In a production pipeline, the median threshold should be learned from training data only.

### 3.5 LongTermCustomer

**Formula:**

`LongTermCustomer = 1 if Tenure >= 24 months, otherwise 0`

**Purpose:** Identifies customers whose tenure is at least two years.

**Interpretation:**

* 1 = Tenure of 24 months or more
* 0 = Tenure below 24 months

**Rationale:** A longer customer relationship may be associated with a lower likelihood of churn.

**Limitation:** This feature is derived from `Tenure`, so it contains overlapping information.

## 4. Feature Validation

After creating the features, the following checks were performed:

* Verified the new columns and their values.
* Checked for customers with zero tenure.
* Checked for missing values in `AvgMonthlySpending`.

Results:

| Validation                           | Result |
| ------------------------------------ | -----: |
| Customers with zero tenure           |      0 |
| Missing values in AvgMonthlySpending |      0 |
| Engineered features created          |      5 |

## 5. Relationship with Churn

Correlation analysis produced the following results:

| Feature            | Correlation with Churn |
| ------------------ | ---------------------: |
| LongTermCustomer   |                -0.5161 |
| TenureYears        |                -0.5092 |
| AvgMonthlySpending |                 0.4047 |
| HighMonthlyCharges |                 0.0872 |
| YearlyCharges      |                 0.1074 |

The negative correlations for LongTermCustomer and TenureYears indicate that higher tenure was associated with lower churn in this dataset. AvgMonthlySpending showed a positive association with churn.

These correlations do not prove that the features cause customers to churn or stay.

## 6. Feature Importance

The initial Random Forest model assigned the following importance values:

| Feature            | Importance |
| ------------------ | ---------: |
| Tenure             |     0.2571 |
| TenureYears        |     0.2454 |
| AvgMonthlySpending |     0.1395 |
| LongTermCustomer   |     0.0754 |
| TotalCharges       |     0.0563 |

Tenure-related features ranked highly. However, `Tenure`, `TenureYears`, and `LongTermCustomer` overlap in the information they represent. Their individual importance values should therefore be interpreted carefully.

## 7. Feature Selection Decisions

The following decisions were made for the final pipeline:

* `CustomerID` was removed because it is an identifier rather than a meaningful predictive feature.
* `MonthlyCharges_Zscore` was removed because it was a helper column used only for outlier detection.
* `PaperlessBilling_Encoded` was removed because the same information was represented by the one-hot encoded `PaperlessBilling` column.
* `TenureYears` was removed from the final inputs to reduce redundancy with `Tenure`.
* Other candidate features were retained for initial model evaluation.

The final set of features should be validated through cross-validation and model comparison before making further removals.

## 8. Conclusion

Five engineered features were created to represent customer spending, annualized cost, tenure, and charge levels. Validation confirmed that there were no zero-tenure customers and no missing values in `AvgMonthlySpending`.

Correlation analysis and Random Forest feature importance helped identify potentially useful features. The findings suggest that customer tenure is strongly associated with churn in this dataset, while further validation is needed to determine whether the engineered features improve generalization.
