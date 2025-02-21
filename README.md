This repository contains seven different machine learning projects focusing on regression and classification tasks. Each project includes data preprocessing, exploratory data analysis (EDA), model training, and evaluation using different machine learning algorithms.

## Table of Contents

- [Common Exploratory Data Analysis (EDA)](#common-exploratory-data-analysis-eda)
- [Projects](#projects)
  - [House Price Prediction](#house-price-prediction)
  - [Gold Price Prediction](#gold-price-prediction)
  - [Car Price Prediction](#car-price-prediction)
  - [Parkinson Disease Detection](#parkinson-disease-detection)
  - [Diabetes Disease Detection](#diabetes-disease-detection)
  - [Heart Disease Detection](#heart-disease-detection)
  - [Breast Cancer Prediction](#breast-cancer-prediction)

---

## Common Exploratory Data Analysis (EDA)

Each project includes an exploratory data analysis phase where the dataset is analyzed using common functions to understand its structure and characteristics. The following EDA steps are commonly used across all projects:

```python
# Importing necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('dataset.csv')  # Replace with actual dataset path

# Basic dataset information
print(df.head())      # First 5 rows
print(df.tail())      # Last 5 rows
print(df.shape)      # Number of rows and columns
print(df.info())     # Data types and missing values
print(df.describe()) # Summary statistics

# Checking missing values
print(df.isnull().sum())

# Correlation heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()
```

---

## Projects

### House Price Prediction

- **Algorithm Used:** XGBoost Regressor
- **Dataset:** Contains features like area, number of rooms, location, etc.
- **Libraries Used:** `pandas`, `numpy`, `seaborn`, `sklearn`, `xgboost`
- **Key Steps:**
  - Data Cleaning and Preprocessing
  - Feature Selection
  - Model Training using `XGBRegressor`
  - Model Evaluation (RMSE, R² Score)

### Gold Price Prediction

- **Algorithm Used:** Random Forest Regressor
- **Dataset:** Historical gold price data with influencing economic indicators.
- **Libraries Used:** `pandas`, `numpy`, `seaborn`, `sklearn`
- **Key Steps:**
  - Handling Missing Values
  - Exploratory Data Analysis
  - Model Training using `RandomForestRegressor`
  - Model Performance Evaluation

### Car Price Prediction

- **Algorithms Used:** Linear Regression, Lasso Regression
- **Dataset:** Contains car attributes like brand, year, fuel type, mileage, etc.
- **Libraries Used:** `pandas`, `numpy`, `seaborn`, `sklearn`
- **Key Steps:**
  - Data Preprocessing (Handling categorical variables, missing values)
  - Applying `LinearRegression` and `Lasso Regression`
  - Comparing Model Performance

### Parkinson Disease Detection

- **Algorithm Used:** Support Vector Machine (SVM)
- **Dataset:** Features extracted from voice recordings to detect Parkinson's disease.
- **Libraries Used:** `pandas`, `numpy`, `seaborn`, `sklearn`
- **Key Steps:**
  - Feature Engineering
  - SVM Model Training
  - Performance Evaluation using Accuracy, Precision, Recall

### Diabetes Disease Detection

- **Algorithm Used:** Support Vector Machine (SVM)
- **Dataset:** Medical dataset with blood glucose levels, BMI, age, etc.
- **Libraries Used:** `pandas`, `numpy`, `seaborn`, `sklearn`
- **Key Steps:**
  - Data Cleaning
  - Applying SVM Classifier
  - Performance Metrics Evaluation (Confusion Matrix, Precision, Recall)

### Heart Disease Detection

- **Algorithm Used:** Logistic Regression
- **Dataset:** Patient medical records including cholesterol, blood pressure, heart rate, etc.
- **Libraries Used:** `pandas`, `numpy`, `seaborn`, `sklearn`
- **Key Steps:**
  - Data Standardization
  - Logistic Regression Model Training
  - Accuracy, F1-Score Evaluation

### Breast Cancer Prediction

- **Algorithm Used:** Logistic Regression
- **Dataset:** Breast cancer dataset with tumor characteristics.
- **Libraries Used:** `pandas`, `numpy`, `seaborn`, `sklearn`
- **Key Steps:**
  - Data Preprocessing
  - Logistic Regression Model Training
  - Performance Evaluation using ROC Curve, AUC Score

---

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/repository-name.git
   ```
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Jupyter Notebook or Python scripts for each project.

---

## Dependencies

Ensure you have the following Python libraries installed:

```bash
pip install pandas numpy seaborn scikit-learn matplotlib xgboost
```

---

## Contributing

Feel free to fork this repository, make improvements, and submit a pull request!

---

## License

This project is licensed under the MIT License.

