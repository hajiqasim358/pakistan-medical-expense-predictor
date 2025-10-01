# Pakistan Medical Expense Predictor 🏥

This project uses a synthetic dataset of 50,000 individuals from Pakistan to predict annual medical expenses using a Linear Regression model. The dataset includes demographic, lifestyle, and health-related features, making it ideal for regression modeling and explainable AI experiments.

## 📊 Dataset Overview

The dataset `synthetic_medical_expenses_pakistan.csv` contains 50,000 rows and the following features:

| Feature            | Type        | Description                                                                 |
|--------------------|-------------|-----------------------------------------------------------------------------|
| Age                | Integer     | Age of the individual (18–65 years)                                        |
| BMI                | Float       | Body Mass Index (normally distributed around 27)                           |
| Children           | Integer     | Number of dependent children (0–4)                                         |
| Smoker             | Categorical | Whether the person smokes (`Yes` or `No`)                                  |
| Region             | Categorical | Region in Pakistan (`Punjab`, `Sindh`, `Khyber Pakhtunkhwa`, `Balochistan`, `Islamabad`) |
| ExerciseFreq       | Integer     | Number of days per week the person exercises (0–6)                         |
| ChronicConditions  | Integer     | Number of chronic health conditions (Poisson-distributed around 1.5)       |
| MedicalExpenses    | Float       | Annual medical expenses in PKR (target variable)                           |

## 🧠 Model Training

The model is trained using a Linear Regression pipeline with one-hot encoding for categorical features. The training script is available in `train_linear_model.py`.

### Steps:
1. Load and preprocess the dataset
2. Encode categorical features (`Smoker`, `Region`)
3. Train a Linear Regression model
4. Save the model as `linear_regression_medical_expense_model.pkl`

## 📦 Files Included

- `synthetic_medical_expenses_pakistan.csv` — the dataset
- `train_linear_model.py` — training script
- `linear_regression_medical_expense_model.pkl` — saved model
- `README.md` — project documentation

## 🚀 How to Use

```bash
# Load the model
import joblib
model = joblib.load("linear_regression_medical_expense_model.pkl")

# Predict new expenses
sample = {
    "Age": 40,
    "BMI": 26.5,
    "Children": 2,
    "Smoker": "No",
    "Region": "Punjab",
    "ExerciseFreq": 3,
    "ChronicConditions": 1
}
import pandas as pd
df = pd.DataFrame([sample])
prediction = model.predict(df)
print(f"Predicted Medical Expense: PKR {prediction[0]:.2f}")
