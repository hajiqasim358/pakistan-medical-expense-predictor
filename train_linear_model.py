import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# Load the dataset
data = pd.read_csv('synthetic_medical_expenses_pakistan.csv') # Ensure the CSV file is in the same directory or provide the correct path

# Define features and target variable
X = data.drop('MedicalExpenses', axis=1)
y = data['MedicalExpenses']

# Preprocessing for categorical data
categorical_features = ["Smoker", "Region"]
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop= "first"), categorical_features)
        ], 
        remainder="passthrough")#keep numeric values as is

# Create a pipeline with preprocessing and model
model_pipeline = Pipeline(steps=[("preprocessor", preprocessor),
                        ("regressor", LinearRegression())]) 

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model_pipeline.fit(X_train, y_train)    

#save the model to disk
joblib.dump(model_pipeline, 'linear_regression_medical_expense_model.pkl')

# Evaluate the model
r2_score = model_pipeline.score(X_test, y_test)
print(f"Model R^2 Score on test set: {r2_score:.4f}")