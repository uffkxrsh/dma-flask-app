import os 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib

print("Loading data...")
# Read data
diamonds =  pd.read_csv('./data/diamonds.csv')

print("Data loaded successfully.")

# Drop index column
diamonds.drop(diamonds.columns[0], axis=1, inplace=True)

# Drop dimensionless diamonds
diamonds = diamonds[(diamonds["x"] > 0) & (diamonds["y"] > 0) & (diamonds["z"] > 0)]

# Remove outliers
diamonds = diamonds[(diamonds["depth"] < 75) & (diamonds["depth"] > 45)]
diamonds = diamonds[(diamonds["table"] < 80) & (diamonds["table"] > 40)]
diamonds = diamonds[(diamonds["x"] < 30)]
diamonds = diamonds[(diamonds["y"] < 30)]
diamonds = diamonds[(diamonds["z"] < 30) & (diamonds["z"] > 2)]

# Move 'price' column to the end
price_column = diamonds['price']
diamonds = diamonds.drop(columns=['price']) 
diamonds['price'] = price_column

print("Preprocessing data...")

# Label Encoding
label_encoder = LabelEncoder()

for column in ['cut', 'color', 'clarity', 'depth', 'carat', 'table', 'x', 'y', 'z']:
    diamonds[column] = label_encoder.fit_transform(diamonds[column])

# Save LabelEncoder
joblib.dump(label_encoder, 'label_encoder.joblib')

# Split data
X = diamonds.drop(columns='price').values
y = diamonds['price'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling Features 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save StandardScaler
joblib.dump(scaler, 'scaler.joblib')

print("Building ensemble model...")

# Define Ensemble Class
class XGBPolynomialEnsembleNoTuning:
    def __init__(self):
        self.xgb_params = {"learning_rate": 0.05, "max_depth": 7, "n_estimators": 250}
        self.poly_degree = 2
        self.xgb_model = Pipeline([
            ("xgb", XGBRegressor(**self.xgb_params))
        ])
        self.poly_reg = Pipeline([
            ("pf", PolynomialFeatures(degree=self.poly_degree)), 
            ("linear_reg", LinearRegression())
        ])

    def fit(self, X_train, y_train):
        # Fit the XGBoost model
        self.xgb_model.fit(X_train, y_train)
        
        # Predictions of XGBoost model on training set
        xgb_train_preds = self.xgb_model.predict(X_train)
        
        # Scale XGBoost predictions
        xgb_train_preds_scaled = scaler.fit_transform(xgb_train_preds.reshape(-1, 1))
        
        # Add scaled predictions as a feature
        X_train_with_xgb_preds = np.concatenate((X_train, xgb_train_preds_scaled), axis=1)
        
        # Fit Polynomial Regression on the combined features
        self.poly_reg.fit(X_train_with_xgb_preds, y_train)

    def predict(self, X_test):
        # Predictions of XGBoost model on test set
        xgb_test_preds = self.xgb_model.predict(X_test)
        
        # Scale XGBoost predictions
        xgb_test_preds_scaled = scaler.transform(xgb_test_preds.reshape(-1, 1))
        
        # Add scaled predictions as a feature
        X_test_with_xgb_preds = np.concatenate((X_test, xgb_test_preds_scaled), axis=1)
        
        # Predict using Polynomial Regression
        return self.poly_reg.predict(X_test_with_xgb_preds)

print("Training and evaluating the model...")

# Instantiate ensemble model with specified hyperparameters
ensemble_model_no_tuning = XGBPolynomialEnsembleNoTuning()

# Fit the ensemble model
ensemble_model_no_tuning.fit(X_train_scaled, y_train)

# Predict using the ensemble model
y_pred_no_tuning = ensemble_model_no_tuning.predict(X_test_scaled)

# Calculate evaluation metrics on the test set
mse_no_tuning = mean_squared_error(y_test, y_pred_no_tuning)
mae_no_tuning = mean_absolute_error(y_test, y_pred_no_tuning)

print(f"Ensemble Model (No Tuning) Test Set MSE: {mse_no_tuning}")
print(f"Ensemble Model (No Tuning) Test Set MAE: {mae_no_tuning}")
print(f"Ensemble Model (No Tuning) Test Set NRMSE: {-np.sqrt(mse_no_tuning)}")

# Save the trained model
joblib.dump(ensemble_model_no_tuning, 'ensemble_model_no_tuning.joblib')

print("Model, scaler, and label encoder saved successfully.")

print("Saving label encoding dictionaries...")

# Create a dictionary to store label encoders for each column
label_encoders_dict = {}

# Store label encoders for each column
for column in ['cut', 'color', 'clarity', 'depth', 'carat', 'table', 'x', 'y', 'z']:
    label_encoders_dict[column] = label_encoder.classes_

# Save the label encoders dictionary
joblib.dump(label_encoders_dict, 'label_encoders_dict.joblib')

print("Label encoding dictionaries saved successfully.")
