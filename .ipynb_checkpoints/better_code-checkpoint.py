import os 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Read data
diamonds =  pd.read_csv('./data/diamonds.csv')

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

# Label Encoding
label_encoder = LabelEncoder()

for column in ['cut', 'color', 'clarity', 'depth', 'carat', 'table']:
    diamonds[column] = label_encoder.fit_transform(diamonds[column])

# Split data
X = diamonds.drop(columns='price').values
y = diamonds['price'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling Features 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define Ensemble Class
class XGBPolynomialEnsemble:
    def __init__(self, xgb_params={}, poly_degree=2):
        self.xgb_model = Pipeline([("xgb", XGBRegressor(**xgb_params))])
        self.poly_reg = Pipeline([("pf", PolynomialFeatures(degree=poly_degree)), ("linear_reg", LinearRegression())])

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

# Instantiate ensemble model
ensemble_model = XGBPolynomialEnsemble(xgb_params={"learning_rate": 0.1, "max_depth": 15, "n_estimators": 200}, poly_degree=2)

# Fit the ensemble model
ensemble_model.fit(X_train_scaled, y_train)

# Hyperparameter tuning for XGBoost
param_grid_xgb = {
    'xgb__learning_rate': [0.05, 0.1, 0.2],
    'xgb__max_depth': [10, 15, 20],
    'xgb__n_estimators': [100, 200, 300]
}

grid_search_xgb = GridSearchCV(ensemble_model.xgb_model, param_grid_xgb, cv=5, scoring='neg_mean_squared_error')
grid_search_xgb.fit(X_train_scaled, y_train)
best_params_xgb = grid_search_xgb.best_params_

# Hyperparameter tuning for Polynomial Regression
param_grid_poly = {
    'pf__degree': [1, 2, 3]
}

grid_search_poly = GridSearchCV(ensemble_model.poly_reg, param_grid_poly, cv=5, scoring='neg_mean_squared_error')
grid_search_poly.fit(X_train_scaled, y_train)
best_poly_degree = grid_search_poly.best_params_['pf__degree']

# Re-initialize the ensemble model with best hyperparameters
ensemble_model_tuned = XGBPolynomialEnsemble(xgb_params=best_params_xgb, poly_degree=best_poly_degree)

# Fit the tuned ensemble model
ensemble_model_tuned.fit(X_train_scaled, y_train)

# Predict using the tuned ensemble model
y_pred_ensemble_tuned = ensemble_model_tuned.predict(X_test_scaled)

# Calculate evaluation metrics on the test set
mse_ensemble_tuned = mean_squared_error(y_test, y_pred_ensemble_tuned)
mae_ensemble_tuned = mean_absolute_error(y_test, y_pred_ensemble_tuned)

print(f"Tuned Ensemble Test Set MSE: {mse_ensemble_tuned}")
print(f"Tuned Ensemble Test Set MAE: {mae_ensemble_tuned}")
print(f"Tuned Ensemble Test Set NRMSE: {-np.sqrt(mse_ensemble_tuned)}")
