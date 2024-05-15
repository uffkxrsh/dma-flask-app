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
import pickle

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
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

# Split data
X = diamonds.drop(columns='price').values
y = diamonds['price'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling Features 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save StandardScaler
with open('standard_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("Building ensemble model...")

# Define Ensemble Class
class XGBPolynomialEnsemble:
    def __init__(self, xgb_params={}, poly_degree=2):
        self.xgb_params = xgb_params
        self.poly_degree = poly_degree
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
    
    def tune_xgb_hyperparameters(self, X_train, y_train, param_grid_xgb):
        print("Tuning XGBoost hyperparameters...")
        # Modify the parameter names to include 'xgb__'
        param_grid_xgb = {'xgb__' + key: value for key, value in param_grid_xgb.items()}
        grid_search_xgb = GridSearchCV(self.xgb_model, param_grid_xgb, cv=5, scoring='neg_mean_squared_error')
        grid_search_xgb.fit(X_train, y_train)
        self.xgb_params = grid_search_xgb.best_params_
        self.xgb_model.set_params(**self.xgb_params)
        
    def tune_poly_degree(self, X_train, y_train, param_grid_poly):
        print("Tuning Polynomial Regression degree...")
        param_grid_poly = {'pf__degree': param_grid_poly['degree']}  # Specify the parameter for PolynomialFeatures
        grid_search_poly = GridSearchCV(self.poly_reg, param_grid_poly, cv=5, scoring='neg_mean_squared_error')
        grid_search_poly.fit(X_train, y_train)
        self.poly_degree = grid_search_poly.best_params_['pf__degree']
        self.poly_reg.set_params(pf__degree=self.poly_degree)

    
    def fit_with_tuned_hyperparameters(self, X_train, y_train):
        print("Fitting ensemble model with tuned hyperparameters...")
        # Fit the XGBoost model with tuned hyperparameters
        self.xgb_model.fit(X_train, y_train)
        
        # Predictions of XGBoost model on training set
        xgb_train_preds = self.xgb_model.predict(X_train)
        
        # Scale XGBoost predictions
        xgb_train_preds_scaled = scaler.fit_transform(xgb_train_preds.reshape(-1, 1))
        
        # Add scaled predictions as a feature
        X_train_with_xgb_preds = np.concatenate((X_train, xgb_train_preds_scaled), axis=1)
        
        # Fit Polynomial Regression on the combined features with tuned hyperparameters
        self.poly_reg.fit(X_train_with_xgb_preds, y_train)
    
    def save_model(self, filename):
        # Create a dictionary to store the model and hyperparameters
        model_data = {
            'model': self,
            'xgb_params': self.xgb_params,
            'poly_degree': self.poly_degree
        }
        # Save the model data to a file
        with open(filename, 'wb') as file:
            pickle.dump(model_data, file)
    
    @classmethod
    def load_model(cls, filename):
        # Load the model data from the file
        with open(filename, 'rb') as file:
            model_data = pickle.load(file)
        # Extract the model and hyperparameters
        model = model_data['model']
        xgb_params = model_data['xgb_params']
        poly_degree = model_data['poly_degree']
        # Set the hyperparameters in the model object
        model.xgb_params = xgb_params
        model.poly_degree = poly_degree
        return model


# Instantiate ensemble model
ensemble_model = XGBPolynomialEnsemble(xgb_params={"learning_rate": 0.1, "max_depth": 15, "n_estimators": 200}, poly_degree=2)

# Hyperparameters for XGBoost
xgb_hyperparameters = {
    'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
    'max_depth': [3, 5, 7, 10, 15],
    'n_estimators': [50, 100, 150, 200, 250, 300]
}

# Hyperparameters for Polynomial Regression degree
poly_hyperparameters = {
    'degree': [1, 2, 3]
}

# Tune XGBoost hyperparameters
ensemble_model.tune_xgb_hyperparameters(X_train_scaled, y_train, xgb_hyperparameters)

# Tune Polynomial Regression degree
ensemble_model.tune_poly_degree(X_train_scaled, y_train, poly_hyperparameters)

# Fit the ensemble model with tuned hyperparameters
ensemble_model.fit_with_tuned_hyperparameters(X_train_scaled, y_train)

# Save the trained model
ensemble_model.save_model('ensemble_model_poly.pkl')

print("Model trained and saved successfully.")

# Load the saved model
loaded_model = XGBPolynomialEnsemble.load_model('ensemble_model_poly.pkl')

print("Model loaded successfully.")

# Predict using the loaded model
y_pred_loaded_model = loaded_model.predict(X_test_scaled)

# Calculate evaluation metrics on the test set
mse_loaded_model = mean_squared_error(y_test, y_pred_loaded_model)
mae_loaded_model = mean_absolute_error(y_test, y_pred_loaded_model)

print(f"Loaded Model Test Set MSE: {mse_loaded_model}")
print(f"Loaded Model Test Set MAE: {mae_loaded_model}")
print(f"Loaded Model Test Set NRMSE: {-np.sqrt(mse_loaded_model)}")
