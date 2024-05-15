import os 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error

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

# Split data
X = diamonds.drop(columns='price').values
y = diamonds['price'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling Features 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Building ensemble model...")

class XGBRandomForestEnsemble:
    def __init__(self, xgb_params={}, rf_params={}):
        self.xgb_params = xgb_params
        self.rf_params = rf_params
        self.xgb_model = XGBRegressor(**self.xgb_params)
        self.rf_model = RandomForestRegressor(**self.rf_params)

    def fit(self, X_train, y_train):
        self.xgb_model.fit(X_train, y_train)
        xgb_train_preds = self.xgb_model.predict(X_train).reshape(-1, 1)
        self.rf_model.fit(np.concatenate((X_train, xgb_train_preds), axis=1), y_train)

    def predict(self, X_test):
        xgb_test_preds = self.xgb_model.predict(X_test).reshape(-1, 1)
        return self.rf_model.predict(np.concatenate((X_test, xgb_test_preds), axis=1))

    def tune_rf_hyperparameters(self, X_train, y_train, param_grid_rf):
        print("Tuning Random Forest hyperparameters...")
        grid_search_rf = GridSearchCV(self.rf_model, param_grid_rf, cv=5, scoring='neg_mean_squared_error')
        grid_search_rf.fit(X_train, y_train)
        self.rf_params = grid_search_rf.best_params_
        self.rf_model.set_params(**self.rf_params)

    def fit_with_tuned_hyperparameters(self, X_train, y_train):
        print("Fitting ensemble model with tuned hyperparameters...")
        self.xgb_model.fit(X_train, y_train)
        xgb_train_preds = self.xgb_model.predict(X_train).reshape(-1, 1)
        self.rf_model.fit(np.concatenate((X_train, xgb_train_preds), axis=1), y_train)

    def save_model(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    @classmethod
    def load_model(cls, filename):
        with open(filename, 'rb') as file:
            return pickle.load(file)

# Instantiate ensemble model
ensemble_model = XGBRandomForestEnsemble(xgb_params={"learning_rate": 0.1, "max_depth": 15, "n_estimators": 200}, 
                                         rf_params={"n_estimators": 100, "max_depth": None})

# Hyperparameters for Random Forest
rf_hyperparameters = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20]
}

# Tune Random Forest hyperparameters
ensemble_model.tune_rf_hyperparameters(X_train_scaled, y_train, rf_hyperparameters)

# Fit the ensemble model with tuned hyperparameters
ensemble_model.fit_with_tuned_hyperparameters(X_train_scaled, y_train)

# Save the trained model
ensemble_model.save_model('ensemble_model_rf.pkl')

print("Model trained and saved successfully.")

# Load the saved model
loaded_model_rf = XGBRandomForestEnsemble.load_model('ensemble_model_rf.pkl')

print("Model loaded successfully.")

# Predict using the loaded model
y_pred_loaded_model_rf = loaded_model_rf.predict(X_test_scaled)

# Calculate evaluation metrics on the test set
mse_loaded_model_rf = mean_squared_error(y_test, y_pred_loaded_model_rf)
mae_loaded_model_rf = mean_absolute_error(y_test, y_pred_loaded_model_rf)

print(f"Loaded Model (Random Forest) Test Set MSE: {mse_loaded_model_rf}")
print(f"Loaded Model (Random Forest) Test Set MAE: {mae_loaded_model_rf}")
print(f"Loaded Model (Random Forest) Test Set NRMSE: {-np.sqrt(mse_loaded_model_rf)}")
