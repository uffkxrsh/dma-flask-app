import numpy as np
from joblib import load
from XGBoostPolyModel import XGBPolynomialEnsembleNoTuning
import pandas as pd

# Load label encoding dictionaries
label_encoder = load('label_encoder.joblib')
print("Data type of label_encoder:", type(label_encoder))

# Load scaler
scaler = load('scaler.joblib')
print("Data type of scaler:", type(scaler))

# Load the trained model
model = load('ensemble_model_no_tuning.joblib')
print("Data type of model:", type(model))

# Example raw data point
raw_data_point = {
    'carat': 0.3,
    'cut': 'Ideal',
    'color': 'E',
    'clarity': 'SI2',
    'depth': 61.5,
    'table': 55,
    'x': 4.2,
    'y': 4.25,
    'z': 2.6
}

df = pd.DataFrame([raw_data_point])

for column in ['cut', 'color', 'clarity', 'depth', 'carat', 'table', 'x', 'y', 'z']:
    df[column] = label_encoder.fit_transform(df[column])

# Scale the features
scaled_data_point = scaler.transform(df)

# Predict the price using the model
predicted_price = model.predict(scaled_data_point)

print("Predicted Price:", predicted_price[0])
