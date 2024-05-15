import os 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score


diamonds =  pd.read_csv('./data/diamonds.csv')
diamonds.sample(10)

diamonds.drop(diamonds.columns[0], axis=1, inplace=True)

#Dropping dimentionless diamonds
diamonds = diamonds.drop(diamonds[diamonds["x"]==0].index)
diamonds = diamonds.drop(diamonds[diamonds["y"]==0].index)
diamonds = diamonds.drop(diamonds[diamonds["z"]==0].index)

# Dropping the outliers from the diamonds dataset
diamonds = diamonds[(diamonds["depth"] < 75) & (diamonds["depth"] > 45)]
diamonds = diamonds[(diamonds["table"] < 80) & (diamonds["table"] > 40)]
diamonds = diamonds[(diamonds["x"] < 30)]
diamonds = diamonds[(diamonds["y"] < 30)]
diamonds = diamonds[(diamonds["z"] < 30) & (diamonds["z"] > 2)]

# Moving 'price' column to the end
price_column = diamonds['price']
diamonds = diamonds.drop(columns=['price']) 
diamonds['price'] = price_column

# Label Encoding
label_encoder = LabelEncoder()

diamonds['cut'] = label_encoder.fit_transform(diamonds['cut'])
diamonds['color'] = label_encoder.fit_transform(diamonds['color'])
diamonds['clarity'] = label_encoder.fit_transform(diamonds['clarity'])
diamonds['depth'] = label_encoder.fit_transform(diamonds['depth'])
diamonds['x'] = label_encoder.fit_transform(diamonds['x'])
diamonds['y'] = label_encoder.fit_transform(diamonds['y'])
diamonds['z'] = label_encoder.fit_transform(diamonds['z'])
diamonds['carat'] = label_encoder.fit_transform(diamonds['carat'])
diamonds['table'] = label_encoder.fit_transform(diamonds['table'])

price_column = diamonds['price']
X = diamonds.drop(columns='price') 
y = diamonds['price']

#Scaling Features 
scaler = StandardScaler()
X = scaler.fit_transform(X)

#Splitting the Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

xgb_model = Pipeline([("xgb", XGBRegressor(learning_rate=0.1, max_depth=15, n_estimators=200))])
xgb_model .fit(X_train, y_train)

# Perform cross-validation
cv_scores = cross_val_score(xgb_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')

# Calculate the average cross-validation score
avg_cv_score = np.mean(cv_scores)

# Fit the model on the entire training data
xgb_model.fit(X_train, y_train)

# Predict on the test set
y_pred_test = xgb_model.predict(X_test)

# Calculate evaluation metrics on the test set
mse_test = mean_squared_error(y_test, y_pred_test)
mae_test = mean_absolute_error(y_test, y_pred_test)

print(f"Average Cross-Validation MSE: {avg_cv_score}")
print(f"Test Set MSE: {mse_test}")
print(f"Test Set MAE: {mae_test}")

y_pred = xgb_model.predict(X)

# Adding Predictions to the dataset
y_pred = pd.DataFrame(scaler.fit_transform(pd.DataFrame(y_pred)))
X = pd.DataFrame(X)
X['y_pred'] = y_pred
X = X.values
y = y.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Ensembling Polynomial Regression with Degree 2
poly_reg_deg2 = Pipeline([("pf2", PolynomialFeatures(degree=2)), ("linear_reg", LinearRegression())])
poly_reg_deg2.fit(X_train, y_train)
price_prediction = poly_reg_deg2.predict(X)

mse = mean_squared_error(y, price_prediction)
mae = mean_absolute_error(y, price_prediction)

print(f"mse : {mse}")
print(f"mae : {mae}")
print(f"neg_rmse : {-np.sqrt(mse)}")