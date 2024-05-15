from flask import Flask, render_template, request, jsonify
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



app=Flask(__name__)

# diamond_price_predictor = pickle.load(open('ensemble_model_rf.pkl', 'rb'))

@app.route('/') 
def home():
    return render_template("index.html")

@app.route('/features') 
def diabetes():
    return render_template("features.html")

@app.route('/diamonds') 
def diamond():
    return render_template("predict.html")

@app.route('/diamonds', methods=['POST'])
def predictprice():
    if request.method == 'POST':
        try:
            # Getting the values from the form
            carat = float(request.form['carat'])
            color = request.form['color']
            depth = float(request.form['depth'])
            table = float(request.form['table'])
            clarity = request.form['clarity']
            cut = request.form['cut']
            x = float(request.form['x'])
            y = float(request.form['y'])
            z = float(request.form['z'])

            # Example raw data point
            data_point = {
                'carat': carat,
                'cut': cut,
                'color': color,
                'clarity': clarity,
                'depth': depth,
                'table': table,
                'x': x,
                'y': y,
                'z': z
            }

            df = pd.DataFrame([data_point])

            # Label encoding for categorical variables
            for column in ['cut', 'color', 'clarity']:
                df[column] = label_encoder.fit_transform(df[column])

            # Scale the features
            scaled_data_point = scaler.transform(df)

            # Predict the price using the model
            predicted_price = model.predict(scaled_data_point)[0]
            
            formatted_price = "{:.2f}".format(predicted_price)

            return render_template('predict.html', message=formatted_price)
        
        except Exception as e:
            print("Error:", e)
            return render_template('predict.html', message="error >_<")

    else:
        return render_template('predict.html', message="error >_<")


if __name__=="__main__":
    app.run(debug=True)