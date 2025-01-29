import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Initialize Flask app
application = Flask(__name__)
app = application

# Load the pre-trained models and scaler
linreg_model = pickle.load(open('models/linreg.pkl', 'rb'))
standard_scaler = pickle.load(open('models/scaler.pkl', 'rb'))

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Route for prediction
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'POST':
        try:
            # Extract input values from the form
            Temperature = float(request.form.get('Temperature'))
            RH = float(request.form.get('RH'))
            Ws = float(request.form.get('Ws'))
            Rain = float(request.form.get('Rain'))
            FFMC = float(request.form.get('FFMC'))
            DMC = float(request.form.get('DMC'))
            ISI = float(request.form.get('ISI'))
            Classes = float(request.form.get('Classes'))
            Region = float(request.form.get('Region'))

            # Scale the input data
            new_data_scaled = standard_scaler.transform([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])

            # Predict using the model
            result = linreg_model.predict(new_data_scaled)

            # Return the prediction result
            return render_template('home.html', result=result[0])

        except Exception as e:
            # If there is an error, return a helpful message
            return render_template('home.html', result=f"Error: {str(e)}")
    else:
        return render_template('home.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)  # Enable debug mode to get more helpful error messages
