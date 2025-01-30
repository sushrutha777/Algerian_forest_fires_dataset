import pickle
import os
from flask import Flask, request, render_template
import numpy as np
from sklearn.preprocessing import StandardScaler

# Initialize Flask app
application = Flask(__name__)
app = application

# Load the pre-trained models and scaler safely
model_path = 'models/linreg.pkl'
scaler_path = 'models/scaler.pkl'

if os.path.exists(model_path) and os.path.exists(scaler_path):
    linreg_model = pickle.load(open(model_path, 'rb'))
    standard_scaler = pickle.load(open(scaler_path, 'rb'))
else:
    linreg_model = None
    standard_scaler = None

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Route to display the prediction form
@app.route('/home')
def home():
    return render_template('home.html')

# Route for prediction
@app.route('/predictdata', methods=['POST'])
def predict_datapoint():
    if not linreg_model or not standard_scaler:
        return render_template('home.html', result="Error: Model files are missing!")

    try:
        # Fetch input values
        Temperature = float(request.form.get('Temperature', '0'))
        RH = float(request.form.get('RH', '0'))
        Ws = float(request.form.get('Ws', '0'))
        Rain = float(request.form.get('Rain', '0'))
        FFMC = float(request.form.get('FFMC', '0'))
        DMC = float(request.form.get('DMC', '0'))
        ISI = float(request.form.get('ISI', '0'))
        Classes = float(request.form.get('Classes', '0'))
        Region = request.form.get('Region', '').strip().lower()  # Normalize region input

        # Map the Region string to a numerical value
        region_mapping = {'north': 0, 'south': 1, 'east': 2, 'west': 3}
        Region = region_mapping.get(Region, -1)  # Default to -1 if not found

        if Region == -1:
            return render_template('home.html', result="Error: Invalid Region. Choose from north, south, east, west.")

        # Scale the input features
        new_data_scaled = standard_scaler.transform([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])
        result = linreg_model.predict(new_data_scaled)

        return render_template('home.html', result=f"THE FWI prediction is {result[0]:.2f}")

    except Exception as e:
        return render_template('home.html', result=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)  # Debug mode enabled for better error reporting
