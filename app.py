from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd  # Import pandas to read the CSV file

# Load the model and scaler using joblib
model = joblib.load('crop_recommendation_model.pkl')  # Crop recommendation model
scaler = joblib.load('scaler.pkl')  # Scaler used during training
label_encoder = joblib.load('label_encoder.pkl')  # Label encoder for decoding predictions

# Load the CSV file containing crop data
crop_data = pd.read_csv('Crop_recommendation.csv')

app = Flask(__name__)

@app.route('/')
def index():
    # Convert CSV data to a list of dictionaries to pass to the template
    table_data = crop_data.to_dict(orient='records')
    return render_template('index.html', table_data=table_data)

@app.route('/predict', methods=['POST'])
def predict_crop():
    # Get input values from the form
    try:
        Nitrogen = float(request.form.get('Nitrogen'))
        Phosphorus = float(request.form.get('Phosphorus'))
        Potassium = float(request.form.get('Potassium'))
        Temperature = float(request.form.get('Temperature'))
        Humidity = float(request.form.get('Humidity'))
        pH = float(request.form.get('pH'))
        Rainfall = float(request.form.get('Rainfall'))
    except (ValueError, TypeError) as e:
        print(f"Error in input data: {e}")
        return render_template('index.html', result="Invalid input data.")

    # Create the input array
    input_features = np.array([[Nitrogen, Phosphorus, Potassium, Temperature, Humidity, pH, Rainfall]])

    try:
        # Apply scaling to the input data
        input_features_scaled = scaler.transform(input_features)
    except Exception as e:
        print(f"Error in scaling input: {e}")
        return render_template('index.html', result="Error during scaling.")

    try:
        # Make prediction using the scaled data
        result_numeric = model.predict(input_features_scaled)
        # Decode the prediction to a readable label
        result_label = label_encoder.inverse_transform(result_numeric)[0]
    except Exception as e:
        print(f"Error during prediction or label decoding: {e}")
        return render_template('index.html', result="Prediction error.")

    # Pass table data again after prediction
    table_data = crop_data.to_dict(orient='records')
    return render_template('index.html', result=result_label, table_data=table_data)

if __name__ == '__main__':
    app.run(port=4000, debug=True)
