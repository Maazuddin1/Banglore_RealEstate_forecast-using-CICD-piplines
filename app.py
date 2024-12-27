from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.externals import joblib

# Initialize the Flask app
app = Flask(__name__)

# Load the pre-trained model and preprocessing pipeline
model = joblib.load("models/house_price_model.pkl")  # Replace with your model file
#preprocessor = joblib.load("models/preprocessor.pkl")  # Replace with your preprocessor file

@app.route('/', methods=['GET'])
def home():
    return "Banglore RealEstate forecasting API !"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse the input JSON data
        data = request.get_json()

        # Convert the input data into a DataFrame
        #input_data = pd.DataFrame([data])

        # Preprocess the input data
        #processed_data = preprocessor.transform(input_data)

        # Make predictions
        predictions = model.predict(data)

        # Return predictions as JSON
        return jsonify({"predicted_price": float(predictions[0])})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
