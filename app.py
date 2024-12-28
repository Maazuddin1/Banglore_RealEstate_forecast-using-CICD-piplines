from flask import Flask, request, jsonify
import pickle
import numpy as np
import os

# Load the trained model
MODEL_PATH = 'models/lr_regg.pkl'
FEATURES_PATH = 'models/feature_names.pkl'

if not os.path.exists(MODEL_PATH) or not os.path.exists(FEATURES_PATH):
    raise FileNotFoundError("Model or feature files not found. Please train the model first.")

with open(MODEL_PATH, 'rb') as model_file:
    model = pickle.load(model_file)

with open(FEATURES_PATH, 'rb') as feature_file:
    feature_names = pickle.load(feature_file)

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    """
    Accepts JSON data with features and returns the predicted price.
    """
    try:
        data = request.get_json()
        
        # Ensure all required features are provided
        input_features = []
        for feature in feature_names:
            if feature in data:
                input_features.append(data[feature])
            else:
                return jsonify({'error': f'Missing feature: {feature}'}), 400

        # Convert to numpy array and reshape
        input_array = np.array(input_features).reshape(1, -1)
        
        # Make a prediction
        prediction = model.predict(input_array)
        return jsonify({'predicted_price': prediction[0]})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/', methods=['GET'])
def home():
    """
    Home route to check if the API is running.
    """
    return "Welcome to the Real Estate Price Prediction API!", 200


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
