from flask import Flask, request, jsonify, render_template
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

@app.route('/')
def home():
    """
    Home route that displays a form where the user can input values to predict the price.
    Also displays a suggestion bar for users to understand what data is needed.
    """
    return render_template('index.html', feature_names=feature_names)


@app.route('/predict', methods=['POST'])
def predict():
    """
    Accepts form data from the user, processes it, and returns the predicted price.
    """
    try:
        # Get form data from the HTML form
        data = request.form.to_dict()
        
        # Ensure all required features are provided
        input_features = []
        missing_features = []
        for feature in feature_names:
            if feature in data:
                input_features.append(float(data[feature]))
            else:
                missing_features.append(feature)
        
        if missing_features:
            return jsonify({'error': f'Missing features: {", ".join(missing_features)}'}), 400

        # Convert to numpy array and reshape for prediction
        input_array = np.array(input_features).reshape(1, -1)
        
        # Make a prediction
        prediction = model.predict(input_array)
        
        return render_template('result.html', prediction=prediction[0])

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
