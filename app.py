from flask import Flask, request, render_template, redirect, url_for
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load the model and feature names
def load_model_and_features():
    try:
        model_path = os.path.join(os.getcwd(), "models", "lr_regg.pkl")
        feature_path = os.path.join(os.getcwd(), "models", "feature_names.pkl")

        with open(model_path, "rb") as file:
            model = pickle.load(file)

        with open(feature_path, "rb") as file:
            feature_names = pickle.load(file)

        return model, feature_names
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Required file not found: {str(e)}")

def predict_price(location, sqft, bath, bhk, model, feature_names):
    x = np.zeros(len(feature_names))

    if 'total_sqft' in feature_names:
        x[feature_names.index('total_sqft')] = sqft
    if 'bath' in feature_names:
        x[feature_names.index('bath')] = bath
    if 'bhk' in feature_names:
        x[feature_names.index('bhk')] = bhk
    if location in feature_names:
        loc_index = feature_names.index(location)
        x[loc_index] = 1
    else:
        raise ValueError(f"Location '{location}' is not recognized.")

    return model.predict([x])[0]

# Load model and features when app starts
model, feature_names = load_model_and_features()

# Get list of locations (excluding non-location features)
locations = [feat for feat in feature_names if feat not in ['total_sqft', 'bath', 'bhk']]
locations.sort()

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        try:
            location = request.form.get('location')
            sqft = float(request.form.get('sqft'))
            bath = int(request.form.get('bath'))
            bhk = int(request.form.get('bhk'))

            price = predict_price(location, sqft, bath, bhk, model, feature_names)
            prediction = round(price / 10, 2)  # Convert to lakhs and round to 2 decimal places

            # Redirect to results page with the prediction details
            return redirect(url_for('results',
                                    predicted_price=prediction,
                                    location=location,
                                    sqft=sqft,
                                    bath=bath,
                                    bhk=bhk))
        except Exception as e:
            error = f"Error: {str(e)}"
            return render_template('index.html', locations=locations, error=error)

    return render_template('index.html', locations=locations)

@app.route('/results')
def results():
    # Fetch query parameters from the URL
    predicted_price = request.args.get('predicted_price')
    location = request.args.get('location')
    sqft = request.args.get('sqft')
    bath = request.args.get('bath')
    bhk = request.args.get('bhk')

    return render_template('results.html',
                           predicted_price=predicted_price,
                           location=location,
                           sqft=sqft,
                           bath=bath,
                           bhk=bhk)

if __name__ == '__main__':
    debug_mode = os.getenv('FLASK_DEBUG', 'False').lower() in ['true', '1', 'yes']
    app.run(debug=debug_mode)
