import pickle
import numpy as np

# Load model and feature names
def load_model_and_features(model_path, feature_path):
    # Load the trained model
    with open(model_path, "rb") as file:
        model = pickle.load(file)
    
    # Load the feature names
    with open(feature_path, "rb") as file:
        feature_names = pickle.load(file)

    return model, feature_names

# Predict price using the model
def predict_price(location, sqft, bath, bhk, model, feature_names):
    # Create an input array with zeros for all features
    x = np.zeros(len(feature_names))

    # Assign values for sqft, bath, and bhk
    if 'total_sqft' in feature_names:
        x[feature_names.index('total_sqft')] = sqft
    if 'bath' in feature_names:
        x[feature_names.index('bath')] = bath
    if 'bhk' in feature_names:
        x[feature_names.index('bhk')] = bhk

    # Set the location column to 1 if it exists in feature names
    if location in feature_names:
        loc_index = feature_names.index(location)
        x[loc_index] = 1

    # Make prediction
    return model.predict([x])[0]

# Test function
def test_house_price_predictions():
    # Paths to the model and feature names
    model_path = "models/lr_regg.pkl"
    feature_path = "models/feature_names.pkl"

    # Load the model and features
    model, feature_names = load_model_and_features(model_path, feature_path)

    # Test cases and expected outputs
    test_cases = [
        {"location": "Whitefield", "sqft": 1200, "bath": 2, "bhk": 2, "expected": 94},
        {"location": "Banaswadi", "sqft": 1500, "bath": 3, "bhk": 3, "expected": 118},
        {"location": "Basavangudi", "sqft": 1800, "bath": 3, "bhk": 4, "expected": 142},
        {"location": "Nonexistent Location", "sqft": 1000, "bath": 2, "bhk": 3, "expected": 79},
        {"location": "Electronic City Phase II", "sqft": 1056, "bath": 2, "bhk": 2, "expected": 83},
        {"location": "Chikka Tirupathi", "sqft": 800, "bath": 2, "bhk": 2, "expected": 63}
    ]

    # Run predictions and validate against expected outputs
    for case in test_cases:
        location = case["location"]
        sqft = case["sqft"]
        bath = case["bath"]
        bhk = case["bhk"]
        expected = case["expected"]

        try:
            predicted_price = predict_price(location, sqft, bath, bhk, model, feature_names)
            assert round(predicted_price / 10) == expected, (
                f"Failed for Location: {location}, "
                f"Expected: {expected}, Got: {predicted_price/10:.0f} lakhs"
            )
            print(f"Test Passed: Location: {location}, Predicted: {predicted_price/10:.0f} lakhs")
        except Exception as e:
            print(f"Prediction failed for Location: {location}, Error: {e}")

# Run the tests
if __name__ == "__main__":
    test_house_price_predictions()
