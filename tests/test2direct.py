import pickle
import numpy as np

def load_model_and_features(model_path, feature_path):
    """
    Load the trained model and feature names from pickle files.

    Args:
        model_path (str): Path to the trained model pickle file.
        feature_path (str): Path to the feature names pickle file.

    Returns:
        tuple: (trained model, feature names)
    """
    # Load the trained model
    with open(model_path, "rb") as file:
        model = pickle.load(file)
    
    # Load the feature names
    with open(feature_path, "rb") as file:
        feature_names = pickle.load(file)

    return model, feature_names

def predict_price(location, sqft, bath, bhk, model, feature_names):
    """
    Predict the price using the trained model.

    Args:
        location (str): Location name.
        sqft (float): Total square footage.
        bath (int): Number of bathrooms.
        bhk (int): Number of bedrooms.
        model: Trained model object.
        feature_names (list): List of feature names.

    Returns:
        float: Predicted price.
    """
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

def main():
    # Paths to the model and feature names
    model_path = "models/lr_regg.pkl"
    feature_path = "models/feature_names.pkl"

    # Load the model and features
    model, feature_names = load_model_and_features(model_path, feature_path)

    # Test cases
    test_cases = [
        {"location": "Whitefield", "sqft": 1200, "bath": 2, "bhk": 2},
        {"location": "Banaswadi", "sqft": 1500, "bath": 3, "bhk": 3},
        {"location": "Basavangudi", "sqft": 1800, "bath": 3, "bhk": 4},
        {"location": "Nonexistent Location", "sqft": 1000, "bath": 2, "bhk": 3},
        {"location": "Electronic City Phase II", "sqft": 1056, "bath": 2, "bhk": 2},
        {"location": "Chikka Tirupathi", "sqft": 800, "bath": 2, "bhk": 2}
    ]

    print("\nPredictions:")
    for case in test_cases:
        location = case["location"]
        sqft = case["sqft"]
        bath = case["bath"]
        bhk = case["bhk"]

        try:
            predicted_price = predict_price(location, sqft, bath, bhk, model, feature_names)
            print(f"Location: {location}, Sqft: {sqft}, Bath: {bath}, BHK: {bhk} -> Predicted Price: {predicted_price/10:.0f} lakhs")
        except Exception as e:
            print(f"Prediction failed for Location: {location}, Error: {e}")

if __name__ == "__main__":
    main()