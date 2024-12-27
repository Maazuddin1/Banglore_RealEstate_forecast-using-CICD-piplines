import numpy as np
import pytest
import pickle
import sys
import os
sys.path.append(r'Banglore_RealEstate_forecast-using-CICD-piplines\src')
from model import ModelBuilder

# Helper function to load preprocessed data
def get_preprocessed_data():
    # Assuming preprocessor saves cleaned data in a specific file
    preprocessed_data_path = "data/preprocessed_data.pkl"
    if not os.path.exists(preprocessed_data_path):
        raise FileNotFoundError(f"Preprocessed data file not found at {preprocessed_data_path}")
    
    with open(preprocessed_data_path, 'rb') as file:
        return pickle.load(file)

# Test case 1: Test for Correctness of Predictions
def test_house_price_predictions():
    # Load preprocessed data
    data = get_preprocessed_data()
    
    # Create ModelBuilder instance
    model_builder = ModelBuilder(data=data)
    X_train, X_test, y_train, y_test = model_builder.split_data(target_column='price')
    
    # Train the model
    model_builder.train_model(X_train, y_train)
    
    # Example known data for prediction
    X_test_sample = np.array([
        [3, 1500, 2],  # 3 bedrooms, 1500 sqft, 2 bathrooms
        [4, 2000, 3],  # 4 bedrooms, 2000 sqft, 3 bathrooms
        [2, 1200, 1]   # 2 bedrooms, 1200 sqft, 1 bathroom
    ])
    expected_prices = [70, 100, 50]  # Replace with expected values in lakhs
    
    y_pred = model_builder.model.predict(X_test_sample)
    assert np.allclose(y_pred, expected_prices, atol=10), f"Predictions {y_pred} are not as expected"

# Test case 2: Test for Model Performance
def test_house_price_performance():
    # Load preprocessed data
    data = get_preprocessed_data()
    
    # Create ModelBuilder instance
    model_builder = ModelBuilder(data=data)
    X_train, X_test, y_train, y_test = model_builder.split_data(target_column='price')
    
    # Train and evaluate the model
    model_builder.train_model(X_train, y_train)
    mse, r2 = model_builder.evaluate_model(X_test, y_test)
    
    # Check for acceptable R2 score (e.g., above 0.7)
    assert r2 > 0.7, f"Model R2 score is below acceptable threshold: {r2}"

# Test case 3: Test Model Serialization and Loading
def test_model_serialization():
    # Load preprocessed data
    data = get_preprocessed_data()
    
    # Create ModelBuilder instance
    model_builder = ModelBuilder(data=data)
    X_train, X_test, y_train, y_test = model_builder.split_data(target_column='price')
    
    # Train the model and save it
    model_builder.train_model(X_train, y_train)
    model_path = model_builder.save_model_as_pickle()

    # Load the model and compare predictions
    loaded_model = model_builder.load_model_from_pickle(model_path)
    y_pred_original = model_builder.model.predict(X_test)
    y_pred_loaded = loaded_model.predict(X_test)
    assert np.allclose(y_pred_original, y_pred_loaded), "Predictions differ after loading the model"

# Test case 4: Test for Handling Extreme Values
def test_house_price_outliers():
    # Load preprocessed data
    data = get_preprocessed_data()
    
    # Create ModelBuilder instance
    model_builder = ModelBuilder(data=data)
    X_train, _, y_train, _ = model_builder.split_data(target_column='price')
    
    # Train the model
    model_builder.train_model(X_train, y_train)
    
    # Test for extreme values
    X_edge = np.array([[10, 10000, 10]])  # 10 bedrooms, 10,000 sqft, 10 bathrooms
    y_pred_edge = model_builder.model.predict(X_edge)
    assert np.isfinite(y_pred_edge).all(), "Prediction for outlier is not finite"

# Test case 5: Test Predictions on New Data
def test_house_price_on_new_data():
    # Load preprocessed data
    data = get_preprocessed_data()
    
    # Create ModelBuilder instance
    model_builder = ModelBuilder(data=data)
    X_train, _, y_train, _ = model_builder.split_data(target_column='price')
    
    # Train the model
    model_builder.train_model(X_train, y_train)
    
    # New test data
    X_new = np.array([
        [3, 2000, 2],  # 3 bedrooms, 2000 sqft, 2 bathrooms
        [5, 3500, 4]   # 5 bedrooms, 3500 sqft, 4 bathrooms
    ])
    y_pred_new = model_builder.model.predict(X_new)
    
    # Assert predictions fall within realistic range
    assert y_pred_new[0] >= 50 and y_pred_new[0] <= 150, f"Prediction {y_pred_new[0]} is outside expected range"
    assert y_pred_new[1] >= 100 and y_pred_new[1] <= 300, f"Prediction {y_pred_new[1]} is outside expected range"
