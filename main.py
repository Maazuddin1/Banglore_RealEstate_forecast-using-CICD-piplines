from src.preprocessing import Preprocessing
from src.model import ModelBuilder
import pandas as pd
import pickle
import os

def main():
    # Load the dataset
    data = pd.read_csv("data/bengaluru_house_prices.csv")

    # Preprocess the data
    print("Starting Data Preprocessing...")
    preprocessor = Preprocessing(data)
    preprocessor.clean_data()
    preprocessor.feature_engineering()
    preprocessor.remove_bhk_outliers()
    preprocessor.encode_features()
    preprocessor.scale_features()
    preprocessor.handle_missing_values()
    print("Preprocessing completed!")

    # Build and evaluate the model
    print("Starting Model Building and Evaluation...")
    model_builder = ModelBuilder(data=preprocessor.data)
    X_train, X_test, y_train, y_test = model_builder.split_data(target_column='price')

    model_builder.train_model(X_train, y_train)
    mse, r2 = model_builder.evaluate_model(X_test, y_test)

    #print(f"Model Evaluation:\nMean Squared Error: {mse}\nR2 Score: {r2}")
    
    # Save the trained model
    print("Trained model saved successfully!")

    # Save the trained model as a pickle file
    model_builder.save_model_as_pickle()

    # Save the feature names as a pickle file
    model_builder.save_features_as_pickle(data=preprocessor.data)

if __name__ == "__main__":
    main()