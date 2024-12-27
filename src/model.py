import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pickle  # Import pickle for saving models
import os  # Import os for directory operations

class ModelBuilder:
    def __init__(self, data):
        """Initialize with the dataset."""
        self.data = data
        self.model = None

    def split_data(self, target_column, test_size=0.2, random_state=42):
        """Splits the data into training and testing sets."""
        if target_column not in self.data.columns:
            raise ValueError(f"Target column '{target_column}' not found in the dataset.")

        X = self.data.drop(columns=[target_column])
        y = self.data[target_column]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        #print('x_test:', X_test.head())
        #print('First 15 column names:', X_test.columns[:15])
        #print('First 15 column data:', X_test.iloc[:15, :10])
        print(f"Data split complete: Train size = {len(X_train)}, Test size = {len(X_test)}")
        return X_train, X_test, y_train, y_test

    def train_model(self, X_train, y_train):
        """Trains a Linear Regression model."""
        self.model = LinearRegression()
        self.model.fit(X_train, y_train)
        print("Model training complete.")

    def evaluate_model(self, X_test, y_test):
        """Evaluates the model on the test set."""
        if self.model is None:
            raise ValueError("Model has not been trained yet.")

        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        accuracy = self.model.score(X_test, y_test)

        print(f"Model Evaluation:\nMean Squared Error: {mse}\nR2 Score(accuracy): {r2}")
        return mse, r2


    def save_model_as_pickle(self, model_path='models/lr_regg.pkl'):
        """Save the trained model as a pickle file."""
        if self.model is None:
            raise ValueError("Model has not been trained yet.")

        # Create the models directory if it doesn't exist
        #os.makedirs(os.path.dirname(model_path), exist_ok=True)

        # Save the model
        with open(model_path, 'wb') as file:
            pickle.dump(self.model, file)

        print(f"Model saved as pickle at {model_path}")
        return model_path
    
    
    def save_features_as_pickle(self, data, target_column='price', file_path='models/feature_names.pkl'):
        """
        Extract feature names from the data and save them as a pickle file.

        Args:
            data (pd.DataFrame): Input dataset.
            target_column (str): Name of the target column to exclude from features.
            file_path (str): Path to save the pickle file.
        """
        # Ensure the target column exists
        if target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' not found in the dataset.")

        # Drop the target column and extract feature names
        feature_names = data.drop(columns=[target_column]).columns.tolist()

        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Save the feature names as a pickle file
        with open(file_path, "wb") as file:
            pickle.dump(feature_names, file)

        print(f"Feature names saved to {file_path}")

    def load_model_from_pickle(self, model_path):
        """Load a model from a pickle file."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No model found at {model_path}")

        with open(model_path, 'rb') as file:
            self.model = pickle.load(file)

        print(f"Model loaded from {model_path}")
        return self.model

