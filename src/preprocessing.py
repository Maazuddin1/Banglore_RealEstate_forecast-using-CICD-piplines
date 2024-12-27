import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

class Preprocessing:
    def __init__(self, data):
        """Initialize with the dataset."""
        self.data = data

    def clean_data(self):
        """Cleans and preprocesses the dataset."""
        # Drop duplicates
        self.data = self.data.drop_duplicates()
        self.data = self.data.drop(['area_type', 'availability', 'society', 'balcony'], axis=1)
        self.data=self.data.dropna()

        # Drop rows with missing target values
        if 'price' in self.data.columns:
            self.data = self.data.dropna(subset=['price'])

        # Fill missing values for numerical columns with median
        numeric_cols = self.data.select_dtypes(include=['int64', 'float64']).columns
        self.data[numeric_cols] = self.data[numeric_cols].fillna(self.data[numeric_cols].median())

        # Fill missing values for categorical columns with mode
        categorical_cols = self.data.select_dtypes(include=['object']).columns
        self.data[categorical_cols] = self.data[categorical_cols].fillna(self.data[categorical_cols].mode().iloc[0])

        # Group rare locations
        if 'location' in self.data.columns:
            location_stats = self.data['location'].value_counts()
            location_stats_lessthan_10 = location_stats[location_stats <= 10]
            self.data['location'] = self.data['location'].apply(
                lambda x: 'other' if x in location_stats_lessthan_10 else x
            )
        return self.data

    def convert_rangesqft_to_avg(self, x):
        """Convert ' - ' separated range sqftarea values to an average."""
        token = x.split('-')
        if len(token) == 2:
            return (float(token[0]) + float(token[1])) / 2
        try:
            return float(x)
        except:
            return None

    def feature_engineering(self):

        """Extracts the "integer" from text bhk or many forms from the 'size' column."""
        self.data['bhk'] = self.data['size'].apply(lambda x: int(x.split(' ')[0]) if isinstance(x, str) else None)
        del self.data['size']  # Remove the 'size' column

        # Convert 'total_sqft' ranges to average values if the column exists
        if 'total_sqft' in self.data.columns:
            self.data['total_sqft'] = self.data['total_sqft'].apply(self.convert_rangesqft_to_avg)  # Apply the function to each value

        # Drop rows where 'total_sqft' is less than 300 times the number of bedrooms (bhk)
        if 'total_sqft' in self.data.columns and 'bhk' in self.data.columns:
            self.data = self.data[~(self.data['total_sqft'] / self.data['bhk'] < 300)]


        if 'bhk' in self.data.columns and 'bath' in self.data.columns:
            self.data = self.data[self.data['bhk'] + 2 > self.data['bath']]

        """Creates new features and drops irrelevant ones."""
        # Create a new feature 'price_per_sqft' if 'total_sqft' and 'price' columns exist
        if 'total_sqft' in self.data.columns and 'price' in self.data.columns:
            self.data['price_per_sqft'] = self.data['price']*100000 / self.data['total_sqft']
        return self.data

    def remove_bhk_outliers(self):
        """Removes outliers based on price_per_sqft for bhk values within each location."""
        exclude_indices = []
        
        for location, location_df in self.data.groupby('location'):
            # Calculate statistics for each bhk in the location
            bhk_stats = {}
            for bhk, bhk_df in location_df.groupby('bhk'):
                bhk_stats[bhk] = {
                    'mean': np.mean(bhk_df['price_per_sqft']),
                    'std': np.std(bhk_df['price_per_sqft']),
                    'count': bhk_df.shape[0]
                }

            # Identify outliers for each bhk in the location
            for bhk, bhk_df in location_df.groupby('bhk'):
                stats = bhk_stats.get(bhk - 1)
                if stats and stats['count'] > 5:
                    exclude_indices.extend(
                        bhk_df[bhk_df['price_per_sqft'] < stats['mean']].index.values
                    )

        # Drop identified outliers
        self.data = self.data.drop(index=exclude_indices)
        print(f"Removed {len(exclude_indices)} outliers based on bhk and price_per_sqft.")
        return self.data
    
    def encode_features(self):
        """Encodes categorical features using pandas.get_dummies for one-hot encoding."""
        categorical_cols = self.data.select_dtypes(include=['object']).columns
        if categorical_cols.empty:
            print("No categorical features found for encoding.")

            return self.data

        # Create one-hot encoded columns for each categorical feature
        dummies = pd.get_dummies(self.data['location'], drop_first=True)
        dummies = dummies.astype(int)  # Convert to integers for consistency
        self.data = pd.concat([self.data, dummies], axis=1)  # Add dummies to the dataset

        # Drop original location column
        self.data = self.data.drop(columns=['location'])

        print(f"Categorical features encoded: {len(categorical_cols)}")
        print(f"New dataset shape after encoding: {self.data.shape}")

        return self.data

    def scale_features(self):
        """Scales numerical features using StandardScaler."""
        scaler = StandardScaler()
        numeric_cols = self.data.select_dtypes(include=['int64', 'float64']).columns
        self.data[numeric_cols] = scaler.fit_transform(self.data[numeric_cols])
        return self.data
    
    def handle_missing_values(self):
        """Handles remaining missing values after scaling."""
        # Drop rows with missing values
        self.data = self.data.dropna()
        return self.data

    def split_data(self, target_column, test_size=0.2, random_state=42):
        """Splits the dataset into training and testing sets.

        Args:
            target_column (str): The column to be used as the target variable.
            test_size (float): Proportion of the dataset to include in the test split.
            random_state (int): Random seed for reproducibility.

        Returns:
            tuple: X_train, X_test, y_train, y_test
        """
        if target_column not in self.data.columns:
            raise ValueError(f"Target column '{target_column}' not found in the dataset.")

        X = self.data.drop(columns=[target_column])
        y = self.data[target_column]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        return X_train, X_test, y_train, y_test

# Example Usage
if __name__ == "__main__":
    df = pd.read_csv("data/bengaluru_house_prices.csv")

    preprocessor = Preprocessing(data=df)
    # Data preprocessing steps
    preprocessor.clean_data()  # Clean the data
    preprocessor.feature_engineering()  # Perform feature engineering
    preprocessor.remove_bhk_outliers()  # Remove outliers
    preprocessor.encode_features()  # Encode features
    preprocessor.scale_features()  # Scale features
    preprocessor.handle_missing_values()  # Handle remaining missing values
    print(preprocessor.data.columns.tolist())
    print(preprocessor.data.shape)
    print("\nprocessing completed !!!")
    