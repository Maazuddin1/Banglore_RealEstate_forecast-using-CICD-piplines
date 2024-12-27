import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class EDA:
    def __init__(self, data_path):
        """Initialize with dataset path."""
        self.data_path = data_path
        self.data = None

    def load_data(self):
        """Loads the dataset from the provided path."""
        self.data = pd.read_csv(self.data_path)
        return self.data

    def basic_info(self):
        """Displays basic information about the dataset."""
        print("\nDataset Info:\n")
        print(self.data.info())
        print("\nShape:", self.data.shape)
        print("\nMissing Values:\n", self.data.isnull().sum())
        print("\nDuplicate Rows:", self.data.duplicated().sum())
        return self.data.describe()
    
    def missing_value_analysis(self):
        """Analyzes and visualizes missing values."""
        missing_data = self.data.isnull().sum()
        missing_data = missing_data[missing_data > 0].sort_values(ascending=False)

        if not missing_data.empty:
            plt.figure(figsize=(8, 6))
            sns.barplot(x=missing_data.index, y=missing_data.values, palette='viridis')
            plt.title('Missing Values Count')
            plt.xticks(rotation=45)
            plt.ylabel('Count')
            plt.show()
        
        return missing_data

    def visualize_distributions(self):
        """Visualizes distributions of numerical features."""
        numeric_cols = self.data.select_dtypes(include=['int64', 'float64']).columns
        self.data[numeric_cols].hist(bins=15, figsize=(10, 8), color='skyblue', edgecolor='black')
        plt.suptitle('Feature Distributions', fontsize=16)
        plt.show()

    def correlation_heatmap(self):
        """Plots a heatmap of feature correlations."""
        plt.figure(figsize=(10, 8))
        sns.heatmap(self.data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Feature Correlation Heatmap')
        plt.show()

    def detect_outliers(self, feature):
        """Detects and visualizes outliers for a given feature."""
        plt.figure(figsize=(8, 6))
        sns.boxplot(x=self.data[feature], color='lightblue')
        plt.title(f'Outliers in {feature}')
        plt.show()

    def feature_summary(self):
        """Provides a summary of categorical and numerical features."""
        categorical_cols = self.data.select_dtypes(include=['object']).columns
        numeric_cols = self.data.select_dtypes(include=['int64', 'float64']).columns

        print("\nCategorical Features:")
        for col in categorical_cols:
            print(f"{col}: {self.data[col].nunique()} unique values")
            print(self.data[col].value_counts().head(10))
            print("---")

        print("\nNumerical Features:")
        for col in numeric_cols:
            print(f"{col}: Mean={self.data[col].mean()}, Median={self.data[col].median()}, Std={self.data[col].std()}")
            print("---")

    def pairwise_scatterplots(self, features):
        """Plots scatterplots for selected features."""
        sns.pairplot(self.data[features], diag_kind='kde', plot_kws={'alpha': 0.5})
        plt.suptitle('Pairwise Scatterplots', fontsize=16)
        plt.show()

    def target_analysis(self, target_col):
        """Analyzes target variable distribution."""
        plt.figure(figsize=(8, 6))
        sns.histplot(self.data[target_col], kde=True, bins=30, color='blue')
        plt.title(f'Distribution of {target_col}')
        plt.xlabel(target_col)
        plt.ylabel('Frequency')
        plt.show()


if __name__ == "__main__":
    eda = EDA(data_path="data/bengaluru_house_prices.csv")
    data = eda.load_data()
    eda.basic_info()
    eda.missing_value_analysis()
    eda.visualize_distributions()
    eda.correlation_heatmap()
    eda.detect_outliers('price')
    eda.feature_summary()
    eda.pairwise_scatterplots(features=['price', 'total_sqft', 'bath', 'bhk'])
    eda.target_analysis(target_col='price')
    print("Missing values summary:")
    print(eda.missing_value_analysis())