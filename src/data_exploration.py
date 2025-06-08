import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load the training and test datasets."""
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    sample_submission = pd.read_csv('sample_submission.csv')
    return train_df, test_df, sample_submission

def examine_data_types(df):
    """Examine and print data types and sample values for each column."""
    print("\n=== DATA TYPES AND SAMPLE VALUES ===")
    for column in df.columns:
        print(f"\nColumn: {column}")
        print(f"Data type: {df[column].dtype}")
        print("Sample values:")
        print(df[column].head())
        print(f"Unique values count: {df[column].nunique()}")

def perform_eda(train_df, test_df):
    """Perform comprehensive exploratory data analysis."""
    print("=== DATASET OVERVIEW ===")
    print(f"Training set shape: {train_df.shape}")
    print(f"Test set shape: {test_df.shape}")
    
    # Examine data types
    print("\nTraining set data types:")
    examine_data_types(train_df)
    
    # Target variable analysis
    print("\n=== TARGET VARIABLE ANALYSIS ===")
    print(train_df['efficiency'].describe())
    
    # Plot target distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(data=train_df, x='efficiency', bins=50)
    plt.title('Distribution of Solar Panel Efficiency')
    plt.savefig('plots/efficiency_distribution.png')
    plt.close()
    
    # Missing values analysis
    print("\n=== MISSING VALUES ANALYSIS ===")
    missing_train = train_df.isnull().sum()
    missing_test = test_df.isnull().sum()
    print("\nTraining set missing values:")
    print(missing_train[missing_train > 0])
    print("\nTest set missing values:")
    print(missing_test[missing_test > 0])
    
    # Correlation analysis
    print("\n=== CORRELATION ANALYSIS ===")
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns
    correlation_matrix = train_df[numeric_cols].corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig('plots/correlation_matrix.png')
    plt.close()
    
    # Categorical variables analysis
    categorical_cols = train_df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        print("\n=== CATEGORICAL VARIABLES ANALYSIS ===")
        for col in categorical_cols:
            print(f"\nUnique values in {col}:")
            print(train_df[col].value_counts())
            
            # Plot categorical variable vs efficiency
            plt.figure(figsize=(10, 6))
            sns.boxplot(data=train_df, x=col, y='efficiency')
            plt.title(f'Efficiency Distribution by {col}')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f'plots/{col}_vs_efficiency.png')
            plt.close()

def main():
    # Create plots directory if it doesn't exist
    Path('plots').mkdir(exist_ok=True)
    
    # Load data
    train_df, test_df, sample_submission = load_data()
    
    # Perform EDA
    perform_eda(train_df, test_df)

if __name__ == "__main__":
    main() 