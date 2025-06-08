import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def clean_numeric_columns(df):
    """
    Clean numeric columns by converting to float and handling invalid values.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Cleaned dataframe with numeric columns
    """
    df = df.copy()
    
    # Get numeric columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    # Print initial data types
    print("\nInitial data types:")
    print(df.dtypes)
    
    # Convert to numeric, replacing invalid values with NaN
    for col in numeric_columns:
        print(f"\nProcessing column: {col}")
        print(f"Before conversion - Sample values: {df[col].head()}")
        
        # Convert to numeric
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Fill NaN values with median
        median_value = df[col].median()
        df[col] = df[col].fillna(median_value)
        
        print(f"After conversion - Sample values: {df[col].head()}")
        print(f"Data type: {df[col].dtype}")
    
    return df

def clean_string_numeric_columns(df):
    """
    Clean columns that should be numeric but are currently strings.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Cleaned dataframe with numeric columns
    """
    df = df.copy()
    
    # Columns that should be numeric but might contain string values
    string_numeric_columns = ['humidity', 'wind_speed', 'pressure']
    
    for col in string_numeric_columns:
        if col in df.columns:
            print(f"\nProcessing string column: {col}")
            print(f"Before cleaning - Sample values: {df[col].head()}")
            
            # Replace 'error' and other non-numeric values with NaN
            df[col] = df[col].replace(['error', 'ERROR', 'Error', 'N/A', 'n/a', 'NA', 'na'], np.nan)
            
            # Convert to numeric
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Fill NaN values with median
            median_value = df[col].median()
            df[col] = df[col].fillna(median_value)
            
            print(f"After cleaning - Sample values: {df[col].head()}")
            print(f"Data type: {df[col].dtype}")
    
    return df

def create_advanced_features(df):
    """
    Create domain-specific features for solar panel efficiency prediction.
    
    Args:
        df (pd.DataFrame): Input dataframe with solar panel sensor data
        
    Returns:
        pd.DataFrame: Enhanced dataframe with new features
    """
    # First clean the string numeric columns
    df = clean_string_numeric_columns(df)
    
    # Then clean the numeric columns
    df = clean_numeric_columns(df)
    
    # Ensure all numeric columns are float
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        df[col] = df[col].astype(np.float64)
    
    # Create features with explicit type conversion
    try:
        # Power-related features
        df['power_output'] = df['voltage'] * df['current']
        df['power_per_area'] = df['power_output'] / (df['irradiance'] + 1e-6)
        
        # Temperature efficiency relationships
        df['temp_efficiency_ratio'] = df['module_temperature'] / (df['temperature'] + 273.15)
        df['temp_difference'] = df['module_temperature'] - df['temperature']
        
        # Environmental impact features
        df['irradiance_temp_interaction'] = df['irradiance'] * df['module_temperature']
        df['humidity_temp_interaction'] = df['humidity'] * df['temperature']
        df['cloud_irradiance_ratio'] = df['cloud_coverage'] / (df['irradiance'] + 1e-6)
        
        # Degradation indicators
        df['age_maintenance_ratio'] = df['panel_age'] / (df['maintenance_count'] + 1)
        df['soiling_age_interaction'] = df['soiling_ratio'] * df['panel_age']
        
        # Cooling effect
        df['wind_cooling_effect'] = df['wind_speed'] / (df['module_temperature'] + 1e-6)
        
        # Efficiency indicators
        df['theoretical_efficiency'] = df['irradiance'] * (1 - df['soiling_ratio']) * (1 - df['cloud_coverage']/100)
        
        # Polynomial features for non-linear relationships
        df['temp_squared'] = df['temperature'] ** 2
        df['irradiance_squared'] = df['irradiance'] ** 2
        df['humidity_squared'] = df['humidity'] ** 2
        
    except Exception as e:
        print(f"\nError creating features: {str(e)}")
        print("\nDataFrame info:")
        print(df.info())
        print("\nSample data:")
        print(df.head())
        raise
    
    return df

def encode_categorical_features(df, categorical_columns):
    """
    Encode categorical features using Label Encoding.
    
    Args:
        df (pd.DataFrame): Input dataframe
        categorical_columns (list): List of categorical column names
        
    Returns:
        pd.DataFrame: DataFrame with encoded categorical features
    """
    df = df.copy()
    label_encoders = {}
    
    for col in categorical_columns:
        if col in df.columns:
            label_encoders[col] = LabelEncoder()
            df[col] = label_encoders[col].fit_transform(df[col].astype(str))
    
    return df, label_encoders

def handle_missing_values(df, numeric_columns, categorical_columns):
    """
    Handle missing values in the dataset.
    
    Args:
        df (pd.DataFrame): Input dataframe
        numeric_columns (list): List of numeric column names
        categorical_columns (list): List of categorical column names
        
    Returns:
        pd.DataFrame: DataFrame with handled missing values
    """
    df = df.copy()
    
    # Handle numeric missing values with median
    for col in numeric_columns:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    
    # Handle categorical missing values with mode
    for col in categorical_columns:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])
    
    return df

def handle_outliers(df, columns, method='iqr'):
    """
    Handle outliers in numeric columns.
    
    Args:
        df (pd.DataFrame): Input dataframe
        columns (list): List of column names to handle outliers for
        method (str): Method to use for outlier detection ('iqr' or 'zscore')
        
    Returns:
        pd.DataFrame: DataFrame with handled outliers
    """
    df = df.copy()
    
    for col in columns:
        if col in df.columns:
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df[col] = df[col].clip(lower_bound, upper_bound)
            elif method == 'zscore':
                mean = df[col].mean()
                std = df[col].std()
                df[col] = df[col].clip(mean - 3*std, mean + 3*std)
    
    return df

def preprocess_data(train_df, test_df, categorical_columns):
    """
    Preprocess the training and test datasets.
    
    Args:
        train_df (pd.DataFrame): Training dataframe
        test_df (pd.DataFrame): Test dataframe
        categorical_columns (list): List of categorical column names
        
    Returns:
        tuple: Processed training and test dataframes
    """
    # Create advanced features
    train_processed = create_advanced_features(train_df)
    test_processed = create_advanced_features(test_df)
    
    # Handle missing values
    numeric_columns = train_processed.select_dtypes(include=[np.number]).columns
    train_processed = handle_missing_values(train_processed, numeric_columns, categorical_columns)
    test_processed = handle_missing_values(test_processed, numeric_columns, categorical_columns)
    
    # Handle outliers in training data
    train_processed = handle_outliers(train_processed, numeric_columns)
    
    # Encode categorical features
    train_processed, label_encoders = encode_categorical_features(train_processed, categorical_columns)
    test_processed, _ = encode_categorical_features(test_processed, categorical_columns)
    
    return train_processed, test_processed 