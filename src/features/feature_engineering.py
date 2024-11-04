import pandas as pd
import numpy as np

def create_time_features(df):
    """
    Create time-based features from date column
    """
    df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    return df

def create_lag_features(df, lag_periods=[1, 7, 30]):
    """
    Create lagged price features
    """
    price_col = 'Modal Price (Rs./Quintal)'
    for lag in lag_periods:
        df[f'price_lag_{lag}'] = df[price_col].shift(lag)
    return df

def create_rolling_features(df, windows=[7, 30, 90]):
    """
    Create rolling average and standard deviation features
    """
    price_col = 'Modal Price (Rs./Quintal)'
    for window in windows:
        df[f'price_rolling_mean_{window}'] = df[price_col].rolling(window=window).mean()
        df[f'price_rolling_std_{window}'] = df[price_col].rolling(window=window).std()
    return df

def engineer_features(data):
    """
    Apply feature engineering to all commodities
    """
    engineered_data = {}
    for commodity, df in data.items():
        print(f"\nEngineering features for {commodity}...")
        print("Input columns:", df.columns.tolist())
        
        if 'date' in df.columns:
            df = create_time_features(df)
            df = create_lag_features(df)
            df = create_rolling_features(df)
            df = df.dropna()  # Remove rows with NaN values
        else:
            print(f"Warning: 'date' column not found for {commodity}. Skipping time-based feature engineering.")
        
        engineered_data[commodity] = df
    return engineered_data