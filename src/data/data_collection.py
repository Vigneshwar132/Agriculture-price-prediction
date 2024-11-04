import pandas as pd
import os

def load_data(file_path):
    """
    Load raw data from a given file path
    """
    return pd.read_csv(file_path)

def collect_all_data():
    """
    Collect data for all commodities
    """
    data = {}
    raw_data_dir = 'data/raw'
    file_path = os.path.join(raw_data_dir, 'Bhindi.csv')
    print(f"Loading data for Bhindi from {file_path}")
    data['Bhindi'] = load_data(file_path)
    return data
