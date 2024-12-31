import os
import pandas as pd

def load_data(file_path: str, sep=',') -> pd.DataFrame:
    """Loads the dataset from a CSV file."""

    try:
        dataframe = pd.read_csv(file_path, sep=sep)
        # dataframe = pd.read_csv(file_path, sep='|')
        # dataframe = pd.read_csv(file_path, sep='\t')
        print("Data loaded successfully.")
        return dataframe
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return None

def summarize_data(data: pd.DataFrame):
    """Prints summary statistics and info about the dataset."""

    try:
        print("\n--- Data Summary ---")
        print(data.describe())
        print("\n--- Data Info ---")
        print(data.info())
        print("\n--- Total columns with Missing Values ---")
        print((data.isnull().sum() > 0).sum())
        print("\n--- Missing Values ---")
        print(data.isnull().sum())
    except Exception as e:
        print(f"Error summarizing data: {e}")

def save_data(dataframe, output_path):
    dataframe.to_csv(output_path, index=False)