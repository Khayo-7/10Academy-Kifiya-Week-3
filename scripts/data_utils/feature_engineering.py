import numpy as np
import pandas as pd
from datetime import datetime

def generate_risk_metrics(data):
    """
    Add derived risk metrics to the dataset, handling invalid or zero TotalPremium values.

    This function calculates two risk metrics: ProfitMargin and Risk. It handles errors that may occur during the calculation process, such as invalid or zero TotalPremium values. The ProfitMargin is calculated as the difference between TotalPremium and TotalClaims. The Risk is calculated as the ratio of TotalClaims to TotalPremium, with error handling for division by zero and invalid TotalPremium values.

    Args:
        data (pd.DataFrame): Input dataset containing 'TotalPremium' and 'TotalClaims' columns.

    Returns:
        pd.DataFrame: The input dataset with additional 'ProfitMargin' and 'Risk' columns.
    """
    
    data = data.copy()
    try:
        # Calculate ProfitMargin for invalid TotalPremium values
        data["ProfitMargin"] = data["TotalPremium"].astype(float) - data["TotalClaims"].astype(float)
    except ValueError as e:
        print(f"Error converting TotalPremium or TotalClaims to float: {e}")
    
    try:
        # Calculate Risk for invalid or zero TotalPremium values
        data["Risk"] = data["TotalClaims"].astype(float) / data["TotalPremium"].astype(float).replace(0, pd.NA)
    except ZeroDivisionError:
        print("Error: Division by zero in Risk calculation. Replacing with NA.")
    except ValueError as e:
        print(f"Error converting TotalClaims or TotalPremium to float for Risk calculation: {e}")
    
    # Ensure 'Risk' column is of float type and drop rows with invalid Risk
    data["Risk"] = pd.to_numeric(data["Risk"], errors='coerce')
    data = data.dropna(subset=["Risk"])
    
    return data

def generate_features(data):
    """
    Generate additional features from existing dataset columns.

    This function generates three new features: VehicleAge, Region, and VehicleCondition. VehicleAge is calculated as the difference between the current year and the vehicle's registration year. Region is created by combining 'Province' and 'PostalCode'. VehicleCondition is categorized as 'New' or 'Old' based on the vehicle's registration year.

    Args:
        data (pd.DataFrame): Input dataset containing 'RegistrationYear', 'Province', and 'PostalCode' columns.

    Returns:
        pd.DataFrame: The input dataset with additional 'VehicleAge', 'Region', and 'VehicleCondition' columns.
    """

    def categorize_vehicle_condition(registration_year):
        """
        Categorize vehicle condition based on registration year.
        """
        cutoff_year = datetime.now().year - 5
        return "New" if registration_year > cutoff_year else "Old"

    data = data.copy()
    try:
        # Create a region column from 'Province' and 'PostalCode'
        data['Region'] = data['Province'] + "_" + data['PostalCode'].astype(str)

        # Create age of the vehicle based on RegistrationYear
        data['VehicleAge'] = datetime.now().year - data['RegistrationYear'].astype(int)   
        # Categorize vehicles as New or Old
        data["VehicleCondition"] = data["RegistrationYear"].apply(categorize_vehicle_condition)

        data = data.drop(columns=["Province"])
        data = data.drop(columns=["PostalCode"])
        data = data.drop(columns=["RegistrationYear"])

    except ValueError as e:
        print(f"Error generating features: {e}")

    return data

def generate_time_series_features(data, date_column='Date'):
    """
    Generates time series features for the dataset.

    This function enriches the dataset by adding lag and rolling features based on the specified date column.

    Args:
        data (pd.DataFrame): The input dataset.
        date_column (str, optional): The name of the column containing dates in the dataset. Defaults to 'Date'.

    Returns:
        pd.DataFrame: The input dataset augmented with time series features.
    """
    data = data.copy()
    
    # Ensure the date column is of datetime type
    data[date_column] = pd.to_datetime(data[date_column])
    data = data.sort_values(by=date_column)
    
    # Generate temporal features
    data["Year"] = data[date_column].dt.year
    data["Month"] = data[date_column].dt.month
    data["Day"] = data[date_column].dt.day
    data["DayOfWeek"] = data[date_column].dt.dayofweek
    data["Quarter"] = data[date_column].dt.quarter
    data['IsWeekend'] = data[date_column].dt.dayofweek.isin([5, 6]) #.astype(int)
    data = data.drop(columns=[date_column])

    return data

def apply_cyclical_encoding(data):
    """
    Applies cyclical encoding to the 'Month' and 'DayOfWeek' columns.

    This function transforms the 'Month' and 'DayOfWeek' columns into their sine and cosine values to capture cyclical patterns.

    Args:
        data (pd.DataFrame): The input dataset.

    Returns:
        pd.DataFrame: The input dataset with cyclical encoding applied to 'Month' and 'DayOfWeek'.
    """
    # Cyclical encoding for Month and DayOfWeek
    data["Month_sin"] = np.sin(2 * np.pi * data["Month"] / 12)
    data["Month_cos"] = np.cos(2 * np.pi * data["Month"] / 12)
    data["DayOfWeek_sin"] = np.sin(2 * np.pi * data["DayOfWeek"] / 7)
    data["DayOfWeek_cos"] = np.cos(2 * np.pi * data["DayOfWeek"] / 7)

    return data

def generate_lag_and_rolling_features(data, target_column):
    """
    Generates lag and rolling features for the specified target column.

    This function enriches the dataset by adding lag and rolling features based on the specified target column.

    Args:
        data (pd.DataFrame): The input dataset.
        target_column (str): The name of the column for which lag and rolling features are generated.

    Returns:
        pd.DataFrame: The input dataset augmented with lag and rolling features.
    """
    # Lag features for 1, 3, 6-month lookbacks
    for lag in [1, 3, 6]:
        data[f"{target_column}_lag_{lag}"] = data[target_column].shift(lag)
    
    # Rolling average 3-month window
    data[f"{target_column}_rolling_3"] = data[target_column].rolling(window=3).mean()

    # Drop rows with NaN (caused by lag/rolling computations)
    data = data.dropna(subset=data.columns.difference(['Year', 'Month', 'Day', 'DayOfWeek', 'Quarter', 'Month_sin', 'Month_cos', 'DayOfWeek_sin', 'DayOfWeek_cos']))

    return data

