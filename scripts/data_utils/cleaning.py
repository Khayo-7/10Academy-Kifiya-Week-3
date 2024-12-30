import pandas as pd

def clean_data(data: pd.DataFrame, date_column='Date') -> pd.DataFrame:
    """
    Perform initial cleaning for exploratory data analysis.
    
    - Converts date columns to datetime.
    - Handles missing values with basic replacements.
    - Drops essential rows with critical missing values.

    Args:
        data (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: Cleaned DataFrame for EDA.
    """
    data = data.copy()
    data = convert_date_column(data, date_column)

    # Remove leading/trailing whitespace in object columns
    data = data.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

    # Replace missing gender with "Notspecified"
    data['Gender'] = data['Gender'].replace(pd.NA, 'Notspecified')

    # Drop critical rows with missing values in essential columns
    essential_columns = ["Province", "PostalCode", "TotalClaims", "TotalPremium"]
    data = data.dropna(subset=essential_columns)

    return data

def validate_data_types(data: pd.DataFrame) -> None:
    """
    Validate and log potential data type issues for initial EDA.

    Args:
        data (pd.DataFrame): Input DataFrame.
    """
    print("Potential data type issues:")
    for col in data.columns:
        if data[col].dtype == "object" and data[col].nunique() < len(data):
            print(f"- {col}: Likely categorical with {data[col].nunique()} unique levels.")
        elif "date" in col.lower() and not pd.api.types.is_datetime64_any_dtype(data[col]):
            print(f"- {col}: Not in datetime format.")
        elif data[col].dtype not in ["float64", "int64", "datetime64[ns]", "object"]:
            print(f"- {col}: Unexpected data type: {data[col].dtype}")

def missing_values_summary(data, threshold=50):
    """
    Summary of missing values in the dataset, highlighting critical missing values.

    Args:
        data (pd.DataFrame): Input dataset.
        threshold (int): Threshold for critical missing values (default=50).

    Returns:
        pd.DataFrame: Columns with percentage of missing values, highlighting critical missing values.
    """
    missing_summary = data.isnull().mean().sort_values(ascending=False) * 100
    missing_summary = missing_summary[missing_summary > 0].rename("Missing (%)")
    
    # Highlight critical missing values
    critical_missing = missing_summary[missing_summary >= threshold]
    non_critical_missing = missing_summary[missing_summary < threshold]
    
    # Return both critical and non-critical missing values
    return critical_missing, non_critical_missing

def convert_date_column(data: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """
    Safely convert a specified column to datetime.

    Args:
        data (pd.DataFrame): Input DataFrame.
        column_name (str): Column name to convert.

    Returns:
        pd.DataFrame: DataFrame with updated column.
    """

    data = data.copy()
    try:
        data[column_name] = pd.to_datetime(data[column_name], errors='coerce')
    except ValueError as e:
        print(f"Error converting column '{column_name}' to datetime: {e}")
    return data