import argparse
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from scripts.data_utils.loaders import load_data
from scripts.data_utils.cleaning import clean_data
from scripts.data_utils.feature_engineering import generate_time_series_features,  generate_risk_metrics, generate_features

def handle_missing_data(data: pd.DataFrame, missing_threshold=0.5) -> pd.DataFrame:
    """
    Handle missing values by dropping high-missing-value columns, imputing numeric/categorical features.
    
    Args:
        data (pd.DataFrame): Input dataset.
        missing_threshold (float): Threshold for dropping columns with too many missing values.

    Returns:
        pd.DataFrame: Dataset prepared for statistical modeling.
    """
    print("Starting handle_missing_data process...")
    data = data.copy()

    # Drop columns with missing ratios above the threshold
    missing_ratios = data.isnull().mean()
    to_drop = missing_ratios[missing_ratios > missing_threshold].index
    data = data.drop(columns=to_drop)

    # Separate columns by data types
    numeric_cols = data.select_dtypes(include=["number"]).columns
    categorical_cols = data.select_dtypes(include=["object", "bool", "category"]).columns
    datetime_cols = data.select_dtypes(include=["datetime"]).columns

    # Impute numeric columns (median)
    imputer_numeric = SimpleImputer(strategy="median")
    data[numeric_cols] = imputer_numeric.fit_transform(data[numeric_cols])

    # Impute categorical columns (mode)
    imputer_categorical = SimpleImputer(strategy="most_frequent")
    data[categorical_cols] = imputer_categorical.fit_transform(data[categorical_cols])

    # Impute datetime columns (mode)
    for col in datetime_cols:
        if data[col].isnull().any():
            data[col] = data[col].fillna(data[col].mode()[0])
            # data.fillna({col: data[col].mode()[0]}, inplace=True)

    print("Finished handle_missing_data process.")
    return data

def aggregate_rare_categories(data, threshold=0.05):
    """
    Aggregates rare categories in each categorical column into 'Other'.
    
    Parameters:
    - data: DataFrame with categorical columns.
    - threshold: Proportion threshold below which a category is considered rare.
    
    Returns:
    - A DataFrame with rare categories grouped into 'Other.'
    """
    print("Starting aggregate_rare_categories process...")
    for col in data.columns:
        freq = data[col].value_counts(normalize=True)
        rare_categories = freq[freq < threshold].index
        data[col] = data[col].replace(rare_categories, 'Other')
    print("Finished aggregate_rare_categories process.")
    return data

def handle_high_cardinality(data, max_unique=20):
    """
    Drops or aggregates high-cardinality categorical columns.
    
    Parameters:
    - data: The categorical data (DataFrame).
    - max_unique: Maximum unique values allowed before considering a column as high cardinality.
    
    Returns:
    - A DataFrame with high-cardinality columns dropped or aggregated.
    """
    print("Starting handle_high_cardinality process...")
    high_cardinality_cols = [col for col in data.columns if data[col].nunique() > max_unique]
    print(f"High-cardinality columns dropped: {high_cardinality_cols}")
    print("Finished handle_high_cardinality process.")
    return data.drop(columns=high_cardinality_cols)


def split_dataframe(data):
    """
    Splits the dataset into categorical and non-categorical columns.
    """
    print("Starting split_dataframe process...")
    # Identify categorical columns
    # categorical_cols = ['Gender', 'Country', 'Province', 'VehicleType']
    categorical_data = data.select_dtypes(include=["object", "bool", "category"])
    non_categorical_data = data.select_dtypes(exclude=["object", "bool", "category"])

    print("Finished split_dataframe process.")
    return categorical_data, non_categorical_data

def recombine_data(categorical_data, non_categorical_data):
    """
    Recombines non-categorical data with encoded categorical data.
    """
    print("Starting recombine_data process...")
    recombine_data = pd.concat([categorical_data, non_categorical_data], axis=1)
    print("Finished recombine_data process.")
    return recombine_data

def one_hot_encode(data):
    """
    Encodes categorical data using one-hot encoding.

    Parameters:
    - data: The categorical data (DataFrame).

    Returns:
    - A DataFrame of encoded features.
    """
    print("Starting one_hot_encode process...")

    # # Apply one-hot encoding
    encoder = OneHotEncoder(drop="first", sparse_output=False)
    column_transformer = ColumnTransformer(
        transformers=[("onehot", encoder, data.columns)],
        remainder="passthrough"
    )

    # Transform data and create a new DataFrame
    transformed_data = column_transformer.fit_transform(data)
    feature_names = column_transformer.get_feature_names_out()   

    # Apply OneHotEncoder directly
    # transformed_data = encoder.fit_transform(data)
    # feature_names = encoder.get_feature_names_out(data.columns)

    feature_names = [col.replace("onehot__", "").replace("remainder__", "") for col in feature_names]
    encoded_data = pd.DataFrame(transformed_data, columns=feature_names, index=data.index)
    # encoded_data.columns = [col.replace("onehot__", "").replace("remainder__", "") for col in encoded_data.columns]

    # One-hot encode categorical features using pandas
    # encoded_data = pd.get_dummies(data, columns=data.columns, drop_first=True)

    print("Finished one_hot_encode process.")
    return encoded_data

def label_encode(column):
    """
    Encodes a single categorical column using label encoding.

    Parameters:
    - column: Series (categorical column).

    Returns:
    - encoded_column: Label-encoded column as a Series.
    - encoder: The fitted LabelEncoder instance.
    """
    print("Starting label_encode process...")
    encoder = LabelEncoder()
    encoded_column = pd.Series(encoder.fit_transform(column), index=column.index, name=column.name)
    print("Finished label_encode process.")
    return encoded_column, encoder

def encode_categorical(data, cardinality_threshold=15):
    """
    Encodes categorical columns dynamically based on cardinality.

    Parameters:
    - data: DataFrame containing categorical columns.
    - cardinality_threshold: Threshold to decide one-hot vs. label encoding.

    Returns:
    - encoded_data: DataFrame with all categorical columns encoded.
    - encoders: Dictionary of encoders used for label encoding.
    """
    print("Starting encode_categorical process...")
    encoded_data = pd.DataFrame(index=data.index)
    encoders = {}

    for col in data.columns:
        unique_count = data[col].nunique()
        if unique_count <= cardinality_threshold:
            # One-hot encoding for low-cardinality columns
            one_hot_encoded = one_hot_encode(data[[col]])
            encoded_data = pd.concat([encoded_data, one_hot_encoded], axis=1)
        else:
            # Label encoding for high-cardinality columns or two-cardinal columns
            encoded_column, encoder = label_encode(data[col])
            encoded_data[col] = encoded_column
            encoders[col] = encoder

    print("Finished encode_categorical process.")
    return encoded_data, encoders

def preprocess_categorical_data(data, cardinality_threshold=15, rare_threshold=0.05, max_unique=20):
    """
    Complete preprocessing pipeline for categorical data, including handling rare categories, 
    high cardinality, and encoding.
    
    Parameters:
    - data: The categorical data (DataFrame).
    - cardinality_threshold: Threshold to decide one-hot vs. label encoding.
    - rare_threshold: Proportion threshold for rare category aggregation.
    - max_unique: Maximum unique values for high-cardinality filtering.

    Returns:
    - encoded_data: Processed DataFrame with encoded categorical data.
    - encoders: Dictionary containing label encoders for label-encoded columns.
    """
    print("Starting preprocess_categorical_data process...")
    # Handle rare categories
    data = aggregate_rare_categories(data, threshold=rare_threshold)
    # data = handle_high_cardinality(data, max_unique=max_unique)

    # Encode categorical data
    encoded_data, encoders = encode_categorical(data, cardinality_threshold=cardinality_threshold)

    print("Finished preprocess_categorical_data process.")
    return encoded_data, encoders

def encode_data(data, cardinality_threshold=15, rare_threshold=0.05):
    """
    Splits, processes, and recombines categorical and non-categorical data.

    Parameters:
    - data: Full DataFrame with mixed types.
    - cardinality_threshold: Threshold for encoding decisions.
    - rare_threshold: Proportion threshold for rare category handling.

    Returns:
    - fully_processed_data: Final DataFrame with both encoded categorical and unmodified non-categorical data.
    - encoders: Dictionary of label encoders for label-encoded columns.
    """
    print("Starting encode_data process...")
    categorical_data, non_categorical_data = split_dataframe(data)
    
    if not categorical_data.empty:
        encoded_categorical_data, encoders = preprocess_categorical_data(
            categorical_data, cardinality_threshold=cardinality_threshold, rare_threshold=rare_threshold
        )
    else:
        encoded_categorical_data = pd.DataFrame(index=data.index)  # Handle case where no categorical columns exist
        encoders = {}
    
    # Recombine processed categorical and non-categorical data
    encoded_data = recombine_data(encoded_categorical_data, non_categorical_data)

    print("Finished encode_data process.")
    return encoded_data, encoders

def split_data(data, target_variable, test_size=0.2, random_state=42):
    """
    Split the data into training and testing sets.

    Args:
        data (pd.DataFrame): Input dataset.
        target_variable (str): Target variable name.
        test_size (float): Test dataset proportion (default=0.2).
        random_state (int): Random seed for reproducibility (default=42).
    
    Returns:
        tuple: X_train, X_test, y_train, y_test.
    """
    print("Starting split_data process...")

    X = data.drop(columns=[target_variable]) # Features
    y = data[target_variable] # Target

    print("Finished split_data process.")
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def time_series_split_data(data, date_column, target_variable, split_date):

    # Ensure date column is datetime and data is sorted
    data[date_column] = pd.to_datetime(data[date_column])
    data = data.sort_values(by=date_column)

    # Split data chronologically
    train = data[data[date_column] < split_date]
    test = data[data[date_column] >= split_date]
    
    # Separate features and target
    X_train = train.drop(columns=[target_variable])
    y_train = train[target_variable]
    X_test = test.drop(columns=[target_variable])
    y_test = test[target_variable]

    return X_train, X_test, y_train, y_test

def scale_features(X_train, X_test):
    """
    Scale the features of the training and testing datasets using StandardScaler.

    Args:
        X_train (array-like): Training features.
        X_test (array-like): Testing features.

    Returns:
        tuple: X_train_scaled, X_test_scaled.
    """
    print("Starting scale_features process...")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Finished scale_features process.")
    return X_train_scaled, X_test_scaled

def prepare_data(data, drop_columns=None):
    """
    Prepare data for modeling by handling specific missing values and dropping irrelevant columns.

    Args:
        data (pd.DataFrame): Input dataset.
        drop_columns (list): List of columns to drop (default=None).
    
    Returns:
        pd.DataFrame: Cleaned dataset ready for preprocessing.
    """
    print("Starting prepare_data process...")
    data = data.copy()

    # Drop specified columns
    if drop_columns:
        data = data.drop(columns=drop_columns, errors="ignore")
    
    # Impute domain-specific missing values
    if "mmcode" in data:
        data["mmcode"] = data["mmcode"].fillna(data["mmcode"].median())
    for col in ["Bank", "AccountType", "Citizenship"]:
        if col in data:
            data[col] = data[col].fillna(data[col].mode()[0])

    # Drop rows where critical targets are missing
    critical_targets = ["TotalClaims", "TotalPremium"]
    data = data.dropna(subset=[col for col in critical_targets if col in data])

    print("Finished prepare_data process.")
    return data

def preprocess_data(data):
    """
    Full preprocessing pipeline: handle missing data, encode categorical features, and prepare data for modeling.

    Args:
        data (pd.DataFrame): The input dataset.
        target_col (str): The target column.
    
    Returns:
        tuple: Preprocessed features, target, X_train, X_test, y_train, y_test.
    """
    print("Starting preprocess_data process...")
    data = data.copy()

    date_column='TransactionMonth'
    drop_columns=None # ["PolicyID"]

    # Clean data
    data = clean_data(data, date_column=date_column)

    # Handle missing data
    data = handle_missing_data(data)

    # Add Risk and Profit Margin and other derived columns in the dataframe
    data = generate_risk_metrics(data)
    data = generate_features(data)

    # Generate Time-Series features
    data = generate_time_series_features(data, date_column=date_column)
    
    # Prepare data for modeling
    data = prepare_data(data, drop_columns=drop_columns)

    # Encode categorical data
    data, _ = encode_data(data)
    
    print("Finished preprocess_data process.")
    return data


if __name__ == "__main__":
    print("Starting preprocessing script...")
    parser = argparse.ArgumentParser(description='Preprocess data')
    parser.add_argument('--input_file', type=str, help='Input CSV file')
    args = parser.parse_args()
    preprocess_data(load_data(args.input_file))
    print("Finished preprocessing script.")
