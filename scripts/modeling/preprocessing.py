import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
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

    return data

def encode_categorical_data(data):
    """
    Encode categorical variables using one-hot encoding.
    
    Args:
        data (pd.DataFrame): Input dataset with categorical features.
    
    Returns:
        pd.DataFrame: Encoded dataset.
    """

    def split_dataframe(data):
        """
        Splits the dataset into categorical and non-categorical columns.
        """
        # Identify categorical columns
        # categorical_cols = ['Gender', 'Country', 'Province', 'VehicleType']
        categorical_cols = data.select_dtypes(include=["object", "bool", "category"]).columns
        non_categorical_cols = data.drop(columns=categorical_cols).columns
        return data[categorical_cols], data[non_categorical_cols]

    def recombine_data(non_categorical_data, encoded_data):
        """
        Recombines non-categorical data with encoded categorical data.
        """
        return pd.concat([non_categorical_data, encoded_data], axis=1)

    def encode_categorical(categorical_data):
        """
        Encodes categorical data using one-hot encoding.
        """
        
        # Apply one-hot encoding
        encoder = OneHotEncoder(drop="first", sparse_output=False)
        column_transformer = ColumnTransformer(
            transformers=[("onehot", encoder, categorical_data.columns)],
            remainder="passthrough"
        )

        # Transform data and create a new DataFrame
        transformed_data = column_transformer.fit_transform(categorical_data)
        encoded_feature_names = column_transformer.get_feature_names_out()
        encoded_categorical_data = pd.DataFrame(transformed_data, columns=encoded_feature_names)
        encoded_categorical_data.columns = [col.replace("onehot__", "").replace("remainder__", "") for col in encoded_categorical_data.columns]

        # Apply OneHotEncoder directly
        # encoder = OneHotEncoder(drop="first", sparse_output=False)
        # transformed_data = encoder.fit_transform(categorical_data)
        # encoded_feature_names = encoder.get_feature_names_out(categorical_data.columns)
        # return pd.DataFrame(transformed_data, columns=encoded_feature_names, index=categorical_data.index)

        # One-hot encode categorical features using pandas
        # encoded_categorical_data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

        return encoded_categorical_data
    
    
    categorical_data, non_categorical_data = split_dataframe(data)
    
    if not categorical_data.empty:
        encoded_categorical_data = encode_categorical(categorical_data)
    else:
        encoded_categorical_data = pd.DataFrame(index=data.index)  # Handle case where no categorical columns exist
    
    encoded_data = recombine_data(non_categorical_data, encoded_categorical_data)

    return encoded_data


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
    X = data.drop(columns=[target_variable]) # Features
    y = data[target_variable] # Target
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
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
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
    data = encode_categorical_data(data)
    
    return data