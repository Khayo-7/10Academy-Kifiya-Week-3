import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def preprocess_data(df):
    # Handle mixed-type columns
    for col in df.select_dtypes(include=["object", "category"]):
        df[col] = df[col].astype(str)
    
    # Separating categorical and numerical columns
    categorical_features = df.select_dtypes(include=["object", "category"]).columns
    numerical_features = df.select_dtypes(include=["number"]).columns

    # Pipelines
    categorical_pipeline = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore"))
        # ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    numerical_pipeline = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="mean")),
        ("scale", StandardScaler())
    ])

    # Combine pipelines
    preprocessor = ColumnTransformer(transformers=[
        ("num", numerical_pipeline, numerical_features),
        ("cat", categorical_pipeline, categorical_features)
    ])

    # Fit and transform
    processed_array = preprocessor.fit_transform(df)


    # Retrieve feature names
    numerical_columns = list(numerical_features)
    categorical_columns = list(preprocessor.named_transformers_['cat']['encode'].get_feature_names_out(categorical_features))
    column_names = numerical_columns + categorical_columns

    expected_columns = len(numerical_columns) + len(categorical_columns)
    actual_columns = processed_array.shape[1]
    print(f"Expected columns: {expected_columns}, Actual columns: {actual_columns}")


    # Converting the processed array back to a DataFrame
    # column_names = list(numerical_features) + list(preprocessor.named_transformers_['cat']['encode'].get_feature_names_out())
    clean_data = pd.DataFrame(processed_array, columns=column_names)

    return clean_data

# Save the cleaned dataset
def save_data(data, output_path):
    data.to_csv(output_path, index=False)

