import pandas as pd

def clean_data(data: pd.DataFrame) -> pd.DataFrame:

    data['TransactionMonth'] = pd.to_datetime(data['TransactionMonth'])
    data = data.applymap(lambda x: x.replace(' ', '') if isinstance(x, str) else x)

    return data

def validate_data_types(data):
    """
    Prints issues with data types for manual resolution.
    """
    print("Columns needing attention:\n")
    for col in data.columns:
        if data[col].dtype == "object" and data[col].nunique() < len(data):
            print(f"{col}: Likely categorical with {data[col].nunique()} levels.")
        elif "date" in col.lower() and not pd.api.types.is_datetime64_any_dtype(data[col]):
            print(f"{col}: Date column not in datetime format.")
        elif data[col].dtype not in ["float64", "int64", "datetime64[ns]", "object"]:
            print(f"{col}: Unexpected data type: {data[col].dtype}")
