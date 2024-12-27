import pandas as pd

def clean_data(data: pd.DataFrame) -> pd.DataFrame:

    data['TransactionMonth'] = pd.to_datetime(data['TransactionMonth'])
    data = data.applymap(lambda x: x.replace(' ', '') if isinstance(x, str) else x)

    return data