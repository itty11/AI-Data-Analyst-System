import pandas as pd
from utils.helpers import get_numeric_columns, get_categorical_columns

def analyze_schema(df: pd.DataFrame):
    schema = {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "numeric_columns": get_numeric_columns(df),
        "categorical_columns": get_categorical_columns(df),
        "missing_values": df.isnull().sum().to_dict()
    }
    return schema
