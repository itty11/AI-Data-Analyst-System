import pandas as pd
from utils.helpers import clean_column_names

def load_data(file):
    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        raise ValueError("Only CSV files are supported")

    df = clean_column_names(df)
    return df
