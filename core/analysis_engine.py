import pandas as pd

def statistical_summary(df: pd.DataFrame):
    summary = df.describe(include="all").transpose()
    return summary
