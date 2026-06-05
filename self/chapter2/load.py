import pandas as pd


def load(file_name) -> pd.DataFrame:
    return pd.read_csv(file_name)
