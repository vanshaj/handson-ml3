import pandas as pd


def load(file_name) -> pd.DataFrame:
    return pd.read_csv(file_name)

def unload(df: pd.DataFrame, file_name) -> None:
    df.to_csv(file_name)
