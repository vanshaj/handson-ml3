import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

def binn(df: pd.DataFrame) -> pd.DataFrame:
    df["overall_cat"] = pd.cut(
        df["Overall"],
        bins=[0, 20, 40, 60, 80, 100],
        labels=[1, 2, 3, 4, 5]
    )
    return df
    
def split(df: pd.DataFrame):
    train_set, test_set = None, None
    sf_split = StratifiedShuffleSplit(n_splits=10, test_size=0.2,  random_state=33)
    for train_loc, test_loc in sf_split.split(df, df['overall_cat']):
        train_set = df.iloc[train_loc]
        test_set = df.iloc[test_loc]
    for set_ in (train_set, test_set):
        set_.drop("overall_cat", axis=1, inplace=True)
    return train_set, test_set