import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer


def clean(df: pd.DataFrame):
    overall_df = df[["Overall"]]
    impute_fit = SimpleImputer(missing_values=np.nan, strategy="mean")
    impute_fit.fit(overall_df)
    impute_fit.transform(overall_df)
    print(overall_df)


def one_hot_encoding(df: pd.DataFrame, column: str) -> pd.DataFrame:
    # encoder = OneHotEncoder()
    mlb = MultiLabelBinarizer()
    positions = df[column].str.split(",")
    # encoded_pos = encoder.fit_transform(df[[column]])
    position_df = pd.DataFrame(
        mlb.fit_transform(positions),
        columns=mlb.classes_,
        index=df.index
    )
    print(position_df.head())
    # Remove column
    df.drop(columns=[column], inplace=True)
    # Merge the new transformed data , axis = 1 means column wise
    df_new = pd.concat([df, position_df], axis=1)
    return df_new
