import os

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

from load import load, unload
from binn import binn, split
from graphs import correlation
from cleanup import clean, one_hot_encoding
from sklearn.pipeline import Pipeline
import customtransf

def create_pipeline(df: pd.DataFrame):
    num_attribs = list(df.select_dtypes(include=['int64', 'float64']).columns)
    cat_attribs = list(df.select_dtypes(include=['object']).columns)

    index_of_value = num_attribs.index("Value")
    index_of_wage = num_attribs.index("Wage")

    index_of_positions_played = cat_attribs.index("Positions Played")

    numreical_steps = [
        ('cleanup', SimpleImputer(missing_values=np.nan, strategy="mean")),
        ('custom', customtransf.CustomTransf("Value", "Wage")),
        ('scaler', StandardScaler())
    ]
    category_steps_pip = Pipeline([
        # ('cleanup', SimpleImputer(strategy="most_frequent")),
        ('encoding', customtransf.CustomMLBEncoding(index_of_positions_played))
    ])
    df_transformed = category_steps_pip.fit_transform(df)
    """
    full_pipeline = ColumnTransformer([
        ("num", Pipeline(numreical_steps), num_attribs),
        ("cat", category_steps_pip, cat_attribs),
    ])
    df_transformed = full_pipeline.fit_transform(df)
    """

    return df_transformed

def main():
    folder = os.getcwd()
    filename = "data/Fifa23Players.csv"
    new_filename = "data/Fifa23PlayersTransformed.csv"
    filepath = os.path.abspath(f"{folder}/{filename}")
    new_filepath = os.path.abspath(f"{folder}/{new_filename}")
    df = load(filepath)
    df = binn(df)
    train_set, test_set = split(df)
    # histogram(train_set)
    # scatterPlot(train_set, "Potential", "Value")
    # correlation(train_set, ["Value", "Overall", "Potential", "Wage"])
    # clean(df)
    # df_new = one_hot_encoding(df, "Positions Played")
    # val_per_wage_t = customtransf.CustomTransf("Value", "Wage")
    # df_new = val_per_wage_t.fit_transform(df_new)
    y_train = train_set["Value"]
    transformed_train_set = create_pipeline(train_set)
    unload(transformed_train_set, new_filepath)



if __name__ == "__main__":
    main()
