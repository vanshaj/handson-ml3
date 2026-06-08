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

def create_pipeline(df: pd.DataFrame, columnName: str):
    # Get  index of the column
    index_of_positions_played = df.columns.get_loc(columnName)
    # Create custom transformer based on the index of that column, why because in np.ndarray column names will not work
    custTransf = customtransf.CustomMLBEncoding(index_of_positions_played)
    # Create a simple pipeline that will use SimpleImputer and then custom transformer
    category_steps_pip = Pipeline([
        ('cleanup', SimpleImputer(strategy="most_frequent")),
        ('encoding', custTransf),
    ])
    # Apply transformation to get the numpy ndarray
    np_transformed = category_steps_pip.fit_transform(df)
    # Drop the original column from the actual data frame so that we can append new columns
    df_without_column = df.drop(columns=[columnName])
    # Get all the existing columns
    existing_columns = df_without_column.columns.values
    # Create an array of existing columns with the new transformed columns that was being set as a variable in customTransf class object
    transformed_columns = np.append(existing_columns, custTransf.classes)
    # Create the dataframe from the transformed numpy ndarray
    df_transformed = pd.DataFrame(np_transformed, columns=transformed_columns, index=df_without_column.index)

    return df_transformed

def main():
    folder = os.getcwd()
    filename = "self/data/Fifa23Players.csv"
    new_filename = "self/data/Fifa23PlayersTransformed.csv"
    filepath = os.path.abspath(f"{folder}/{filename}")
    new_filepath = os.path.abspath(f"{folder}/{new_filename}")
    df = load(filepath)
    df = binn(df)
    train_set, test_set = split(df)
    y_train = train_set["Value"]
    transformed_train_set = create_pipeline(train_set)
    unload(transformed_train_set, new_filepath)



if __name__ == "__main__":
    main()
