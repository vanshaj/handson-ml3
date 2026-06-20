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
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="mean")),
        ('scaler', StandardScaler())
    ])
    num_attribs = df.select_dtypes(include=['int64','float64']).columns.tolist()
    # Get  index of the column
    cat_attribs = [columnName]
    # Create custom transformer based on the index of that column, why because in np.ndarray column names will not work
    custTransf = customtransf.CustomMLBEncoding()
    # Create a simple pipeline that will use SimpleImputer and then custom transformer
    category_steps_pip = Pipeline([
        ('cleanup', SimpleImputer(strategy="most_frequent")),
        ('encoding', custTransf),
    ])
    processor = ColumnTransformer(transformers=[
        ('num', num_pipeline, num_attribs),
        ('cat', category_steps_pip, cat_attribs)],
        remainder='passthrough'
    )

    # Apply transformation to get the numpy ndarray
    np_transformed = processor.fit_transform(df)
    transformed_columns = processor.get_feature_names_out()
    # Create the dataframe from the transformed numpy ndarray
    df_transformed = pd.DataFrame(np_transformed, columns=transformed_columns, index=df.index)

    return df_transformed

def main():
    folder = os.getcwd()
    filename = "self/chapter2/data/Fifa23Players.csv"
    new_filename = "self/chapter2/data/Fifa23PlayersTransformed.csv"
    filepath = os.path.abspath(f"{folder}/{filename}")
    new_filepath = os.path.abspath(f"{folder}/{new_filename}")
    df = load(filepath)
    df = binn(df)
    train_set, test_set = split(df)
    y_train = train_set["Value"]
    transformed_train_set = create_pipeline(train_set, "Positions Played")
    unload(transformed_train_set, new_filepath)



if __name__ == "__main__":
    main()
