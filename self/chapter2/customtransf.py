import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MultiLabelBinarizer


class CustomTransf(BaseEstimator, TransformerMixin):
    def __init__(self, columnA, columnB):
        self.columnA = columnA
        self.columnB = columnB

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["derived"] = X[self.columnA] / X[self.columnB]
        return X


class CustomMLBEncoding(BaseEstimator, TransformerMixin):
    def __init__(self, column_index):
        self.column_index = column_index
        self.classes = None

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Copy nd array
        X = X.copy()
        # Create multilabel binarizer
        mlb = MultiLabelBinarizer()
        # Get data from the target column i.e "RW,ST,CB"
        target_column = X[:, self.column_index]
        """
        Now for fit_transform we need 2d array [[], [], ...] like this
        so we iterate over each row and split with , and then append each array as a row in 2d array
        then applied mlb
        """
        split_columns = [row.split(',') for row in target_column]
        new_multiple_columns = mlb.fit_transform(split_columns)
        # We want the column passed to be removed from our dataset
        X = np.delete(X, self.column_index, axis=1)
        # Store the classes generated from the mlb as it will be used as new columns in the data when create df
        self.classes = mlb.classes_
        # Concat the new data generated after transformation in the exiting data and return
        X = np.concatenate((X, new_multiple_columns), axis=1)
        return X
        """
        Below code is used if you are getting dataframe in X rather than np.ndarray
            values_df = pd.DataFrame(
                mlb.fit_transform(split_columns), columns=mlb.classes_, index=X.index
            )
            X.drop(columns=[self.column], inplace=True)
            X = pd.concat([X, values_df], axis=1)
        """
