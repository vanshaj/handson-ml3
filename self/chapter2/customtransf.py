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
    def __init__(self):
        self.classes = None

    def fit(self, X, y=None):
        return self

    # In order to support the functionality of the get_feature_names_out in the processor
    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.classes)

    def transform(self, X: np.ndarray):
        # Copy nd array
        X = X.copy()
        # Create multilabel binarizer
        mlb = MultiLabelBinarizer()
        # Get data from the target column i.e "RW,ST,CB"
        # index of column will always be 0 as we will pass only 1 column in our transformer
        target_column = X[:, 0]
        """
        Now for fit_transform we need 2d array [[], [], ...] like this
        so we iterate over each row and split with , and then append each array as a row in 2d array
        then applied mlb
        """
        split_columns = [row.split(',') for row in target_column]
        new_multiple_columns = mlb.fit_transform(split_columns)
        self.classes = mlb.classes_
        return new_multiple_columns