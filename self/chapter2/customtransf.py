import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MultiLabelBinarizer


class CustomTransf(BaseEstimator, TransformerMixin):

    def __init__(self, columnA, columnB):
        self.numer = columnA
        self.denom = columnB

    def fit(self, X, y = None):
        return self

    def transform(self, X):
        X = X.copy()
        X["derived"] = X[self.numer] / X[self.denom]
        return X

class CustomMLBEncoding(BaseEstimator, TransformerMixin):
    
    def __init__(self, column):
        self.column = column

    def fit(self, X, y= None):
        return self

    def transform(self, X):
        X = X.copy()
        mlb = MultiLabelBinarizer()
        values = X[self.column].str.split(",")
        values_df = pd.DataFrame(
            mlb.fit_transform(values),
            columns=mlb.classes_,
            index=X.index
        )
        X.drop(columns=[self.column], inplace=True)
        X = pd.concat([X, values_df], axis=1)
        return X
