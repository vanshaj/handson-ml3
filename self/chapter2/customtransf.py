
from sklearn.base import BaseEstimator, TransformerMixin


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

