import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt


def histogram(df: pd.DataFrame):
    customDf = df[["Value", "Wage"]]
    customDf.hist(figsize=(15, 10), bins=10, edgecolor="black")
    plt.tight_layout()
    plt.show()


def scatterPlot(df: pd.DataFrame, xAxis: str, yAxis: str):
    plt.figure(figsize=(10, 6))
    plt.scatter(x=df[xAxis], y=df[yAxis], alpha=0.3)
    plt.xlabel(xAxis)
    plt.ylabel(yAxis)
    plt.title(f"{xAxis} vs {yAxis}")
    plt.tight_layout()
    plt.show()


def correlation(df: pd.DataFrame, targetColumns: []):
    corr_matrix = df.corr(numeric_only=True)
    print(corr_matrix["Value"].sort_values(ascending=False))
    scatter_matrix(df[targetColumns], alpha=0.3, figsize=(10, 6))
    plt.show()
