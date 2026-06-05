import os

from sklearn.preprocessing import StandardScaler

from load import load, unload
from binn import binn, split
from graphs import correlation
from cleanup import clean, one_hot_encoding
from sklearn.pipeline import Pipeline
import customtransf


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
    df_new = one_hot_encoding(df, "Positions Played")
    val_per_wage_t = customtransf.CustomTransf("Value", "Wage")
    df_new = val_per_wage_t.fit_transform(df_new)
    unload(df_new, new_filepath)



if __name__ == "__main__":
    main()
