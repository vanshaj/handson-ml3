import os

from load import load
from binn import binn, split
from graphs import correlation


def main():
    folder = os.getcwd()
    filename = "data/Fifa23Players.csv"
    filepath = os.path.abspath(f"{folder}/{filename}")
    df = load(filepath)
    df = binn(df)
    train_set, test_set = split(df)
    # histogram(train_set)
    # scatterPlot(train_set, "Potential", "Value")
    correlation(train_set, ["Value", "Overall", "Potential", "Wage"])


if __name__ == "__main__":
    main()
