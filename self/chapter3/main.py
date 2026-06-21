import os

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from load import load, unload
from sklearn.pipeline import Pipeline
import pandas as pd

def main():
    folder = os.getcwd()
    filename = "self/chapter2/data/Fifa23Players.csv"
    new_filename = "self/chapter2/data/Fifa23PlayersTransformed.csv"
    filepath = os.path.abspath(f"{folder}/{filename}")
    new_filepath = os.path.abspath(f"{folder}/{new_filename}")
    df = load(filepath)

if __name__ == "__main__":
    main()
