import pandas as pd
import numpy as np
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

FILE_NAME = "./data/Fifa23Players.csv"


def remove_null_values(df: pd.DataFrame, property: str):
    # print(df.isnull().any(axis=1))
    # df_num_prop = df.select_dtypes(include=[np.number])
    df_prop = df[property].to_frame()
    imputer = SimpleImputer(strategy="median")
    imputer.fit(df_prop)
    df_non_empty = imputer.transform(df_prop)
    df[property] = df_non_empty
    df.to_csv("./data/Fifa23Players_new.csv")


# Here we will try to encode column Club Position
def encode_column(df: pd.DataFrame, column: str):
    df_prop = df[[column]]
    encoder = OneHotEncoder(handle_unknown="ignore")
    df_prop_encoded = encoder.fit_transform(df_prop)
    print(df_prop_encoded.toarray())
    print(encoder.categories_)
    df_output = pd.DataFrame(
        df_prop_encoded,
        columns=encoder.get_feature_names_out(),
        index=df_prop.index,
    )
    print(df_output)


def top_player_by_clubs(df, n: int):
    top_clubs = df["Club Name"].value_counts().head(n)
    plt.figure(figsize=(20, 8))
    plt.bar(top_clubs.index, top_clubs.values)
    plt.xticks(rotation=46, ha="right")
    plt.xlabel("Club Name")
    plt.ylabel("Number of Players")
    plt.title("Top 30 Club Names by Number of Players")
    plt.tight_layout()
    plt.show()


def corr_with_properties(df, props):
    scatter_matrix(df[props], alpha=0.2, figsize=(12, 8))
    plt.tight_layout()
    plt.show()


def main():
    train_set = pd.DataFrame()
    test_set = pd.DataFrame()

    players_all = pd.read_csv(FILE_NAME)
    players = players_all.drop("Nationality", axis=1)
    # print(players.info())

    players["overall_cat"] = pd.cut(
        players["Overall"], bins=[0, 60, 70, 90, 100], labels=[1, 2, 3, 4]
    )
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_set_index, test_set_index in split.split(players, players["overall_cat"]):
        train_set = players.iloc[train_set_index]
        test_set = players.iloc[test_set_index]
    print(len(train_set))
    print(len(test_set))

    # Drop overall_cat now
    train_set = train_set.drop("overall_cat", axis=1)
    test_set = test_set.drop("overall_cat", axis=1)

    """
    Identify top players by n club names
    top_player_by_clubs(players, 20)
    """

    """
    Plot graph of different attributes among each other to understand the correlation
    Also we can create custom attributes by performing operation on multiple attributes
    In this case we identify a new attributes
    1. growth_potential which is the 'current potential of the player' - 'overall rating of the player'
    2. attack
    3. defense

    attacking_attributes = ["Pace Total", "Shooting Total", "Dribbling Total"]
    defensive_attributes = ["Defending Total", "Physicality Total"]
    train_set["attack"] = train_set[attacking_attributes].mean(axis=1)
    train_set["defense"] = train_set[defensive_attributes].mean(axis=1)
    train_set["growth_potential"] = train_set["Potential"] = train_set["Overall"]
    corr_with_properties(
        train_set,
        ["Overall", "Value(in Euro)", "Age", "growth_potential", "attack", "defense"],
    )
    """
    """
    Separate predictors from labels

    fifa_train = train_set.drop("Value(in Euro)", axis=1)
    fifa_train_label = train_set["Value(in Euro)"].copy()
    """

    remove_null_values(train_set, "Age")
    encode_column(train_set, "Club Position")


if __name__ == "__main__":
    main()
