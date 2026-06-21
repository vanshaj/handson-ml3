import os

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from load import load, unload
from binn import binn, split
from self.chapter2.train import lin_train, tree_train, random_forrest_train, random_forrest_train_gridcv
from sklearn.pipeline import Pipeline
import pandas as pd

# FEATURES_TO_KEEP = [
#     'Overall', 'Potential', 'Age', 'Wage', 'International Reputation',
#     'Weak Foot Rating', 'Skill Moves',
#     'Height(in cm)', 'Weight(in kg)', 'TotalStats', 'BaseStats',
#     'Pace Total', 'Shooting Total', 'Passing Total', 'Dribbling Total', 
#     'Defending Total', 'Physicality Total', 'Reactions', 'Composure',
#     'Attacking Work Rate', 'Defensive Work Rate'
# ]

def main():
    folder = os.getcwd()
    filename = "self/chapter2/data/Fifa23Players.csv"
    new_filename = "self/chapter2/data/Fifa23PlayersTransformed.csv"
    filepath = os.path.abspath(f"{folder}/{filename}")
    new_filepath = os.path.abspath(f"{folder}/{new_filename}")
    df = load(filepath)
    FEATURES_TO_KEEP = df.select_dtypes(include=['int64','float64']).columns.tolist()
    df_modelling = df[FEATURES_TO_KEEP].copy()
    
    # Create test and train set
    df_modelling = binn(df_modelling)
    train_set, test_set = split(df_modelling)
    y_train = train_set["Value"]
    X_train = train_set.drop("Value", axis=1)

    y_test = test_set["Value"]
    X_test = test_set.drop("Value", axis=1)

    # Now tranform training data
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    df_transf_arr = num_pipeline.fit_transform(X_train)
    # Remove value as that will not be a column for X data frames both test and train
    FEATURES_TO_KEEP.remove("Value")
    X_train_transformed = pd.DataFrame(df_transf_arr, columns=FEATURES_TO_KEEP, index=X_train.index)

    df_transf_test_arr = num_pipeline.fit_transform(X_test)
    X_test_transformed = pd.DataFrame(df_transf_test_arr, columns=FEATURES_TO_KEEP, index=X_test.index)
    
    # lin_train(X_train_transformed, y_train, X_test_transformed, y_test)
    # tree_train(X_train_transformed, y_train, X_test_transformed, y_test)
    random_forrest_train_gridcv(X_train_transformed, y_train, X_test_transformed, y_test)

    # lin_train(X_train_transformed, y_train, X_train_transformed, y_train)
    # tree_train(X_train_transformed, y_train, X_train_transformed, y_train)
    # random_forrest_train(X_train_transformed, y_train, X_train_transformed, y_train)

    # transformed_train_set = pipeline.create_pipeline(train_set, "Positions Played")
    # unload(transformed_train_set, new_filepath)



if __name__ == "__main__":
    main()
