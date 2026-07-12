import os
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from load import load, unload
from sklearn.pipeline import Pipeline
import pandas as pd
from binn import binn, split
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_curve, roc_curve, roc_auc_score
from plotting import plot_precision_recall_vs_threshold, plot_roc_curve

def main():
    folder = os.getcwd()
    filename = "self/chapter2/data/Fifa23Players.csv"
    new_filename = "self/chapter2/data/Fifa23PlayersTransformed.csv"
    filepath = os.path.abspath(f"{folder}/{filename}")
    new_filepath = os.path.abspath(f"{folder}/{new_filename}")
    df = load(filepath)
    FEATURES_TO_KEEP = df.select_dtypes(include=['int64','float64']).columns.tolist()
    # df_modelling = df[FEATURES_TO_KEEP].copy()

    df_modelling = binn(df)
    train_set, test_set = split(df_modelling)

    Y_train_set_raw = (train_set["Overall"] >= 80).astype(int)
    X_train_set_raw = train_set[FEATURES_TO_KEEP].copy()

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    df_transf = num_pipeline.fit_transform(X_train_set_raw)
    sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3, random_state=33)
    sgd_clf.fit(df_transf, Y_train_set_raw)

    X_test_set_raw = test_set[FEATURES_TO_KEEP].copy()
    df_test_transf = num_pipeline.transform(X_test_set_raw)
    prediction = sgd_clf.predict(df_test_transf[0:1])

    print("\n--- Single player prediction result ---")
    player_name = test_set.iloc[0]["Full Name"]
    print(f"Is this player {player_name} predicted to be Elite? {prediction[0]}")

    y_train_pred = cross_val_predict(sgd_clf, df_transf, Y_train_set_raw, cv=3)

    conf_matrix = confusion_matrix(Y_train_set_raw, y_train_pred)
    print("========Confusion Matrix======")
    print(conf_matrix)
    print("-" * 30)

    precision = precision_score(Y_train_set_raw, y_train_pred)
    recall = recall_score(Y_train_set_raw, y_train_pred)
    f1 = f1_score(Y_train_set_raw, y_train_pred)

    print("========Classification Metrics======")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print("-" * 30)

    y_scores = cross_val_predict(sgd_clf, df_transf, Y_train_set_raw, cv=3, method="decision_function")

    precisions, recalls, pr_thresholds = precision_recall_curve(Y_train_set_raw, y_scores)
    fpr, tpr, roc_thresholds = roc_curve(Y_train_set_raw, y_scores)
    auc_score = roc_auc_score(Y_train_set_raw, y_scores)

    print("========ROC AUC Score======")
    print(f"ROC AUC: {auc_score:.4f}")
    print("-" * 30)

    plot_precision_recall_vs_threshold(precisions, recalls, pr_thresholds)
    plt.savefig(os.path.join(folder, "./self/chapter3/precision_recall_vs_threshold.png"))

    plot_roc_curve(fpr, tpr, label=f"SGDClassifier (AUC = {auc_score:.4f})")
    plt.savefig(os.path.join(folder, "./self/chapter3/roc_curve.png"))


if __name__ == "__main__":
    main()
