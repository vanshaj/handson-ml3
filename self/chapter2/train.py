from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
import numpy as np

def lin_train(X_train, y_train, X_test, y_test):
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)

    # Evaluate on test set
    lin_prediction = lin_reg.predict(X_test);
    lin_mse = mean_squared_error(y_test, lin_prediction)
    lin_rmse = np.sqrt(lin_mse)
    print(f"Linear Regression Training RMSE: {lin_rmse:.2f}\n")

def tree_train(X_train, y_train, X_test, y_test):
    tree_reg = DecisionTreeRegressor(random_state=42)
    tree_reg.fit(X_train, y_train)

    # Evaluate on test set
    tree_prediction = tree_reg.predict(X_test);
    tree_mse = mean_squared_error(y_test, tree_prediction)
    tree_rmse = np.sqrt(tree_mse)
    print(f"Tree Regression Training RMSE: {tree_rmse:.2f}\n")

def random_forrest_train(X_train, y_train, X_test, y_test):
    # n_estimator tells us about the number of trees you want
    # max_features tells us with how many features you want to train the model
    tree_reg = RandomForestRegressor(random_state=42, n_estimators=30, max_features=6)
    tree_reg.fit(X_train, y_train)

    # Evaluate on test set
    tree_prediction = tree_reg.predict(X_test);
    tree_mse = mean_squared_error(y_test, tree_prediction)
    tree_rmse = np.sqrt(tree_mse)
    print(f"Random Forrest Regression Training RMSE: {tree_rmse:.2f}\n")

def random_forrest_train_gridcv(X_train, y_train, X_test, y_test):
    tree_reg = RandomForestRegressor(random_state=42)
    param_grid = [{
        'n_estimators':[30, 50, 80],
        'max_features':[4, 6, 8]
    }]
    gscv = GridSearchCV(estimator=tree_reg, param_grid=param_grid, cv=5, scoring="neg_mean_squared_error")
    gscv.fit(X_train, y_train)

    # Evaluate on test set
    tree_prediction = gscv.predict(X_test);
    tree_mse = mean_squared_error(y_test, tree_prediction)
    tree_rmse = np.sqrt(tree_mse)
    print("Best Params:", gscv.best_params_)
    print("Best Estimator:", gscv.best_estimator_)
    print(f"Random Forrest Regression Training RMSE: {tree_rmse:.2f}\n")


"""
# Linear Regression CV
lin_scores = cross_val_score(lin_reg, fifa_prepared, fifa_labels,
                             scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
display_scores("Linear Regression", lin_rmse_scores)

# Decision Tree CV
tree_scores = cross_val_score(tree_reg, fifa_prepared, fifa_labels,
                             scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-tree_scores)
display_scores("Decision Tree", tree_rmse_scores)

# Random Forest CV (Note: This will take noticeably longer to execute)
forest_scores = cross_val_score(forest_reg, fifa_prepared, fifa_labels,
                                scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)
display_scores("Random Forest", forest_rmse_scores)
"""