import pandas as pd

dataset = pd.read_csv("00_data/penguins.csv")

feature_names = [
    "Culmen Length (mm)",
    "Culmen Depth (mm)",
    "Flipper Length (mm)",
]
target_name = "Body Mass (g)"

dataset = dataset[feature_names + [target_name]].dropna(axis="rows", how="any")
dataset = dataset.sample(frac=1, random_state=0).reset_index(drop=True)
data, target = dataset[feature_names], dataset[target_name]

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, cross_validate

tree = DecisionTreeRegressor(random_state=0)
forest = RandomForestRegressor(random_state=0)

tree_scores = cross_val_score(estimator=tree, X=data, y=target, cv=10)

forest_scores = cross_val_score(estimator=forest, X=data, y=target, cv=10)

forest_scores > tree_scores

forest_5 = RandomForestRegressor(random_state=0, n_estimators=5)
forest_100 = RandomForestRegressor(random_state=0, n_estimators=100)

forest_5_scores = cross_val_score(estimator=forest_5, X=data, y=target, cv=10)
forest_100_scores = cross_val_score(estimator=forest_100, X=data, y=target, cv=10)

forest_100_scores > forest_5_scores

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import validation_curve


def plot_validation_curve(model, data, target, param_name, param_range):
    
    # Calculate the validation curve
    train_scores, test_scores = validation_curve(
        model, data, target, param_name=param_name, 
        param_range=param_range, scoring="neg_mean_absolute_error", cv=10
    )

    # Convert the scores to means and standard deviations
    train_scores_mean = -np.mean(train_scores, axis=1)
    train_scores_std = -np.std(train_scores, axis=1)
    test_scores_mean = -np.mean(test_scores, axis=1)
    test_scores_std = -np.std(test_scores, axis=1)

    # Plot the validation curve
    sns.lineplot(x=param_range, y=train_scores_mean, label="Training score")
    sns.lineplot(x=param_range, y=test_scores_mean, label="Test score")

    # Add the standard deviations to the plot
    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                    train_scores_mean + train_scores_std, alpha=0.2)
    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                    test_scores_mean + test_scores_std, alpha=0.2)

    # Add labels and show the plot
    plt.xlabel("Number of trees")
    plt.ylabel("Mean absolute error")
    plt.legend()
    plt.show()
    
    return {'train_scores':train_scores, 'test_scores': test_scores}

forest_5 = RandomForestRegressor(random_state=0, max_depth=5)

scores = plot_validation_curve(
    model=forest_5,
    data=data,
    target=target,
    param_name="n_estimators",
    param_range=[1, 2, 5, 10, 20, 50, 100, 200, 500, 1_000]
)


rf_1_tree = RandomForestRegressor(n_estimators=1, random_state=0)
cv_results_tree = cross_validate(
    rf_1_tree, data, target, cv=10, return_train_score=True
)
cv_results_tree["train_score"]

tree = DecisionTreeRegressor(random_state=0)
cv_results_tree = cross_validate(
    tree, data, target, cv=10, return_train_score=True
)
cv_results_tree["train_score"]


from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor

param_range = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1_000]

regressor = HistGradientBoostingRegressor(random_state=0)

scores = plot_validation_curve(
    data=data,
    model=regressor,
    param_name="max_iter", # number of trees
    param_range=param_range,
    target=target
)