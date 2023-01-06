from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

data, target = fetch_california_housing(return_X_y=True, as_frame=True)
target *= 100  # rescale the target in k$
data_train, data_test, target_train, target_test = train_test_split(
    data, target, random_state=0, test_size=0.5)

from sklearn.ensemble import GradientBoostingRegressor

# Create the model with max_depth=5 and learning_rate=0.5
model = GradientBoostingRegressor(max_depth=5, learning_rate=0.5)

# Fit the model to the training data
model.fit(data_train, target_train)

from sklearn.ensemble import RandomForestRegressor

forest = RandomForestRegressor(max_depth=None)

forest.fit(data_train, target_train)

from sklearn.model_selection import validation_curve

# Create the range of values for the number of trees
param_range = [1, 2, 5, 10, 20, 50, 100]

# Calculate the validation curve
train_scores, test_scores = validation_curve(
    forest, data_train, target_train, param_name="n_estimators", 
    param_range=param_range, scoring="neg_mean_absolute_error", cv=5
)

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Convert the scores to means and standard deviations
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

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

model_2 = GradientBoostingRegressor(n_estimators=1000, n_iter_no_change=5)

train_scores, test_scores = validation_curve(
    model_2, data_train, target_train, param_name="n_estimators", 
    param_range=param_range, scoring="neg_mean_absolute_error", cv=5
)

# Convert the scores to means and standard deviations
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

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