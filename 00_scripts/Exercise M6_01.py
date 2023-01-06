from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

data, target = fetch_california_housing(as_frame=True, return_data_y=True)
target *= 100  # rescale the target in k$
data_train, data_test, target_train, target_test = train_test_split(
    data, target, random_state=0, test_size=0.5)

from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

# Create a DecisionTreeRegressor as the base estimator
base_estimator = DecisionTreeRegressor()

# Create the BaggingRegressor
regressor = BaggingRegressor(base_estimator=base_estimator)

# Train the regressor
regressor.fit(data_train, target_train)

# Predict on the test set
y_pred = regressor.predict(data_test)

# Evaluate the performance using the mean absolute error
mae = mean_absolute_error(target_test, y_pred)
print(f"Mean Absolute Error: {mae:.1f} %")

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

# Create a DecisionTreeRegressor as the base estimator
base_estimator = DecisionTreeRegressor()

# Create the BaggingRegressor
regressor = BaggingRegressor(base_estimator=base_estimator)

# Define the parameter distribution for the RandomizedSearchCV
param_distributions = {
    'n_estimators': randint(1, 100),
    'max_samples': randint(1, 100),
    'max_features': randint(1, 100),
    'bootstrap': [True, False],
    'bootstrap_features': [True, False]
}

# Create the RandomizedSearchCV instance
random_search = RandomizedSearchCV(
    estimator=regressor,
    param_distributions=param_distributions,
    scoring='neg_mean_absolute_error',
    cv=5,
    n_iter=10,
    random_state=42
)

# Fit the RandomizedSearchCV instance to the training data
random_search.fit(data_train, target_train)

# Print the best set of parameters found
print("Best set of parameters:", random_search.best_params_)

# Get the best estimator
best_estimator = random_search.best_estimator_

# Predict on the test set
target_pred = best_estimator.predict(data_test)

# Evaluate the performance using the mean absolute error
mae = mean_absolute_error(target_test, target_pred)
print(f"Mean Absolute Error: {mae:.1f} %")