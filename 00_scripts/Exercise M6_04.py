from sklearn.datasets import fetch_california_housing

data, target = fetch_california_housing(return_X_y=True, as_frame=True)
target *= 100  # rescale the target in k$

from sklearn.ensemble import GradientBoostingRegressor

model = GradientBoostingRegressor(n_estimators=1000, n_iter_no_change=5)

# Create the parameter grid
param_grid = {
    'max_depth': [3, 8],
    'max_leaf_nodes': [15, 31],
    'learning_rate': [0.1, 1]
}

from sklearn.model_selection import GridSearchCV, KFold, cross_validate

# Create the inner and outer cross-validation objects
inner_cv = KFold(n_splits=10, shuffle=True, random_state=0)
# outer_cv = KFold(n_splits=10, shuffle=True, random_state=0)

# Create the grid search object
grid_search = GridSearchCV(model, param_grid, cv=inner_cv)

# Perform the nested cross-validation
# model_fit = cross_validate(grid_search, data, target, cv=outer_cv,
#                            return_estimator=True)

model.fit(X=data, y=target)

for i in range(0, len(model.estimators_)):
    print(model.estimators_[i])