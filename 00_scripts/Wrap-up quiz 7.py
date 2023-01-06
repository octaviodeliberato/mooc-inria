import pandas as pd

cycling = pd.read_csv("00_data/bike_rides.csv", index_col=0,
                      parse_dates=True)
cycling.index.name = ""
target_name = "power"
data, target = cycling.drop(columns=target_name), cycling[target_name]
data

import numpy as np

data_new = data.copy()
data_new['speed_cubed'] = data['speed'] ** 3
data_new['sin_slope'] = data['speed'] * np.sin(np.arctan(data['slope']))
data_new['acceleration'] = data['acceleration'].clip(lower=0)
data_new['speed_acceleration'] = data['speed'] * data['acceleration']

data_new = data_new[['speed_cubed', 'speed', 'sin_slope', 'speed_acceleration']]

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import ShuffleSplit, cross_validate
from sklearn.metrics import mean_absolute_error

# Assuming the new data frame is called "data_new" and the target column is called "target"
X = data_new
y = target

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create the model
model = RidgeCV(cv=4)

# Create the cross-validation object
cv = ShuffleSplit(n_splits=4, random_state=0, test_size=0.2)

# Compute the cross-validation scores
scores = cross_validate(model, X_scaled, y, cv=cv, return_estimator=True, 
                        return_train_score=True, 
                        scoring='neg_mean_absolute_error')

# Print the results
print(f"Train scores: {-scores['train_score']}")
print(f"Test scores: {-scores['test_score']}")

X['sin_slope'].mean()
-scores['test_score'].mean()

for i, model in enumerate(scores['estimator']):
    print(f"Model {i+1}: weights = {model.coef_}, intercept = {model.intercept_}")

from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor

X = data
y = target

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create the model
model = HistGradientBoostingRegressor(max_iter=1000, early_stopping=True)

# Create the cross-validation object
cv = ShuffleSplit(n_splits=4, random_state=0, test_size=0.2)

# Compute the cross-validation scores
scores = cross_validate(model, X_scaled, y, cv=cv, return_estimator=True, 
                        return_train_score=True, 
                        scoring='neg_mean_absolute_error')

# Print the results
print(f"Train scores: {-scores['train_score']}")
print(f"Test scores: {-scores['test_score']}")

-scores['test_score'].mean()

len(np.unique(X.index.date))
len(np.unique(X.index.time))


from sklearn.model_selection import LeaveOneGroupOut
df=data

# Assuming the data frame is called "df" and it has a "group" column
# Encode the group values as integer indices
group, group_labels = pd.factorize(df.index.date)

# Get the group indices as a 1D array
df['group'] = group

# Create the cross-validation object
cv = LeaveOneGroupOut()

# Iterate over the splits
linear_model = RidgeCV(cv=4)
gradient_boosting_model = HistGradientBoostingRegressor(max_iter=1000, 
                                                        early_stopping=True)

X=df.copy()
X.drop(columns='group')
for train_index, test_index in cv.split(X, y, group):
    # Get the train and test data for this split
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Fit and evaluate the linear model on this split
    linear_model.fit(X_train, y_train)
    linear_score = linear_model.score(X_test, y_test)

    # Fit and evaluate the gradient boosting model on this split
    gradient_boosting_model.fit(X_train, y_train)
    gradient_boosting_score = gradient_boosting_model.score(X_test, y_test)

    # Print the results
    print(f"Linear model score: {linear_score:.2f}")
    print(f"Gradient boosting model score: {gradient_boosting_score:.2f}")
