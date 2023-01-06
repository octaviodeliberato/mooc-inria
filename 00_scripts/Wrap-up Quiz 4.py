import pandas as pd

ames_housing = pd.read_csv("00_data/ames_housing_no_missing.csv")
target_name = "SalePrice"
data = ames_housing.drop(columns=target_name)
target = ames_housing[target_name]

from sklearn.compose import make_column_selector

select_numeric_cols = make_column_selector(dtype_exclude="object")

numerical_features = select_numeric_cols(data)

numerical_features = [
    "LotFrontage", "LotArea", "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2",
    "BsmtUnfSF", "TotalBsmtSF", "1stFlrSF", "2ndFlrSF", "LowQualFinSF",
    "GrLivArea", "BedroomAbvGr", "KitchenAbvGr", "TotRmsAbvGrd", "Fireplaces",
    "GarageCars", "GarageArea", "WoodDeckSF", "OpenPorchSF", "EnclosedPorch",
    "3SsnPorch", "ScreenPorch", "PoolArea", "MiscVal",
]

data_numerical = data[numerical_features]

from sklearn.linear_model import RidgeCV
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('linear_model', Ridge(alpha=0))
])

# Use the pipeline to fit and predict on your data
cv_results = cross_validate(pipe, data_numerical, target, cv=10, 
                            return_estimator=True)

coefs=[coef for coef in cv_results['estimator'][1].named_steps['linear_model'].coef_]
print("Coefficients: {}".format(coefs))


pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('linear_model', Ridge(alpha=1))
])

# Use the pipeline to fit and predict on your data
cv_results = cross_validate(pipe, data_numerical, target, cv=10, 
                            return_estimator=True)

coefs=[coef for coef in cv_results['estimator'][5].named_steps['linear_model'].coef_]
print("Coefficients: {}".format(coefs))


import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, KFold

# Create the pipeline
model = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', Ridge())
])

# Create a KFold object with 5 folds
kfold = KFold(n_splits=10)

# Initialize a list to store the coefficients for each fold
coefs = []
X=data_numerical.to_numpy()
y=target

# Loop through the folds
for train_index, test_index in kfold.split(X):
    # Split the data into train and test sets
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # Fit the model to the training data
    model.fit(X_train, y_train)
    
    # Get the coefficients from the Ridge regression model
    coefs.append(model.named_steps['regressor'].coef_)

# Convert the list of arrays into a single array
coefs = np.concatenate(coefs)

# Create a figure
fig, ax = plt.subplots()

# Create a boxplot of the coefficients
ax.boxplot(coefs.T)

# Show the plot
plt.show()


# Load the data
X = data_numerical.to_numpy()
y = target

# Create the pipeline
model = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', RidgeCV(alphas=np.logspace(-3, 3, num=101)))
])

# Use 5-fold cross-validation to evaluate the model
scores = cross_val_score(model, X, y, cv=10)

# Fit the model to the full dataset
model.fit(X, y)

# Get the alpha_ parameter from the RidgeCV model
alpha_ = model.named_steps['regressor'].alpha_
print("Alpha: {}".format(alpha_))

# Print the results
print("Cross validation scores: {}".format(scores))
print("Mean score: {:.2f}".format(np.mean(scores)))
print("Standard deviation: {:.2f}".format(np.std(scores)))


adult_census = pd.read_csv("00_data/adult-census.csv")
target = adult_census["class"]
data = adult_census.select_dtypes(["integer", "floating"])
data = data.drop(columns=["education-num"])

data.info()

from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier

# Load the data
X = data.to_numpy()
y = target

# Create the linear model
linear_model = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression())
])

# Create the dummy model
dummy_model = DummyClassifier(strategy='most_frequent')

# Create a KFold object with 10 folds
kfold = KFold(n_splits=10)

# Evaluate the linear model using 10-fold cross-validation
linear_scores = cross_val_score(linear_model, X, y, cv=kfold)

# Evaluate the dummy model using 10-fold cross-validation
dummy_scores = cross_val_score(dummy_model, X, y, cv=kfold)

# Print the results
print("Linear model accuracy: {:.2f} (+/- {:.2f})".format(np.mean(linear_scores), np.std(linear_scores)))
print("Dummy model accuracy: {:.2f} (+/- {:.2f})".format(np.mean(dummy_scores), np.std(dummy_scores)))



# Create the linear model
linear_model = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression())
])

# Create the dummy model
dummy_model = DummyClassifier(strategy='most_frequent')

# Create a KFold object with 10 folds
kfold = KFold(n_splits=10)

# Initialize a counter for the number of times the linear model has a better score
better_score_count = 0

# Loop through the folds
for train_index, test_index in kfold.split(X):
    # Split the data into train and test sets
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # Fit the linear model to the training data
    linear_model.fit(X_train, y_train)
    
    # Get the linear model score on the test set
    linear_score = linear_model.score(X_test, y_test)
    
    # Fit the dummy model to the training data
    dummy_model.fit(X_train, y_train)
    
    # Get the dummy model score on the test set
    dummy_score = dummy_model.score(X_test, y_test)
    
    # Increment the counter if the linear model has a better score
    if linear_score > dummy_score:
        better_score_count += 1

# Select the range which the count belongs to
if better_score_count == 0:
    range_ = "0"
elif better_score_count > 0 and better_score_count <= 3:
    range_ = "1-3"
elif better_score_count > 3 and better_score_count <= 6:
    range_ = "4"



# Use 10-fold cross-validation to evaluate the model
scores = cross_val_score(linear_model, X, y, cv=10)

# Fit the model to the full dataset
linear_model.fit(X, y)

# Get the coefficients of the logistic regression model
coefs = linear_model.named_steps['classifier'].coef_
print("Coefficients: {}".format(coefs))

adult_census = pd.read_csv("00_data/adult-census.csv")
target = adult_census["class"]
data = adult_census.drop(columns=["class", "education-num"])

X=data.to_numpy()
y=target

from sklearn.compose import ColumnTransformer

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder



# Split the data into numerical and categorical features
X_num = data.select_dtypes(include=['integer', 'floating']).to_numpy()
num_cols=data.select_dtypes(include=['integer', 'floating']).columns
X_cat = data.select_dtypes(include='object').to_numpy()
cat_cols=data.select_dtypes(include=['object']).columns

# Preprocess the numerical and categorical features separately
preprocessor_num = StandardScaler()
preprocessor_cat = OneHotEncoder()

# Use ColumnTransformer to preprocess the numerical and categorical features
preprocessor = ColumnTransformer([
    ('num', preprocessor_num, num_cols),
    ('cat', preprocessor_cat, cat_cols)
])

# Create the pipeline
model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])

# Use 10-fold cross-validation to evaluate the model
scores = cross_val_score(model, data, y, cv=10)

# Print the results
print("Cross validation scores: {}".format(scores))
print("Mean score: {:.2f}".format(np.mean(scores)))
print("Standard deviation: {:.2f}".format(np.std(scores)))


# Use 10-fold cross-validation to evaluate the model
scores = cross_val_score(model, data, y, cv=10)

# Fit the model to the full dataset
model.fit(data, y)

# Get the coefficients of the logistic regression model
coefs = model.named_steps['classifier'].coef_
print("Coefficients: {}".format(coefs))