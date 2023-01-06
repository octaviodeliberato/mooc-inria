import pandas as pd

ames_housing = pd.read_csv("00_data/ames_housing_no_missing.csv")
target_name = "SalePrice"
data = ames_housing.drop(columns=target_name)
target = ames_housing[target_name]

numerical_features = [
    "LotFrontage", "LotArea", "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2",
    "BsmtUnfSF", "TotalBsmtSF", "1stFlrSF", "2ndFlrSF", "LowQualFinSF",
    "GrLivArea", "BedroomAbvGr", "KitchenAbvGr", "TotRmsAbvGrd", "Fireplaces",
    "GarageCars", "GarageArea", "WoodDeckSF", "OpenPorchSF", "EnclosedPorch",
    "3SsnPorch", "ScreenPorch", "PoolArea", "MiscVal",
]

data_numerical = data[numerical_features]

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, cross_validate

linear_reg = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LinearRegression())
])

linear_reg_scores = cross_val_score(
    estimator=linear_reg,
    X=data_numerical,
    y=target,
    cv=10
)

# Create the decision tree regressor
tree_reg = DecisionTreeRegressor(random_state=0)

tree_reg_scores = cross_val_score(
    estimator=tree_reg,
    X=data_numerical,
    y=target,
    cv=10
)

linear_reg_scores > tree_reg_scores

from sklearn.model_selection import GridSearchCV, KFold

# Create the parameter grid
param_grid = {'max_depth': range(1, 16)}

# Create the inner and outer cross-validation objects
inner_cv = KFold(n_splits=10, shuffle=True, random_state=0)
outer_cv = KFold(n_splits=10, shuffle=True, random_state=0)

# Create the grid search object
grid_search = GridSearchCV(tree_reg, param_grid, cv=inner_cv)

# Perform the nested cross-validation
# scores = cross_val_score(grid_search, data_numerical, target, cv=outer_cv)
# Print the average score
# print("Average score: {:.2f}".format(scores.mean()))
tree_fit = cross_validate(grid_search, data_numerical, target, cv=outer_cv,
                          return_estimator=True)

for i in range(0, len(tree_fit['estimator'])):
    print(tree_fit['estimator'][i].best_params_)
    print(tree_fit['test_score'][i].mean() > linear_reg_scores[i])

data_categorical = data.select_dtypes(include="object")
data_full = pd.concat([data_categorical, data_numerical], axis=1)

tree_fit = cross_validate(grid_search, data_numerical, target, cv=outer_cv,
                          return_estimator=True)

for i in range(0, len(tree_fit['estimator'])):
    print(tree_fit['estimator'][i].best_params_)
    print(tree_fit['test_score'][i].mean() > linear_reg_scores[i])

from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer([
    ('ordinal_encoder', OrdinalEncoder(handle_unknown="ignore", unknown_value=-1), data_categorical.columns)
], remainder='passthrough')

tree_model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', DecisionTreeRegressor(random_state=0, max_depth=7))
])

tree_fit = cross_validate(tree_model, data_full, target, cv=outer_cv)

for i in range(0, 10):
    print(tree_fit['test_score'][i].mean())
    print(tree_fit['test_score'][i].mean() > linear_reg_scores[i])