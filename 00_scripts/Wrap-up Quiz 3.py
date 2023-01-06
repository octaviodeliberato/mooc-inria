import pandas as pd

penguins = pd.read_csv("00_data/penguins.csv")

columns = ["Body Mass (g)", "Flipper Length (mm)", "Culmen Length (mm)"]
target_name = "Species"

# Remove lines with missing values for the columns of interest
penguins_non_missing = penguins[columns + [target_name]].dropna()

data = penguins_non_missing[columns]
target = penguins_non_missing[target_name]

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

model = Pipeline(steps=[
    ("preprocessor", StandardScaler()),
    ("classifier", KNeighborsClassifier(n_neighbors=5)),
])

from sklearn.model_selection import cross_validate

cv_results = cross_validate(model, data, target, cv=10, scoring="balanced_accuracy")
scores = cv_results["test_score"]
print(f"Accuracy score via cross-validation:\n"
      f"{scores.mean():.3f} ± {scores.std():.3f}")

model.get_params()

cv_results = []
for K in [5, 51]:
    model.set_params(classifier__n_neighbors=K)
    cv_res = cross_validate(model, data, target, cv=10)
    cv_results.append(cv_res)
    scores = cv_res["test_score"]
    print(f"Accuracy score via cross-validation with K={K}:\n"
          f"{scores.mean():.3f} ± {scores.std():.3f}")
    
(cv_results[0]['test_score'] > cv_results[1]['test_score'])

cv_results = []
for K in [5, 101]:
    model.set_params(classifier__n_neighbors=K)
    cv_res = cross_validate(model, data, target, cv=10)
    cv_results.append(cv_res)
    scores = cv_res["test_score"]
    print(f"Accuracy score via cross-validation with K={K}:\n"
          f"{scores.mean():.3f} ± {scores.std():.3f}")
    
(cv_results[0]['test_score'] > cv_results[1]['test_score'])


model = Pipeline(steps=[
    ("preprocessor", StandardScaler()),
    ("classifier", KNeighborsClassifier(n_neighbors=5)),
])

cv_results = cross_validate(model, data, target, cv=10, scoring="balanced_accuracy")
scores = cv_results["test_score"]
print(f"Accuracy score via cross-validation:\n"
      f"{scores.mean():.3f} ± {scores.std():.3f}")

model_raw = Pipeline(steps=[
    ("classifier", KNeighborsClassifier(n_neighbors=5)),
])

cv_results = cross_validate(model_raw, data, target, cv=10, scoring="balanced_accuracy")
scores = cv_results["test_score"]
print(f"Accuracy score via cross-validation:\n"
      f"{scores.mean():.3f} ± {scores.std():.3f}")

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer


all_preprocessors = [
    None,
    StandardScaler(),
    MinMaxScaler(),
    QuantileTransformer(n_quantiles=100),
    PowerTransformer(method="box-cox"),
]

from sklearn.model_selection import GridSearchCV

model = Pipeline(steps=[
    ("preprocessor", all_preprocessors[1]),
    ("classifier", KNeighborsClassifier()),
])

param_grid = {
    'classifier__n_neighbors': (5, 51, 101)
}

%%time
best = []
for preproc in all_preprocessors:
    
    model = Pipeline(steps=[
        ("preprocessor", preproc),
        ("classifier", KNeighborsClassifier()),
    ])
    
    model_grid_search = GridSearchCV(model, param_grid=param_grid,
                                     n_jobs=2, cv=10, scoring="balanced_accuracy")
    
    model_grid_search.fit(X=data, y=target)
    best.append([
        model_grid_search.best_params_,
        model_grid_search.best_estimator_,
        model_grid_search.best_score_]
    )
    
cv_results = cross_validate(
    model_grid_search,
    data,
    target,
    cv=10,
    n_jobs=2,
    scoring="balanced_accuracy",
    return_estimator=True,
)
cv_results = pd.DataFrame(cv_results)
cv_test_scores = cv_results['test_score']

print(
    "Generalization score with hyperparameters tuning:\n"
    f"{cv_test_scores.mean():.3f} +/- {cv_test_scores.std():.3f}"
)