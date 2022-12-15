import pandas as pd

from sklearn.model_selection import train_test_split

adult_census = pd.read_csv("00_data/adult-census.csv")

target_name = "class"
target = adult_census[target_name]
data = adult_census.drop(columns=[target_name, "education-num"])

data_train, data_test, target_train, target_test = train_test_split(
    data, target, train_size=0.2, random_state=42)

from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as selector
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

categorical_preprocessor = OneHotEncoder(handle_unknown="ignore")

preprocessor = ColumnTransformer(
    [('cat_preprocessor', categorical_preprocessor,
      selector(dtype_include=object))],
    remainder='passthrough', sparse_threshold=0)

from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate

from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier

learning_rates = [0.01, 0.1, 1, 10]
max_nodes = [3, 10, 30]
cv_results = []

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", HistGradientBoostingClassifier(random_state=42))
])

for lr in learning_rates:
    
    for mn in max_nodes:
        
        model.set_params(classifier__learning_rate = lr, classifier__max_leaf_nodes = mn)
        
        res = cross_validate(
            model, data_train, target_train, cv=5, 
            scoring="balanced_accuracy",
            return_train_score=True
        )
        
        cv_results.append(res)

        res_df = pd.DataFrame(res)
        print(res_df[["train_score", "test_score"]].mean())
        print(f"learning rate = {lr:.2f} and max leaf nodes = {mn}\n")
    
# Best: learning rate = 0.10 and max leaf nodes = 30

model.set_params(classifier__learning_rate = 0.1, classifier__max_leaf_nodes = 30)

clf = model.fit(X=data_train, y=target_train)
clf.score(X=data_test, y=target_test)