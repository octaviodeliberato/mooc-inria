import pandas as pd

adult_census = pd.read_csv("00_data/adult-census.csv")
# drop the duplicated column `"education-num"` as stated in the first notebook
adult_census = adult_census.drop(columns="education-num")

target_name = "class"
target = adult_census[target_name]

data = adult_census.drop(columns=[target_name])

data["native-country"].value_counts().sort_index()

data.dtypes

data.T.head()

from sklearn.compose import make_column_selector as selector

categorical_columns_selector = selector(dtype_include=object)
categorical_columns = categorical_columns_selector(data)
categorical_columns

data_categorical = data[categorical_columns]
data_categorical.head()

print(f"The dataset is composed of {data_categorical.shape[1]} features")

from sklearn.preprocessing import OrdinalEncoder

education_column = data_categorical[["education"]]

encoder = OrdinalEncoder()
education_encoded = encoder.fit_transform(education_column)
education_encoded

encoder.categories_

data_encoded = encoder.fit_transform(data_categorical)
data_encoded[:5]

print(
    f"The dataset encoded contains {data_encoded.shape[1]} features"
)

from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse=False)
education_encoded = encoder.fit_transform(education_column)
education_encoded

feature_names = encoder.get_feature_names(input_features=["education"])
education_encoded = pd.DataFrame(education_encoded, columns=feature_names)
education_encoded

print(
    f"The dataset is composed of {data_categorical.shape[1]} features"
)
data_categorical.T.head()

data_encoded = encoder.fit_transform(data_categorical)
data_encoded[:5]

columns_encoded = encoder.get_feature_names(data_categorical.columns)
data_categorical_encoded = pd.DataFrame(data_encoded, columns=columns_encoded)

data_categorical_encoded.head()

data["native-country"].value_counts()

from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

model = make_pipeline(
    OneHotEncoder(handle_unknown="ignore"), LogisticRegression(max_iter=500)
)

from sklearn.model_selection import cross_validate
cv_results = cross_validate(model, data_categorical, target)
cv_results

scores = cv_results["test_score"]
print(f"The accuracy is: {scores.mean():.3f} ± {scores.std():.3f}")

from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

categorical_preprocessor = OneHotEncoder(handle_unknown="ignore")
numerical_preprocessor = StandardScaler()

numerical_columns_selector = selector(dtype_exclude=object)
numerical_columns = numerical_columns_selector(data)
numerical_columns

preprocessor = ColumnTransformer([
    ('one_hot_encoder', categorical_preprocessor, categorical_columns),
    ('standard_scaler', numerical_preprocessor, numerical_columns)])

model = make_pipeline(preprocessor, LogisticRegression(max_iter=500))

cv_results = cross_validate(model, data, target, cv=10)
cv_results

scores = cv_results["test_score"]
print(f"The accuracy is: {scores.mean():.3f} ± {scores.std():.3f}")
