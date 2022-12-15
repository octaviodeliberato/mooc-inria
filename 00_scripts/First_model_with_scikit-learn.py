import pandas as pd

adult_census = pd.read_csv("00_data/adult-census-numeric.csv")

adult_census.head()

target_name = "class"
target = adult_census[target_name]
target

data = adult_census.drop(columns=target_name)
data.head()

data.columns

print(f"The testing dataset contains {data.shape[0]} samples and "
      f"{data.shape[1]} features")

from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier()

_ = model.fit(data, target)

target_predicted = model.predict(data)

target_predicted[:5]

print(f"Number of correct predictions: "
      f"{(target[:5] == target_predicted[:5]).sum()} / 5")

(target == target_predicted).mean()

adult_census_test = pd.read_csv('00_data/adult-census-numeric-test.csv')

target_test = adult_census_test[target_name]

data_test = adult_census_test.drop(columns=target_name)

print(f"The testing dataset contains {data_test.shape[0]} samples and "
      f"{data_test.shape[1]} features")

target_predicted = model.predict(data_test)

(target_test == target_predicted).mean()

accuracy = model.score(data_test, target_test)
model_name = model.__class__.__name__

print(f"The test accuracy using a {model_name} is "
      f"{accuracy:.3f}")
