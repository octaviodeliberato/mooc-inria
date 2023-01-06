import pandas as pd

penguins = pd.read_csv("00_data/penguins_classification.csv")
# only keep the Adelie and Chinstrap classes
penguins = penguins.set_index("Species").loc[
    ["Adelie", "Chinstrap"]].reset_index()

culmen_columns = ["Culmen Length (mm)", "Culmen Depth (mm)"]
target_column = "Species"

from sklearn.model_selection import train_test_split

penguins_train, penguins_test = train_test_split(penguins, random_state=0)

data_train = penguins_train[culmen_columns]
data_test = penguins_test[culmen_columns]

target_train = penguins_train[target_column]
target_test = penguins_test[target_column]

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

logistic_regression = make_pipeline(
    StandardScaler(), LogisticRegression(penalty="l2"))

from sklearn.inspection import DecisionBoundaryDisplay

from sklearn.preprocessing import OrdinalEncoder
from sklearn.inspection import DecisionBoundaryDisplay

Cs = [0.01, 0.1, 1, 10]

colors = ['red', 'blue']

categories = OrdinalEncoder(handle_unknown="use_encoded_value",
                           unknown_value=2).fit_transform(penguins_train[[target_column]])

color_indices = [int(cat) for cat in categories]

for k in Cs:
    logistic_regression = make_pipeline(
        StandardScaler(), LogisticRegression(C=k))

    logistic_regression.fit(X=data_train, y=target_train)

    disp = DecisionBoundaryDisplay.from_estimator(
        logistic_regression,
        X=data_train,
        response_method="predict",
        cmap="RdBu",
        alpha=0.5
    )

    disp.ax_.scatter(data_train[culmen_columns[0]],
                     data_train[culmen_columns[1]],
                     c=color_indices)