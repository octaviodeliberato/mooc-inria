import pandas as pd

penguins = pd.read_csv("00_data/penguins_classification.csv")
culmen_columns = ["Culmen Length (mm)", "Culmen Depth (mm)"]
target_column = "Species"

from sklearn.model_selection import train_test_split
data, target = penguins[culmen_columns], penguins[target_column]
data_train, data_test, target_train, target_test = train_test_split(
    data, target, random_state=0
)

from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier

# Create the classifier
clf = DecisionTreeClassifier(max_depth=2)

# Train the classifier using the fit method
clf.fit(data_train, target_train)

from sklearn.inspection import DecisionBoundaryDisplay

colors = ['red', 'blue']

y_train = target_train.to_numpy().reshape(-1, 1)

categories = OrdinalEncoder(
    handle_unknown="use_encoded_value",
    unknown_value=4).fit_transform(y_train)

color_indices = [int(cat) for cat in categories]


# Plot the decision boundary
dbd = DecisionBoundaryDisplay.from_estimator(
    clf,
    X=data_train,
    response_method="predict",
    cmap="RdBu",
    alpha=0.5
)

dbd.ax_.scatter(
    data_train[culmen_columns[0]],
    data_train[culmen_columns[1]],
    c=color_indices
)

# Write your code here.
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Plot the decision tree
fig, ax = plt.subplots(figsize=(20, 20))
plot_tree(clf, feature_names=culmen_columns, ax=ax)

# Show the plot
plt.show()

# Test the classifier on the test data
accuracy = clf.score(data_test, target_test)

print(
    f"The test accuracy score of the decision tree is: "
    f"{accuracy:.2f}"
)
