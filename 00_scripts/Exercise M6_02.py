import pandas as pd
from sklearn.model_selection import train_test_split

penguins = pd.read_csv("00_data/penguins_regression.csv")
feature_name = "Flipper Length (mm)"
target_name = "Body Mass (g)"
data, target = penguins[[feature_name]], penguins[target_name]
data_train, data_test, target_train, target_test = train_test_split(
    data, target, random_state=0)

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Create the random forest
forest = RandomForestRegressor(n_estimators=3, random_state=0)

# Train the random forest on the training set
forest.fit(data_train, target_train)

# Evaluate the generalization performance of the random forest on the testing set
y_pred = forest.predict(data_test)
mae = mean_absolute_error(target_test, y_pred)

print(f"Mean Absolute Error: {mae:.2f}")


import numpy as np
import matplotlib.pyplot as plt

# Create a new dataset containing evenly spaced values over the interval between 170 and 230
X_new = np.linspace(170, 230, num=10).reshape(-1, 1)

# Get the predictions from the individual trees in the forest
predictions = [tree.predict(X_new) for tree in forest.estimators_]

# Plot the predictions from the individual trees
for i in range(0, 3):
    plt.plot(X_new, predictions[i], "ro-", linewidth=1, markersize=10)
    
plt.xlabel("Input feature")
plt.ylabel("Prediction")
plt.title("Predictions from the individual trees in the random forest")
plt.show()

# Get the predictions from the individual trees in the forest
tree_predictions = [tree.predict(X_new) for tree in forest.estimators_]

# Get the predictions from the random forest
forest_predictions = forest.predict(X_new)


# Plot the data as a scatter plot
plt.scatter(data, target, c="b", label="Data")

# Plot the decisions from the individual trees
for tree_prediction in tree_predictions:
    plt.plot(X_new, tree_prediction, "ro-", linewidth=1, markersize=7, alpha=0.5)

# Plot the decision from the random forest
plt.plot(X_new, forest_predictions, "g--", linewidth=2, markersize=12, alpha=0.8, label="Random Forest")

# Add labels and a legend
plt.xlabel("Input feature")
plt.ylabel("Target")
plt.legend(loc="upper left")

# Show the plot
plt.show()
