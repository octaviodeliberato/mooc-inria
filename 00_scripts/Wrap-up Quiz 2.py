# Load your data and split it into X and y
import pandas as pd

blood_transfusion = pd.read_csv("00_data/blood_transfusion.csv")
target_name = "Class"
data = blood_transfusion.drop(columns=target_name)
target = blood_transfusion[target_name]

from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_val_score

# Create a DummyClassifier with the "most_frequent" strategy
clf = DummyClassifier(strategy="most_frequent")

# Calculate the accuracy scores for each fold of the cross-validation
scores = cross_val_score(clf, data, target, cv=10, scoring="balanced_accuracy")

# Calculate the average accuracy
average_accuracy = scores.mean()

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_validate

# Create the pipeline object
pipeline = make_pipeline(
    StandardScaler(),  # Scale the data
    KNeighborsClassifier(n_neighbors=1)  # Fit a KNeighborsClassifier
)

# Use the pipeline to fit and predict on your data
cv_results = cross_validate(pipeline, data, target, cv=10, 
                            scoring="balanced_accuracy",
                            return_train_score=True)
cv_results

scores = cv_results["test_score"]
print(f"The accuracy is: {scores.mean():.3f} ± {scores.std():.3f}")

pipe_fit = pipeline.fit(X=data, y=target)

params = pipe_fit.get_params()
print(params)

param_range = [1, 2, 5, 10, 20, 50, 100, 200, 500]
cv_results = []

for n in param_range:
    
    pipeline = make_pipeline(
        StandardScaler(),  # Scale the data
        KNeighborsClassifier(n_neighbors=n)  # Fit a KNeighborsClassifier
    )
    
    res = cross_validate(
        pipeline, data, target, cv=5, 
        scoring="balanced_accuracy",
        return_train_score=True
    )
    
    cv_results.append(res)

    res_df = pd.DataFrame(res)
    print(res_df[["train_score", "test_score"]].mean())