import pandas as pd

adult_census = pd.read_csv("00_data/adult-census-numeric-all.csv")
data, target = adult_census.drop(columns="class"), adult_census["class"]

from sklearn.model_selection import ShuffleSplit

rs = ShuffleSplit(n_splits=10, train_size=0.5, random_state=0)

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pipe = make_pipeline(
    StandardScaler(),
    LogisticRegression()
)

from  sklearn.model_selection import cross_val_score

pipe_scores = pd.Series(cross_val_score(X=data, y=target, cv=rs, estimator=pipe))

from sklearn.dummy import DummyClassifier

# Create a dummy classifier that predicts the most frequent class
dummy_classifier = DummyClassifier(strategy='most_frequent')

# Compute the cross-validation scores
dummy_scores = pd.Series(cross_val_score(dummy_classifier, data, target, cv=rs))

scores = pd.concat([pipe_scores, dummy_scores], axis='columns')
scores.columns = ['logistic', 'dummy']

scores.plot(kind='hist')

def compare_model_vs_dummy_classifier(clf, dummy_str, data, target, cv):
    
    dummy_classifier = DummyClassifier(strategy=dummy_str)
    
    pipe_scores = pd.Series(cross_val_score(estimator=clf, X=data, y=target, 
                                            cv=cv))
    
    dummy_scores = pd.Series(cross_val_score(dummy_classifier, data, target, 
                                             cv=cv))
    
    scores = pd.concat([pipe_scores, dummy_scores], axis='columns')
    scores.columns = ['logistic', 'dummy']
    scores.plot(kind='hist')
    
    return scores

scores = compare_model_vs_dummy_classifier(pipe, 'stratified', data, target, rs)
