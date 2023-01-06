from sklearn.datasets import make_regression

data, target, coef = make_regression(
    n_samples=2000,
    n_features=5,
    n_informative=2,
    shuffle=False,
    coef=True,
    random_state=0,
    noise=30,
)

import pandas as pd

feature_names = [
    "Relevant feature #0",
    "Relevant feature #1",
    "Noisy feature #0",
    "Noisy feature #1",
    "Noisy feature #2",
]

coef = pd.Series(coef, index=feature_names)
coef.plot.barh()
coef

from sklearn.linear_model import LinearRegression

lm = LinearRegression()

lm.fit(X=data, y=target)

lm.coef_

data_df = pd.DataFrame(data)
target_df = pd.DataFrame(target)
data_full = pd.concat([data_df, target_df], axis=1)
data_full.columns = feature_names + ['target']
data_full['Relevant feature #00'] = data_full['Relevant feature #0']
data_full['Relevant feature #01'] = data_full['Relevant feature #0']
data_full['Relevant feature #10'] = data_full['Relevant feature #1']
data_full['Relevant feature #11'] = data_full['Relevant feature #1']

data_full = data_full.reindex(columns = [col for col in data_full.columns if col != 'target'] + ['target'])
data_full

data = data_full.drop(columns='target')
target = data_full['target']

lm = LinearRegression()

lm.fit(X=data, y=target)

lm.coef_

from sklearn.linear_model import RidgeCV

lm = RidgeCV(alphas=[0.1, 1, 10, 100])

lm.fit(X=data, y=target)

lm.coef_