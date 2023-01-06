import numpy as np
# Set the seed for reproduction
rng = np.random.RandomState(0)

# Generate data
n_sample = 100
data_max, data_min = 1.4, -1.4
len_data = (data_max - data_min)

data = rng.rand(n_sample) * len_data - len_data / 2
noise = rng.randn(n_sample) * .3
target = data ** 3 - 0.5 * data ** 2 + noise

import pandas as pd
full_data = pd.DataFrame({"data": data, "target": target})

import seaborn as sns

_ = sns.scatterplot(data=full_data, x="data", y="target", color="black",
                    alpha=0.5)

def f(data, weight=0, intercept=0):
    target_predict = weight * data + intercept
    return target_predict

from sklearn.linear_model import LinearRegression

lm = LinearRegression()

lm.fit(X=data.reshape(-1, 1), y=target.reshape(-1, 1))

a = lm.coef_
b = lm.intercept_

y_hat = f(data=data, intercept=b, weight=a)

pred_df = pd.DataFrame({'actual': target, 'pred': y_hat.reshape(n_sample,)})

_ = sns.scatterplot(data=pred_df, x="actual", y="pred", color="black",
                    alpha=0.5)

np.sum((y_hat.reshape(n_sample,) - target)**2)

from sklearn.tree import DecisionTreeRegressor

tree = DecisionTreeRegressor(max_depth=3).fit(
    data.reshape(-1, 1), target.reshape(-1, 1)
)
target_predicted = tree.predict(data.reshape(-1, 1))

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(target, target_predicted)
mse

pred_df = pd.DataFrame({'actual': target, 'pred': target_predicted})

_ = sns.scatterplot(data=pred_df, x="actual", y="pred", color="black",
                    alpha=0.5)
_ = ax.set_title(f"Mean squared error = {mse:.2f}")

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

polynomial_regression = make_pipeline(
    PolynomialFeatures(degree=3, include_bias=False),
    LinearRegression(),
)

polynomial_regression.fit(data.reshape(-1, 1), target.reshape(-1, 1))
target_predicted = polynomial_regression.predict(data.reshape(-1, 1))
mse = mean_squared_error(target, target_predicted)

from sklearn.svm import SVR

svr = SVR(kernel="linear")
svr.fit(data.reshape(-1, 1), target.reshape(-1, 1))
target_predicted = svr.predict(data.reshape(-1, 1))
mse = mean_squared_error(target, target_predicted)

svr = SVR(kernel="poly", degree=3)
svr.fit(data.reshape(-1, 1), target.reshape(-1, 1))
target_predicted = svr.predict(data.reshape(-1, 1))
mse = mean_squared_error(target, target_predicted)
mse
