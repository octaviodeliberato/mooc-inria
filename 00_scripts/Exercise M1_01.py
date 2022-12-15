import pandas as pd

penguins = pd.read_csv("00_data/penguins_classification.csv")

penguins

penguins.info()

penguins['Species'].value_counts()

penguins.columns

penguins['Culmen Length (mm)'].hist()

penguins['Culmen Depth (mm)'].hist()

import seaborn as sns

pair_plt = sns.pairplot(penguins, hue="Species")
