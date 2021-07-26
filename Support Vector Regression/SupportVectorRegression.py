import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

y = y.reshape(len(y), 1)

sc_x = StandardScaler()
x = sc_x.fit_transform(x)
sc_y = StandardScaler()
y = sc_x.fit_transform(y)

regressor = SVR(kernel = 'rbf')
regressor.fit(x, y)

sc_y.inverse_transform(regressor.predict(sc_x.transform([[6.5]])))
