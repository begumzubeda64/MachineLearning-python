import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error

# 1. Linear Regression Method
# (['data', 'target', 'frame', 'DESCR', 'feature_names', 'data_filename', 'target_filename'])
diabetes = datasets.load_diabetes()

# print(diabetes.keys())
# (['data', 'target', 'frame', 'DESCR', 'feature_names', 'data_filename', 'target_filename'])
# diabetes_X = diabetes.data[:, np.newaxis, 2] #One feature

diabetes_X = diabetes.data
diabetes_X_train = diabetes_X[:-30] #first 30
diabetes_X_test = diabetes_X[-30:] #last 30

diabetes_Y_train = diabetes.target[:-30]
diabetes_Y_test = diabetes.target[-30:]

# diabetes_X = np.array([[1], [2], [3]])
# diabetes_X_train = diabetes_X
# diabetes_X_test = diabetes_X
#
# diabetes_Y_train = np.array([3, 2, 4])
# diabetes_Y_test = np.array([3, 2, 4])

model = linear_model.LinearRegression()
model.fit(diabetes_X_train, diabetes_Y_train)

diabetes_Y_predict = model.predict(diabetes_X_test)
print("Mean squared error: ", mean_squared_error(diabetes_Y_test, diabetes_Y_predict))
print("Weights: ", model.coef_)
print("Intercept: ", model.intercept_)

# one feature
# plt.scatter(diabetes_X_test, diabetes_Y_test)
# plt.plot(diabetes_X_test, diabetes_Y_predict)
# plt.show()

# Output: One feature
# Mean squared error:  3035.0601152912695
# Weights:  [941.43097333]
# Intercept:  153.39713623331698

