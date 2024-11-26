# Simple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
# print(X_train)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

plt.scatter(X_train, y_train, color='red')

# Sorting X_train for proper line plotting
plt.plot(np.sort(X_train, axis=0), regressor.predict(np.sort(X_train, axis=0)), color='blue')

# Adding labels and title
plt.title('Salary vs Experience (Training Set)')
plt.xlabel("Years of Experience")
plt.ylabel("Salary")

# Display the plot
plt.show()
