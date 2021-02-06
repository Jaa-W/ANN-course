import numpy as np
import pandas as pd 
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score


iris = load_iris()

x = iris.data[:,(2,3)] # petal length and width
y = (iris.target == 0).astype(np.int) # 1 - Setosa; 0 - another

perceptron = Perceptron(random_state = 42)
perceptron.fit(x,y)

y_pred = perceptron.predict(x)
accuracy = accuracy_score(y, y_pred)
print(accuracy)

print(f"coefficients: {perceptron.coef_} \nintercept: {perceptron.intercept_}")