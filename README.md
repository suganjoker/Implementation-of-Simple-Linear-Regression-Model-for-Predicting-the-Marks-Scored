# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Use the standard libraries in python.
2.Set variables for assigning dataset values.
3.Import LinearRegression from the sklearn.
4.Assign the points for representing the graph.
5.Predict the regression for marks by using the representation of graph.
6.Compare the graphs and hence we obtain the LinearRegression for the given datas.
```
## Program:
```py
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: sugavarathanl
RegisterNumber:  212221220051
*/
```
```py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataset = pd.read_csv('student_scores.csv')
dataset.head()
X = dataset.iloc[:,:-1].values
X
Y = dataset.iloc[:,1].values
Y
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 1/3,random_state = 0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred = regressor.predict(X_test)
Y_pred
Y_test
plt.scatter(X_train,Y_train,color="green")
plt.plot(X_train,regressor.predict(X_train),color="purple")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(X_test,Y_test,color="pink")
plt.plot(X_train,regressor.predict(X_train),color="orange") 
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
```
## Output:

![ml ex2](https://github.com/KISHORE7812883161/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/142528124/0653c2dc-bcbb-40d0-8637-4d8ddb996da1)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
