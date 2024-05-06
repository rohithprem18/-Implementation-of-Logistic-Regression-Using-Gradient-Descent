# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
step 1.Start

step 2.Import Libraries: Import the necessary libraries - pandas, numpy, and matplotlib.pyplot.

step 3.Load Dataset: Load the dataset using pd.read_csv.

step 4.Remove irrelevant columns (sl_no, salary).

step 5.Convert categorical variables to numerical using cat.codes.

step 6.Separate features (X) and target variable (Y).

step 7.Define Sigmoid Function: Define the sigmoid function.

step 8.Define Loss Function: Define the loss function for logistic regression.

step 9.Define Gradient Descent Function: Implement the gradient descent algorithm to optimize the parameters.

step 10.Training Model: Initialize theta with random values, then perform gradient descent to minimize the loss and obtain the optimal parameters.

step 11.Define Prediction Function: Implement a function to predict the output based on the learned parameters.

step 12.Evaluate Accuracy: Calculate the accuracy of the model on the training data.

step 13.Predict placement status for a new student with given feature values (xnew).

step 14.Print Results: Print the predictions and the actual values (Y) for comparison.

step 15.Stop.
## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: ROHITH PREM S
RegisterNumber:  212223040172

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset=pd.read_csv("C:\sem-1\Placement_Data.csv")
dataset
dataset=dataset.drop('sl_no',axis=1)
dataset=dataset.drop("salary",axis=1)
dataset ["gender"] = dataset ["gender"].astype('category')
dataset["ssc_b"] = dataset["ssc_b"].astype('category')
dataset["hsc_b"] = dataset ["hsc_b"].astype('category')
dataset ["degree_t"] = dataset ["degree_t"].astype('category')
dataset ["workex"] = dataset ["workex"].astype('category')
dataset["specialisation"] = dataset ["specialisation"].astype('category')
dataset ["status"] = dataset["status"].astype('category')
dataset ["hsc_s"] = dataset ["hsc_s"].astype('category')
dataset.dtypes
dataset ["gender"] = dataset ["gender"].cat.codes
dataset ["ssc_b"] = dataset["ssc_b"].cat.codes
dataset ["hsc_b"] = dataset ["hsc_b"].cat.codes
dataset ["degree_t"] = dataset["degree_t"].cat.codes
dataset["workex"] = dataset["workex"].cat.codes
dataset["specialisation"] = dataset["specialisation"].cat.codes
dataset["status"] = dataset ["status"].cat.codes
dataset["hsc_s"] = dataset["hsc_s"].cat.codes
dataset
X=dataset.iloc[:, :-1].values
Y=dataset.iloc[:, -1].values
Y
theta=np.random.randn(X.shape[1])
y=Y
def sigmoid(z):
    return 1 / (1+np.exp(-z))
def loss(theta,X,y):
    h=sigmoid(X.dot(theta))
    return -np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
def gradient_descent (theta, X, y, alpha, num_iterations):
    m = len(y)
    for i in range(num_iterations):
        h = sigmoid(X.dot(theta))
        gradient = X.T.dot(h-y) / m
        theta -= alpha * gradient
    return theta
theta =  gradient_descent(theta, X, y, alpha=0.01, num_iterations=1000)
```
```
def predict(theta, X): 
    h = sigmoid(X.dot(theta))
    y_pred = np.where(h >= 0.5, 1, 0)
    return y_pred
y_pred = predict(theta, X)
accuracy = np.mean(y_pred.flatten()==y)
print("Accuracy:", accuracy)
print(y_pred)
print(Y)
xnew = np.array([[0, 87, 0, 95, 0, 2, 78, 2, 0, 0, 1, 0]]) 
y_prednew = predict(theta, xnew) 
print(y_prednew)
xnew = np.array([[0, 0, 0, 0, 0, 2, 8, 2, 0, 0, 1, 0]]) 
y_prednew = predict(theta, xnew) 
print(y_prednew)
*/
```
## Output:
### dataset:
![Screenshot 2024-04-22 143407](https://github.com/Aadithya2201/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/145917810/b759be5c-9ff7-44ab-ba18-0f95fef59a0a)
### dataset.dtypes:
![Screenshot 2024-04-22 143421](https://github.com/Aadithya2201/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/145917810/0ac77d57-1afe-49a7-aa8d-4414f90e9e92)
### dataset:
![Screenshot 2024-04-22 143450](https://github.com/Aadithya2201/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/145917810/730e6d8a-e533-43b2-9d7a-71428c7ab09a)
### Y:
![Screenshot 2024-04-22 143458](https://github.com/Aadithya2201/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/145917810/c5a2dc74-8f83-46f5-8c99-ca4fc9fba3c3)
### y_pred:
![Screenshot 2024-04-22 143507](https://github.com/Aadithya2201/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/145917810/6dce3b27-0063-401d-b7e4-5f8dfb501496)
### Y:
![Screenshot 2024-04-22 143511](https://github.com/Aadithya2201/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/145917810/cbc84213-853e-4afc-b9f9-20a48dcd8903)
### y_prednew:
![Screenshot 2024-04-22 143518](https://github.com/Aadithya2201/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/145917810/fcfeba87-3f2f-4d1d-b625-b63326f7da16)
```
```
## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

