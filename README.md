# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import Libraries: Import the necessary libraries - pandas, numpy, and matplotlib.pyplot.

2.Load Dataset: Load the dataset using pd.read_csv.

3.Remove irrelevant columns (sl_no, salary).

4.Convert categorical variables to numerical using cat.codes.

5.Separate features (X) and target variable (Y).

6.Define Sigmoid Function: Define the sigmoid function.

7.Define Loss Function: Define the loss function for logistic regression.

8.Define Gradient Descent Function: Implement the gradient descent algorithm to optimize the parameters.

9.Training Model: Initialize theta with random values, then perform gradient descent to minimize the loss and obtain the optimal parameters.

10.Define Prediction Function: Implement a function to predict the output based on the learned parameters.

11.Evaluate Accuracy: Calculate the accuracy of the model on the training data.

12.Predict placement status for a new student with given feature values (xnew).

13.Print Results: Print the predictions and the actual values (Y) for comparison.

<br><br><br><br><br><br>

## Program:
```
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: ROHITH PREM S
RegisterNumber: 212223040172

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset=pd.read_csv("C:/Users/admin/Desktop/Placement_Data.csv")
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
```

## Output:

### Dataset

![Screenshot 2024-04-22 143340](https://github.com/rohithprem18/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/146315115/ffcf0c40-dd74-4f1c-b1dd-ba1ed91c7660)

### dataset.dtypes

![Screenshot 2024-04-22 143349](https://github.com/rohithprem18/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/146315115/977e46e8-e51b-49cc-b79b-4c3d358e167a)

### dataset

![Screenshot 2024-04-22 143355](https://github.com/rohithprem18/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/146315115/53d4729a-21e7-4709-a842-d76cd166a4a2)

### Y

![Screenshot 2024-04-22 143400](https://github.com/rohithprem18/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/146315115/4e236842-cc36-46d8-ba42-5edd381163b8)

#### y_pred

![Screenshot 2024-04-22 143406](https://github.com/rohithprem18/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/146315115/0771f0ec-daec-489c-8080-b92975baa08f)

#### Y

![Screenshot 2024-04-22 143412](https://github.com/rohithprem18/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/146315115/aa6470f8-fe69-42ce-84c2-80edfee53821)

#### y_prednew

![Screenshot 2024-04-22 143417](https://github.com/rohithprem18/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/146315115/446673ff-6430-4d32-8ca2-5d3e99c5c130)

#### y_prednew

![Screenshot 2024-04-22 143421](https://github.com/rohithprem18/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/146315115/ee45f003-3aa4-4a8b-9b55-25d23611d52d)


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

