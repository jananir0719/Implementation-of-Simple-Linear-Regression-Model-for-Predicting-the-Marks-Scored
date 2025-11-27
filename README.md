# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: JANANI R
RegisterNumber:  25018734
*/
# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Step 1: Input data (Hours studied)
X = np.array([[1], [2], [3], [4], [5]])   # independent variable (hours)
y = np.array([20, 40, 60, 80, 100])       # dependent variable (marks)

# Step 2: Create the model
model = LinearRegression()

# Step 3: Train the model
model.fit(X, y)

# Step 4: Predict marks for a new input
hours = np.array([[6]])   # predict for 6 hours studied
predicted_marks = model.predict(hours)

print("Predicted Marks for 6 study hours:", predicted_marks[0])

# Step 5: Plot the data and regression line
plt.scatter(X, y, color='blue', label='Actual Marks')
plt.plot(X, model.predict(X), color='red', label='Regression Line')
plt.xlabel('Hours Studied')
plt.ylabel('Marks Scored')
plt.title('Linear Regression - Marks vs Hours')
plt.legend()
plt.show()
```

## Output:
<img width="1000" height="744" alt="Screenshot 2025-11-27 081453" src="https://github.com/user-attachments/assets/38beff89-b371-4f69-96bb-235e4d0c4f8b" />



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
