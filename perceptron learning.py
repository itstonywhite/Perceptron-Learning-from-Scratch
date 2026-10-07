# Importing dependencies & packages
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Object-oriented Perceptron API
class Perceptron:
    """
    Perceptron Classifier.
    
    Parameters:
        1. eta : float
            Learning rate (between 0.0 and 1.0)
        2. n-iters : int
            Passes over the training dataset.
        3. random_state : int
            Random number generator seed for random
            weight initialization.
            
    Attributes:
        1. w_ : 1d-array
            Weights after fitting.
        2. b_ : Scaler
            Bias unit after fitting.
        
        errors_ : list
            Number of misclassifications (updates) in each epoch.
    """
    
    def __init__(self, eta=0.01, n_iters=50, random_state=1):
        self.eta = eta
        self.n_iters = n_iters
        self.random_state = random_state
        
    def fit(self, X, y):
        """
        Fit training data.
        
        Parameters:
            1. X : {array-like}, shape = [n_example, n_features]
                Training vectors, where n_example is the number of
                examples and n_features is the number of features.
            2. y : array-like, shape = [n_examples]
                Target values.
                
        Returns:
            self : object
        """
        
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = np.float_(0.)
        self.errors_ = []
        
        for _ in range(self.n_iters):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_ += update * xi
                self.b_ += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self
    
    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_) + self.b_
    
    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, 0)


# Training a perceptron model on the Iris dataset

# Loading the dataset
df = pd.read_csv('./iris.txt', header=None, encoding='utf-8')

# Select setosa and versicolor
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', 0, 1)

# Extract the sepal length and petal length
X = df.iloc[0:100, [0, 2]].values

# Plot data
plt.scatter(X[0:50, 0], X[0:50, 1], color='red', marker='o', label='Setosa')
plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='s', label='Versicolor')
plt.xlabel('Sepal length [cm]')
plt.ylabel('Petal length [cm]')
plt.legend(loc='upper left')
plt.show() # Renders the plot





