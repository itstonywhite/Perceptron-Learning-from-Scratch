import numpy as np

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
            weight initialization
            
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



