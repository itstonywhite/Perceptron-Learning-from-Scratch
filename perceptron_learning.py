# Importing dependencies & packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


# Perceptron API
class Perceptron:
    """
    Perceptron Classifier.
    
    Parameters:
        1. eta : float
            Learning rate (between 0.0 and 1.0)
        2. n_iters : int
            Passes over the training dataset.
        3. random_state : int
            Random number generator seed for random
            weight initialization.
            
    Attributes:
        1. w_ : 1d-array
            Weights after fitting.
        2. b_ : Scalar
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
            1. X : {array-like}, shape = [n_examples, n_features]
                Training vectors, where n_examples is the number of
                examples and n_features is the number of features.
            2. y : array-like, shape = [n_examples]
                Target values.
                
        Returns:
            self : object
        """
        
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = np.float64(0.)
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


# visualizing the decision boundaries for two-dimensional datasets
def plot_decision_regions(X, y, classifier, resolution=0.02):
    """
    Plot the decision regions of a classifier for 2D data.

    Parameters:
        1. X : array-like, shape = [n_examples, 2]
            Feature matrix. Must have exactly two columns, since
            the plot is two-dimensional.
        2. y : array-like, shape = [n_examples]
            Class labels. At most 5 distinct classes.
        3. classifier : object
            Fitted model with a predict(X) method that accepts an
            array of shape [n_points, 2] and returns class labels.
        4. resolution : float, optional (default=0.02)
            Grid step size in feature units. Smaller values give a
            smoother boundary but need more predict calls.

    Returns:
        None
    """
    
    # Setup marker generator and color map
    markers = ('o', 's', '^', 'v', '<')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    # Plotting the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    lab = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    lab = lab.reshape(xx1.shape)
    plt.contourf(xx1, xx2, lab, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    # Plot class labels
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=f'Class {cl}',
                    edgecolors='black')

def main():
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
    plt.title('Data Plot')
    plt.xlabel('Sepal length [cm]')
    plt.ylabel('Petal length [cm]')
    plt.legend(loc='upper left')
    plt.show() # Renders the plot

    # Training perceptron algorithm on the Iris data subset
    ppn = Perceptron(eta=0.1, n_iters=10)
    ppn.fit(X, y)

    # Plotting the misclassification error for each epoch
    plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
    plt.title('Misclassification Errors')
    plt.xlabel('Epochs')
    plt.ylabel('Number of updates')
    plt.show() # Renders the plot

    # Perceptron’s Decision Regions Contour plot
    plot_decision_regions(X, y, classifier=ppn)
    plt.title('Perceptron’s Decision Regions')
    plt.xlabel('Sepal length [cm]')
    plt.ylabel('Petal length [cm]')
    plt.legend(loc='upper left')
    plt.show() # Renders the plot


if __name__ == "__main__":
    main()


# Tony White ✍️
