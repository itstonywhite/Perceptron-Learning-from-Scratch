# 🧠 Perceptron Learning Algorithm From Scratch

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Scientific%20Computing-013243.svg)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Processing-150458.svg)](https://pandas.pydata.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557c.svg)](https://matplotlib.org/)

An implementation of the **Perceptron Learning Algorithm** using Python and NumPy, designed to demonstrate the core mechanics of a binary linear classifier without relying on a pre-built machine learning implementation.

The model is implemented using an **Object-Oriented Programming (OOP)** approach and trained on a binary subset of the **Iris dataset**. The project also visualizes the dataset, training errors across epochs, and the final decision regions learned by the Perceptron.

---

## 📌 Project Overview

The **Perceptron** is one of the foundational algorithms in machine learning and represents one of the earliest forms of an artificial neural network.

In this project, I implemented the complete Perceptron learning process, including:

- Random weight initialization
- Net input calculation
- Binary prediction using a unit-step function
- Weight and bias updates
- Training across multiple epochs
- Tracking the number of misclassifications
- Visualization of the learning process
- Visualization of the final decision boundary

## 📈 Visualizations

The project generates three main visualizations to analyze the dataset and the learning process.

### 1. Data Plot

The original two-dimensional training data is plotted using:

- Sepal Length
- Petal Length

![Data Plot](./Plots/Data%20Plot.png)

### 2. Misclassification Errors

The number of Perceptron updates is plotted for every training epoch.

![Misclassification Errors](./Plots/Misclassification%20Errors.png)

### 3. Decision Regions

After training, the learned classifier is evaluated over a two-dimensional grid to visualize its decision regions and separating boundary.

![Decision Regions](./Plots/Decision%20Regions.png)

The resulting contour plot shows how the trained Perceptron divides the feature space into the two predicted classes.

## 🧠 Perceptron Algorithm

For an input vector \(x\), the Perceptron calculates the net input:

$$
z = w^Tx + b
$$

The predicted class is then determined using a unit-step function:

$$
\hat{y} =
\begin{cases}
1 & z \geq 0 \\
0 & z < 0
\end{cases}
$$

If a sample is misclassified, the model updates its parameters according to the Perceptron learning rule:

$$
w := w + \eta(y-\hat{y})x
$$

$$
b := b + \eta(y-\hat{y})
$$

Where:

- \( w \) = Weight vector
- \( b \) = Bias
- \( x \) = Input feature vector
- \( y \) = True class label
- \( ŷ \) = Predicted class label (y-hat)
- \( η \) = Learning rate (eta)

The implementation repeats this process for a predefined number of epochs while recording the number of updates made during each epoch.

## 🛠️ Implementation

The Perceptron is implemented as an object-oriented classifier:

```python
class Perceptron:
```

### Main Methods

#### `fit(X, y)`

Trains the Perceptron on the provided training data.

During training, it:

1. Initializes the weights and bias.
2. Iterates over the training dataset for multiple epochs.
3. Calculates predictions.
4. Computes the Perceptron update.
5. Updates the weights and bias.
6. Tracks the number of updates for each epoch.

#### `net_input(X)`

Calculates the linear combination:

```python
np.dot(X, self.w_) + self.b_
```

#### `predict(X)`

Applies the unit-step activation function and returns the predicted binary class.

## ▶️ How to Run

### 📦 Requirements

Install the required Python packages with:

```bash
pip install numpy pandas matplotlib
```

Clone the repository:

```bash
git clone https://github.com/itstonywhite/Perceptron-Learning-from-Scratch.git
```

## 📂 Repository Structure

```text
.
├── iris.txt                           # Iris Flower Dataset
├── perceptron_learning.py             # All the Codes
├── Plots                              # Plots Directory
│   ├── Data Plot.png                  # Dataset Plot
│   ├── Decision Regions.png           # Decision Boundary Plot
│   └── Misclassification Errors.png   # Misclassification Errors Plot
└── README.md                          # Documentation
```

---

\- [Tony White](https://github.com/itstonywhite) ✍️
