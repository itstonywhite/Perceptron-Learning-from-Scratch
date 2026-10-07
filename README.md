# 🧠 Perceptron Learning Algorithm From Scratch

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Scientific%20Computing-013243.svg)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Processing-150458.svg)](https://pandas.pydata.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557c.svg)](https://matplotlib.org/)

A from-scratch implementation of the **Perceptron Learning Algorithm** using Python and NumPy, designed to demonstrate the core mechanics of a binary linear classifier without relying on a pre-built machine learning implementation.

The model is implemented using an **Object-Oriented Programming (OOP)** approach and trained on a binary subset of the **Iris dataset**. The project also visualizes the dataset, training errors across epochs, and the final decision regions learned by the Perceptron.

---

## 📌 Project Overview

The **Perceptron** is one of the foundational algorithms in machine learning and represents one of the earliest forms of an artificial neural network.

In this project, I implemented the complete Perceptron learning process from scratch, including:

- Random weight initialization
- Net input calculation
- Binary prediction using a unit-step function
- Weight and bias updates
- Training across multiple epochs
- Tracking the number of misclassifications
- Visualization of the learning process
- Visualization of the final decision boundary

The implementation does **not** use `sklearn.linear_model.Perceptron` or any other pre-built Perceptron classifier.

---

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

- \(w\) = Weight vector
- \(b\) = Bias
- \(x\) = Input feature vector
- \(y\) = True class label
- \(\hat{y}\) = Predicted class label
- \(\eta\) = Learning rate

The implementation repeats this process for a predefined number of epochs while recording the number of updates made during each epoch.

---

## 📊 Dataset Description (`iris.txt`)

This project uses the classic **Iris dataset**.

For the Perceptron experiment, the dataset is reduced to a binary classification problem using:

- **Iris-setosa**
- **Iris-versicolor**

The project uses the first **100 samples** from the dataset and selects two features for visualization and classification.

| Feature          | Dataset Column | Description         | Unit |
| :--------------- | :------------- | :------------------ | :--- |
| **Sepal Length** | `0`            | Length of the sepal | cm   |
| **Petal Length** | `2`            | Length of the petal | cm   |

The target variable is converted into binary labels:

| Original Class    | Encoded Label |
| :---------------- | :-----------: |
| `Iris-setosa`     |      `0`      |
| `Iris-versicolor` |      `1`      |

Using only two features also makes it possible to visualize the learned decision boundary in a two-dimensional space.

---

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

---

## ⚙️ Model Configuration

The model in this experiment is initialized with:

```python
ppn = Perceptron(
    eta=0.1,
    n_iters=10
)
```

Where:

- **Learning Rate (`eta`)**: `0.1`
- **Number of Epochs (`n_iters`)**: `10`
- **Random State**: `1` (default)

The implementation stores the training history in:

```python
ppn.errors_
```

which contains the number of parameter updates made during each epoch.

---

## 📈 Visualizations

The project generates three main visualizations to analyze the dataset and the learning process.

### 1. Data Plot

The original two-dimensional training data is plotted using:

- Sepal Length
- Petal Length

![Data Plot](./Plots/Data%20Plot.png)

This visualization provides an initial view of the two classes before training the classifier.

---

### 2. Misclassification Errors

The number of Perceptron updates is plotted for every training epoch.

![Misclassification Errors](./Plots/Misclassification%20Errors.png)

This allows the training process to be observed across epochs and shows how the number of classification updates changes during learning.

---

### 3. Decision Regions

After training, the learned classifier is evaluated over a two-dimensional grid to visualize its decision regions and separating boundary.

![Decision Regions](./Plots/Decision%20Regions.png)

The resulting contour plot shows how the trained Perceptron divides the feature space into the two predicted classes.

---

## 🔬 Machine Learning Workflow

The complete workflow of this project is:

1. **Load the Iris dataset** from `iris.txt`.
2. **Select two classes**: Iris-setosa and Iris-versicolor.
3. **Encode class labels** as `0` and `1`.
4. **Select two input features**: Sepal Length and Petal Length.
5. **Visualize the original dataset**.
6. **Initialize the Perceptron model**.
7. **Train the model from scratch** using the Perceptron learning rule.
8. **Track misclassification updates** for every epoch.
9. **Visualize the training error history**.
10. **Generate and visualize the learned decision regions**.

---

## 📂 Repository Structure

```text
.
├── iris.txt
├── perceptron_learning.py
├── Plots
│   ├── Data Plot.png
│   ├── Decision Regions.png
│   └── Misclassification Errors.png
└── README.md
```

### File Descriptions

| File / Directory                     | Description                                                              |
| :----------------------------------- | :----------------------------------------------------------------------- |
| `iris.txt`                           | Iris dataset used for training                                           |
| `perceptron_learning.py`             | Main Python implementation of the Perceptron and visualization functions |
| `Plots/Data Plot.png`                | Visualization of the selected Iris samples                               |
| `Plots/Misclassification Errors.png` | Misclassification updates across epochs                                  |
| `Plots/Decision Regions.png`         | Learned Perceptron's decision regions                                    |
| `README.md`                          | Project documentation                                                    |

---

## ▶️ How to Run

Clone the repository and run the main Python file:

```bash
python perceptron_learning.py
```

The program will:

- Load the dataset
- Display the original data
- Train the Perceptron
- Display the misclassification error plot
- Display the final decision regions

### Requirements

Install the required Python packages with:

```bash
pip install numpy pandas matplotlib
```

---

## 🎯 Learning Objectives

This project was built to understand the internal mechanics of a fundamental machine learning algorithm rather than simply using an existing implementation.

Through this project, I practiced:

- Implementing a machine learning algorithm from scratch
- Object-Oriented Programming for ML models
- NumPy vector and matrix operations
- Model training and parameter updates
- Binary classification
- Tracking training behavior across epochs
- Decision boundary visualization
- Working with real-world datasets
- Separating model logic from visualization logic

---

## 🚀 Future Improvements

Possible extensions for this project include:

- Adding a train/test split
- Adding an accuracy evaluation method
- Implementing a `score()` method similar to common ML libraries
- Supporting more flexible input validation
- Adding a decision function that returns the raw \(w^Tx+b\) value
- Experimenting with different learning rates and epoch counts
- Extending the implementation toward multiclass classification

---

- [Tony White](https://github.com/itstonywhite) ✍️
