import numpy as np                  # Import NumPy for numerical operations (arrays, math, etc.)
import pandas as pd                 # Import Pandas for handling CSV dataset
from matplotlib import pyplot as plt # Import Matplotlib for plotting (not used much in this code)

# Load dataset (digits dataset from Kaggle MNIST competition)
data = pd.read_csv('./data/train.csv')

# Convert dataframe to numpy array for fast matrix operations
data = np.array(data)

# Get number of rows (m) and columns (n) of dataset
m, n = data.shape

# Shuffle dataset so training and testing data are mixed randomly
np.random.shuffle(data) 

# Prepare development (validation) set - first 1000 samples
data_dev = data[0:1000].T   # Take first 1000 rows and transpose (so each column is a sample)
Y_dev = data_dev[0]         # First row is labels
X_dev = data_dev[1:n]       # Remaining rows are features (pixels)
X_dev = X_dev / 255.        # Normalize pixel values (0–255 → 0–1)

# Prepare training set - remaining samples
data_train = data[1000:m].T # Take rows from 1000 onward and transpose
Y_train = data_train[0]     # First row is labels
X_train = data_train[1:n]   # Remaining rows are features
X_train = X_train / 255.    # Normalize features
_, m_train = X_train.shape  # Get number of training examples

# ---------------- Neural Network Functions ---------------- #

# Initialize weights and biases randomly (small values centered at 0)
def init_params():
    W1 = np.random.rand(10, 784) - 0.5  # Weights for first layer (10 neurons, 784 inputs)
    b1 = np.random.rand(10, 1) - 0.5    # Biases for first layer
    W2 = np.random.rand(10, 10) - 0.5   # Weights for second layer (10 outputs, 10 hidden neurons)
    b2 = np.random.rand(10, 1) - 0.5    # Biases for second layer
    return W1, b1, W2, b2

# ReLU activation function (max(0, x)) - introduces non-linearity
def ReLU(Z):
    return np.maximum(Z, 0)

# Softmax activation - converts scores into probability distribution
def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A
    
# Forward propagation (compute outputs of each layer)
def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1      # Linear combination for first layer
    A1 = ReLU(Z1)            # Apply ReLU activation
    Z2 = W2.dot(A1) + b2     # Linear combination for second layer
    A2 = softmax(Z2)         # Apply softmax for probability outputs
    return Z1, A1, Z2, A2

# Derivative of ReLU (used in backpropagation)
def ReLU_deriv(Z):
    return Z > 0   # Returns 1 where Z>0, else 0

# Convert labels Y into one-hot encoded vectors
def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1)) # Create zeros matrix
    one_hot_Y[np.arange(Y.size), Y] = 1         # Place 1 at correct label position
    one_hot_Y = one_hot_Y.T                     # Transpose to match dimensions
    return one_hot_Y

# Backward propagation (calculate gradients)
def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    one_hot_Y = one_hot(Y)                          # Convert Y to one-hot encoding
    dZ2 = A2 - one_hot_Y                            # Gradient of loss w.r.t Z2
    dW2 = 1 / m * dZ2.dot(A1.T)                     # Gradient for W2
    db2 = 1 / m * np.sum(dZ2)                       # Gradient for b2
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)            # Backprop through ReLU
    dW1 = 1 / m * dZ1.dot(X.T)                      # Gradient for W1
    db1 = 1 / m * np.sum(dZ1)                       # Gradient for b1
    return dW1, db1, dW2, db2

# Update weights and biases using gradients
def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1   # Update W1
    b1 = b1 - alpha * db1   # Update b1
    W2 = W2 - alpha * dW2   # Update W2
    b2 = b2 - alpha * db2   # Update b2
    return W1, b1, W2, b2

# Get predictions (class with highest probability)
def get_predictions(A2):
    return np.argmax(A2, 0)

# Calculate accuracy by comparing predictions with true labels
def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

# Gradient descent training loop
def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2 = init_params()                         # Initialize weights and biases
    for i in range(iterations):                            # Loop through iterations
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)   # Forward propagation
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y) # Backpropagation
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha) # Update params
        if i % 10 == 0:                                    # Every 10 iterations, print progress
            print("Iteration: ", i)
            predictions = get_predictions(A2)
            print(get_accuracy(predictions, Y))
    return W1, b1, W2, b2

# Train the neural network with learning rate 0.10 and 1001 iterations
W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.10, 1001)
