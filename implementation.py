import numpy as np

# ReLU activation function and its derivative
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

# Softmax function
def softmax(x):
    exps = np.exp(x - np.max(x))  # Subtract max for numerical stability
    return exps / np.sum(exps, axis=1, keepdims=True)

# Cross-entropy loss function
def cross_entropy_loss(y_pred, y_true):
    m = y_true.shape[0]
    log_likelihood = -np.log(y_pred[range(m), y_true])
    return np.sum(log_likelihood) / m

# Accuracy calculation
def accuracy(y_pred, y_true):
    predictions = np.argmax(y_pred, axis=1)
    return np.mean(predictions == y_true)

# Forward propagation function
def forward_propagation(X, W1, b1, W2, b2, W3, b3):
    Z1 = np.dot(X, W1) + b1  # Layer 1 linear transformation
    A1 = relu(Z1)  # ReLU activation after layer 1
    Z2 = np.dot(A1, W2) + b2  # Layer 2 linear transformation
    A2 = relu(Z2)  # ReLU activation after layer 2
    Z3 = np.dot(A2, W3) + b3  # Output layer linear transformation
    A3 = softmax(Z3)  # Softmax to get probabilities from output
    return A1, A2, A3

# Backward propagation function
def backpropagation(X, Y, A1, A2, A3, W2, W3):
    m = X.shape[0]
    
    # One-hot encode the labels
    Y_one_hot = np.zeros((m, 4))
    Y_one_hot[np.arange(m), Y] = 1
    
    # Backpropagation for output layer
    dZ3 = A3 - Y_one_hot
    dW3 = np.dot(A2.T, dZ3) / m
    db3 = np.sum(dZ3, axis=0, keepdims=True) / m
    
    # Backpropagation for second hidden layer
    dZ2 = np.dot(dZ3, W3.T) * relu_derivative(A2)
    dW2 = np.dot(A1.T, dZ2) / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m
    
    # Backpropagation for first hidden layer
    dZ1 = np.dot(dZ2, W2.T) * relu_derivative(A1)
    dW1 = np.dot(X.T, dZ1) / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m
    
    return dW1, db1, dW2, db2, dW3, db3

# Gradient descent optimization
def update_weights(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, learning_rate):
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W3 -= learning_rate * dW3
    b3 -= learning_rate * db3
    return W1, b1, W2, b2, W3, b3

# Training function
def train_neural_network(X_train, Y_train, X_test, Y_test, iterations=1000, learning_rate=0.1):
    np.random.seed(1)
    
    # Initialize weights and biases
    W1 = np.random.randn(14, 100) * 0.01  # First hidden layer has 100 nodes
    b1 = np.zeros((1, 100))
    W2 = np.random.randn(100, 40) * 0.01  # Second hidden layer has 40 nodes
    b2 = np.zeros((1, 40))
    W3 = np.random.randn(40, 4) * 0.01  # Output layer
    b3 = np.zeros((1, 4))
    
    train_costs = []
    test_costs = []
    train_accuracies = []
    test_accuracies = []
    
    for i in range(iterations):
        # Forward propagation on training data
        A1_train, A2_train, A3_train = forward_propagation(X_train, W1, b1, W2, b2, W3, b3)
        train_cost = cross_entropy_loss(A3_train, Y_train)
        
        # Backward propagation to compute gradients
        dW1, db1, dW2, db2, dW3, db3 = backpropagation(X_train, Y_train, A1_train, A2_train, A3_train, W2, W3)
        
        # Update weights using gradient descent
        W1, b1, W2, b2, W3, b3 = update_weights(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, learning_rate)
        
        # Forward propagation on test data
        _, _, A3_test = forward_propagation(X_test, W1, b1, W2, b2, W3, b3)
        test_cost = cross_entropy_loss(A3_test, Y_test)
        
        # Track the cost and accuracy
        train_costs.append(train_cost)
        test_costs.append(test_cost)
        train_accuracies.append(accuracy(A3_train, Y_train))
        test_accuracies.append(accuracy(A3_test, Y_test))
        
        # Print cost and accuracy every 100 iterations
        if i % 100 == 0:
            print(f"Iteration {i} | Training cost: {train_cost} | Test cost: {test_cost}")
    
    return train_costs, test_costs, train_accuracies, test_accuracies

# Sample code to train the network
# Assuming X_train, Y_train, X_test, Y_test are properly loaded numpy arrays
# Each row in X_train is a 14-dimensional input, and each Y_train entry is a class label (0-3)

# Example placeholder data
X_train = np.random.randn(500, 14)  # 500 examples, 14 input features
Y_train = np.random.randint(0, 4, 500)  # 500 labels with 4 classes
X_test = np.random.randn(100, 14)  # 100 test examples
Y_test = np.random.randint(0, 4, 100)  # 100 test labels

# Train the neural network
train_costs, test_costs, train_accuracies, test_accuracies = train_neural_network(X_train, Y_train, X_test, Y_test, iterations=1000, learning_rate=0.1)

# You can plot the costs and accuracies here using matplotlib, if needed.
