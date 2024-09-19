import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt

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

# Update weights using gradient descent
def update_weights(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, learning_rate):
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W3 -= learning_rate * dW3
    b3 -= learning_rate * db3
    return W1, b1, W2, b2, W3, b3

# Accuracy function to calculate the percentage of correct predictions
def accuracy(y_pred, y_true):
    # Convert the predicted probabilities into class labels by taking the class with the highest probability
    y_pred_labels = np.argmax(y_pred, axis=1)
    
    # Calculate the percentage of correct predictions
    accuracy = np.mean(y_pred_labels == y_true) * 100  # Accuracy as a percentage
    
    return accuracy

# Train the neural network
def train_neural_network(X_train, Y_train, X_test, Y_test, W1_init, b1_init, W2_init, b2_init, W3_init, b3_init, iterations=1000, learning_rate=0.1):
    
    # Initialize weights and biases with the provided initial values
    W1 = W1_init.copy()
    b1 = b1_init.copy()
    W2 = W2_init.copy()
    b2 = b2_init.copy()
    W3 = W3_init.copy()
    b3 = b3_init.copy()

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

    # Plot train cost w.r.t. iterations
    plt.figure()
    plt.title(f'Train Cost w.r.t. Iterations (lr={learning_rate})')
    plt.plot(range(iterations), train_costs)
    plt.xlabel('Iterations')
    plt.ylabel('Training Cost')
    plt.savefig(f'Task_2/train_costs_lr_{learning_rate}.png')

    # Plot test cost w.r.t. iterations
    plt.figure()
    plt.title(f'Test Cost w.r.t. Iterations (lr={learning_rate})')
    plt.plot(range(iterations), test_costs)
    plt.xlabel('Iterations')
    plt.ylabel('Test Cost')
    plt.savefig(f'Task_2/test_costs_lr_{learning_rate}.png')    

    # Plot train and test accuracies
    plt.figure()
    plt.title(f'Train and Test Accuracies w.r.t. Iterations (lr={learning_rate})')
    plt.plot(range(iterations), train_accuracies, label='Training accuracy')
    plt.plot(range(iterations), test_accuracies, label='Test accuracy')
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f'Task_2/accuracies_lr_{learning_rate}.png') 
           

### Main function to run the verification

def main():
    X_train = pd.read_csv('Task_2/x_train.csv').to_numpy()
    Y_train = pd.read_csv('Task_2/y_train.csv').to_numpy().flatten()
    X_test = pd.read_csv('Task_2/x_test.csv').to_numpy()
    Y_test = pd.read_csv('Task_2/y_test.csv').to_numpy().flatten()

    # Initialize weights and biases once and reuse them
    np.random.seed(1)  # Ensure reproducibility
    W1_init = np.random.randn(14, 100) * 0.01  # First hidden layer has 100 nodes
    b1_init = np.zeros((1, 100))
    W2_init = np.random.randn(100, 40) * 0.01  # Second hidden layer has 40 nodes
    b2_init = np.zeros((1, 40))
    W3_init = np.random.randn(40, 4) * 0.01  # Output layer
    b3_init = np.zeros((1, 4))

    # Train with different learning rates, but always start from the same initial weights
    learning_rates = [1, 0.1, 0.001]
    for lr in learning_rates:
        print(f"Training with learning rate {lr}")
        train_neural_network(X_train, Y_train, X_test, Y_test, W1_init, b1_init, W2_init, b2_init, W3_init, b3_init, iterations=1000, learning_rate=lr)

if __name__ == "__main__":
    main()
