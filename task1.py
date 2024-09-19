import numpy as np
import csv
import pandas as pd
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
    #print("Y_one_hot",Y_one_hot, Y_one_hot.shape)
    
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

# Load data and weights
def read_csv_and_split(filename, row_splits, m):
    """
    Reads a CSV file and splits it into multiple numpy arrays based on specified row splits.

    Args:
        filename: The name of the CSV file.
        row_splits: A list of integers indicating the number of rows for each split.

    Returns:
        A list of numpy arrays, each representing a split portion of the CSV data.
    """

    with open(filename, 'r') as file:
        csv_reader = csv.reader(file)

        # Initialize empty lists for each split
        splits = [[] for _ in row_splits]

        # Loop through the CSV rows and distribute them to the appropriate splits
        for row in csv_reader:
            numeric_values = [float(value) for value in row[m:]]  # Convert the rest to float4
            n = len(numeric_values)
            if n == 100:
                splits[0].append(numeric_values)
            elif n == 40:
                splits[1].append(numeric_values)
            elif n == 4:
                splits[2].append(numeric_values)

        # Convert each split to a numpy array
        return [np.array(split) for split in splits]

# Calculate gradients using backpropagation
def calculate_gradients(X, Y, W1, b1, W2, b2, W3, b3):
    # Perform forward pass
    A1, A2, A3 = forward_propagation(X, W1, b1, W2, b2, W3, b3)

    # print("A1",A1.shape)
    # print("A2",A2.shape)
    # print("A3",A3.shape)
    # print("W1",W1.shape)
    # print("W2",W2.shape)
    # print("W3",W3.shape)
    # print("b1",b1.shape)
    # print("b2",b2.shape)
    # print("b3",b3.shape)
    # print("Y",Y.shape)
    # print("X",X.shape)

    
    # Perform backward pass to calculate gradients
    dW1, db1, dW2, db2, dW3, db3 = backpropagation(X, Y, A1, A2, A3, W2, W3)
    
    return dW1, db1, dW2, db2, dW3, db3

# Verify the correctness of the gradients
def verify_gradients(computed_dW, computed_db, true_dW, true_db, tolerance=0.05):
    # Compare computed gradients with true gradients
    w_diff = np.abs(computed_dW - true_dW)
    b_diff = np.abs(computed_db - true_db)

    print("Max difference in weight gradients:", np.max(w_diff))
    print("Max difference in bias gradients:", np.max(b_diff))

    # Check if the differences are within the tolerance
    if np.all(w_diff < tolerance) and np.all(b_diff < tolerance):
        print("Gradients are correct!")
    else:
        print("Gradients are incorrect!")

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


# Saving calculated db and dw values to csv files
def create_csv(computed_dw1, computed_db1, computed_dw2, computed_db2, computed_dw3, computed_db3):
    
    # Specify the CSV file name
    filename1 = 'Task_1/dw.csv'
    filename2 = 'Task_1/db.csv'

    # Write the NumPy array to the CSV file
    with open(filename1, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        for row in computed_dw1:
            csv_writer.writerow(row)
        for row in computed_dw2:
            csv_writer.writerow(row)
        for row in computed_dw3:
            csv_writer.writerow(row)
    
    # Write the NumPy array to the CSV file
    with open(filename2, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        for row in computed_db1:
            csv_writer.writerow(row)
        for row in computed_db2:
            csv_writer.writerow(row)
        for row in computed_db3:
            csv_writer.writerow(row)


### Main function 

def main():
    # Hardcoded data point and label
    X = np.array([[-1, 1, 1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1]])  # Shape: (1, 14)
    Y = np.array([3])  # Class label

    #print("X:",X)
    #print("Y:",Y)

    W1, W2, W3 = read_csv_and_split('Task_1/b/w-100-40-4.csv', [14, 100, 40], 1)
    b1, b2, b3 = read_csv_and_split('Task_1/b/b-100-40-4.csv', [100, 40, 4], 1)

    # print("W1:",W1,W1.shape)
    # print("b1:",b1,b1.shape)
    # print("W2:",W2,W2.shape)
    # print("b2:",b2,b2.shape)
    # print("W3:",W3,W3.shape)
    # print("b3:",b3,b3.shape)
    
    # Step 3: Calculate the gradients using your code
    computed_dW1, computed_db1, computed_dw2, computed_db2, computed_dw3, computed_db3 = calculate_gradients(X, Y, W1, b1, W2, b2, W3, b3)

    # print("computed_dW1:",computed_dW1, computed_dW1.shape)
    # print("computed_db1:",computed_db1, computed_db1.shape)
    # print("computed_dw2:",computed_dw2, computed_dw2.shape)
    # print("computed_db2:",computed_db2, computed_db2.shape)
    # print("computed_dw3:",computed_dw3, computed_dw3.shape)
    # print("computed_db3:",computed_db3, computed_db3.shape)

    create_csv(computed_dW1, computed_db1, computed_dw2, computed_db2, computed_dw3, computed_db3)
    
    # # Step 4: Verify the gradients by comparing them with the true values
    #true_dw1, true_dw2, true_dw3 = read_csv_and_split('Task_1/a/true-dw.csv', [14, 100, 40], 0) 
    #true_db1, true_db2, true_db3 = read_csv_and_split('Task_1/a/true-db.csv', [1, 1, 1], 0)

    # print("true_dw1:", true_dw1.shape)
    # print("true_db1:", true_db1.shape)
    # print("true_dw2:", true_dw2.shape)
    # print("true_db2:", true_db2.shape)
    # print("true_dw3:", true_dw3.shape)
    # print("true_db3:", true_db3.shape)

    # verify_gradients(computed_dW1, computed_db1, true_dw1, true_db1)
    # verify_gradients(computed_dw2, computed_db2, true_dw2, true_db2)
    # verify_gradients(computed_dw3, computed_db3, true_dw3, true_db3)

    ################# Training the neural network #################
    # X_train = pd.read_csv('Task_2/x_train.csv').to_numpy()
    # Y_train = pd.read_csv('Task_2/y_train.csv').to_numpy().flatten()
    # X_test = pd.read_csv('Task_2/x_test.csv').to_numpy()
    # Y_test = pd.read_csv('Task_2/y_test.csv').to_numpy().flatten()

    # # Initialize weights and biases once and reuse them
    # np.random.seed(1)  # Ensure reproducibility
    # W1_init = np.random.randn(14, 100) * 0.01  # First hidden layer has 100 nodes
    # b1_init = np.zeros((1, 100))
    # W2_init = np.random.randn(100, 40) * 0.01  # Second hidden layer has 40 nodes
    # b2_init = np.zeros((1, 40))
    # W3_init = np.random.randn(40, 4) * 0.01  # Output layer
    # b3_init = np.zeros((1, 4))

    # # Train with different learning rates, but always start from the same initial weights
    # learning_rates = [1, 0.1, 0.001]
    # for lr in learning_rates:
    #     print(f"Training with learning rate {lr}")
    #     train_neural_network(X_train, Y_train, X_test, Y_test, W1_init, b1_init, W2_init, b2_init, W3_init, b3_init, iterations=1000, learning_rate=lr)

if __name__ == "__main__":
    main()
