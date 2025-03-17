import iris_dataset_generation as iris
import numpy as np

# Activation functions
def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

# Loss function
def compute_loss(output, target):
    return np.mean((output - target) ** 2)

# Forward pass
def forward_pass(network, inputs):
    activations = [inputs]
    for layer in network:
        inputs = relu(np.dot(inputs, layer['weights']) + layer['biases'])
        activations.append(inputs)
    return activations

# Backward pass
def backward_pass(network, activations, target):
    deltas = []
    loss_grad = activations[-1] - target
    delta = loss_grad * relu_derivative(activations[-1])
    deltas.append(delta)

    for i in reversed(range(len(network) - 1)):
        delta = np.dot(delta, network[i + 1]['weights'].T) * relu_derivative(activations[i + 1])
        deltas.append(delta)

    deltas.reverse()
    return deltas

# Update weights and biases
def update_weights(network, activations, deltas, learning_rate):
    for i in range(len(network)):
        network[i]['weights'] -= learning_rate * np.dot(activations[i].T, deltas[i])
        network[i]['biases'] -= learning_rate * np.sum(deltas[i], axis=0, keepdims=True)

# Training function
def train_network(network, data, epochs, learning_rate, early_stopping_threshold=1e-5):
    previous_loss = float('inf')
    for epoch in range(epochs):
        total_loss = 0
        for inputs, target in data:
            inputs = np.array(inputs).reshape(1, -1)
            target = np.array(target).reshape(1, -1)

            activations = forward_pass(network, inputs)
            total_loss += compute_loss(activations[-1], target)
            deltas = backward_pass(network, activations, target)
            update_weights(network, activations, deltas, learning_rate)

        total_loss /= len(data)

        if abs(previous_loss - total_loss) < early_stopping_threshold:
            print(f"Early stopping at epoch {epoch} with loss {total_loss}")
            break

        previous_loss = total_loss
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss}")

# Load the Iris dataset
practice_data = iris.generate_iris_dataset()

input_size = len(practice_data[0][0])
output_size = len(practice_data[0][1])
structure = [input_size, 10, 10, output_size]

# Initialize the network
network = []
for i in range(1, len(structure)):
    layer = {
        'weights': np.random.randn(structure[i-1], structure[i]) * np.sqrt(2. / structure[i-1]),
        'biases': np.zeros((1, structure[i]))
    }
    network.append(layer)

# Train the network
train_network(network, practice_data, epochs=1000, learning_rate=0.01)

# Test the network
for inputs, target in practice_data:
    inputs = np.array(inputs).reshape(1, -1)
    target = np.array(target).reshape(1, -1)
    activations = forward_pass(network, inputs)
    print(f"Input: {inputs}, Predicted: {activations[-1]}, Target: {target}")