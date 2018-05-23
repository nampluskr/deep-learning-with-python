import numpy as np


# Set hyper-parameters:
n_epoch, lr = 1000, 0.01

# Load data:
n_data = 1000
noise = np.random.normal(0, 1, (n_data, 1))
x_train = np.random.uniform(-10, 10, (n_data,1))
y_train = x_train*2 + 10 + noise

# Setup a model
np.random.seed(0)
w1 = np.random.randn(1,1)
b1 = np.zeros(1)

# Train the model:
for epoch in range(n_epoch):
    # Forward propagation:
    output = np.dot(x_train, w1) + b1
    loss = np.mean((output - y_train)**2)

    # Backward propagation:
    grad_output = 2*(output - y_train)/y_train.shape[0]
    grad_w1 = np.dot(x_train.T, grad_output)
    grad_b1 = np.sum(grad_output, axis=0)

    # Update model parameters:
    w1 -= lr*grad_w1
    b1 -= lr*grad_b1

# Evaluate the trained model:
print("Epoch[%3d] >> Loss = %f" % (n_epoch, loss))
print(w1, b1)