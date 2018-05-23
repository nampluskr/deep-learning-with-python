import torch


# Set hyper-parameters:
n_epoch, lr = 1000, 0.01

# Load data:
n_data = 1000
noise = torch.nn.init.normal_(torch.Tensor(n_data,1), std=1)
x_train = torch.nn.init.uniform_(torch.Tensor(n_data,1), -10, 10)
y_train = 2*x_train + 10 + noise

# Setup a model
torch.manual_seed(0)
w1 = torch.randn(1,1)
b1 = torch.zeros(1)

# Train the model:
for epoch in range(n_epoch):
    # Forward propagation:
    output = torch.mm(x_train, w1) + b1
    loss = torch.mean((output- y_train)**2)

    # Backward propagation:
    grad_output = 2*(output- y_train)/y_train.size(0)
    grad_w1 = torch.mm(x_train.t(), grad_output)
    grad_b1 = torch.sum(grad_output, dim=0)

    # Update model parameters:
    w1 -= lr*grad_w1
    b1 -= lr*grad_b1

# Evaluate the trained model:
print("Epoch[%3d] >> Loss = %f" % (n_epoch, loss))
print(w1, b1)