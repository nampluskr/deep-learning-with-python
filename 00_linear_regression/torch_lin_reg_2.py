import torch

# Set hyper-parameters:
n_epoch, lr = 1000, 0.01

# Load data:
torch.manual_seed(0)
n_data = 1000
noise = torch.nn.init.normal_(torch.Tensor(n_data,1), std=1)
x_train = torch.nn.init.uniform_(torch.Tensor(n_data,1), -10, 10)
y_train = 2*x_train + 10 + noise

# Setup a model
model = torch.nn.Linear(1,1)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

# Train the model:
for epoch in range(n_epoch):
    # Forward propagation:
    output = model(x_train)
    loss = criterion(output, y_train)

    # Backward propagation and update model parameters:
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Evaluate the trained model:
print("Epoch[%3d] >> Loss = %f" % (n_epoch, loss))
print(list(model.parameters()))