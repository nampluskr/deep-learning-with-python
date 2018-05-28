# Autoencoder of MNIST using neural networks

- Model: **Encoder - Decoder - Mean squared error**
  - Encoder: Linear(784, 50) - Relu
  - Decoder: Linear(50, 10) - Sigmoid
- Optimizer: Adam method
- Learning rate: 0.0001
- Epochs: 10 (batch_size: 64, shuffle)

### pytorch
```python
class Autoencoder(torch.nn.Module):
    def __init__(self):
        super(Autoencoder,self).__init__()
        self.encoder = torch.nn.Sequential(
                torch.nn.Linear(28*28, 50),
                torch.nn.ReLU())
        self.decoder = torch.nn.Sequential(
                torch.nn.Linear(50, 28*28),
                torch.nn.Sigmoid())

    def forward(self, x):
        x = x.view(x.size(0), -1)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded.view(x.size(0), 1, 28, 28)


# Set hyper-parameters:
n_epoch, batch_size, lr = 10, 64, 0.0001
shuffle, verbose = True, True

# Load data:
train_loader = DataLoader(tch_nn.MNIST(train=True),
                          batch_size=batch_size, shuffle=shuffle)

# Setup a model:
model = Autoencoder().to(device)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
```