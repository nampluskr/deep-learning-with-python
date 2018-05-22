# Classification of MNIST datasets

Given MNIST datasets,
```python
data, target = x_train, y_train
```

## Numpy
### numpy (1)
- [munual graident calculation using `numpy.ndarray`]
```python
# Setup a model:
w1 = np.random.randn(784, 200)
b1 = np.zeros(200)
w2 = np.random.randn(200, 10)
b2 = np.zeros(10)

# Train the model:
for epoch in range(n_epoch):
    # Forward propagation:
    lin1 = np.dot(data, w1) + b1
    sig1 = sigmoid(lin1)
    lin2 = np.dot(sig1, w2) + b2
    output  = lin2

    # Backward progapation:
    grad_output = (output - target)/target.shape[0]
    grad_lin2 = grad_output
    grad_w2 = np.dot(sig1.T, grad_lin2)
    grad_b2 = np.sum(grad_lin2, axis=0)

    grad_sig1 = np.dot(grad_lin2, w2.T)
    grad_lin1 = sig1*(1 - sig1)*grad_sig1
    grad_w1 = np.dot(data.T, grad_lin1)
    grad_b1 = np.sum(grad_lin1, axis=0)

    # Update model parameters:
    w2 -= lr*grad_w2
    b2 -= lr*grad_b2
    w1 -= lr*grad_w1
    b1 -= lr*grad_b1
```

### numpy (2)
- [manual gradient calculation (class version)]
```python
class NetNumpy:
    def __init__(self):
        self.w1 = np.random.randn(784, 200)
        self.b1 = np.zeros(200)
        self.w2 = np.random.randn(200, 10)
        self.b2 = np.zeros(10)
        
    def forward(self, x):
        self.data = x
        lin1 = np.dot(self.data, self.w1) + self.b1
        self.sig1 = sigmoid(lin1)
        lin2 = np.dot(self.sig1, self.w2) + self.b2
        return lin2

    def backward(self, dout=1):
        grad_output = (self.output - target)/target.shape[0]
        grad_lin2 = grad_output
        self.grad_w2 = np.dot(self.sig1.T, grad_lin2)
        self.grad_b2 = np.sum(grad_lin2, axis=0)

        grad_sig1 = np.dot(grad_lin2, self.w2.T)
        grad_lin1 = self.sig1*(1 - self.sig1)*grad_sig1
        self.grad_w1 = np.dot(self.data.T, grad_lin1)
        self.grad_b1 = np.sum(grad_lin1, axis=0)

    def update(self, lr):
        self.w2 -= lr*self.grad_w2
        self.b2 -= lr*self.grad_b2
        self.w1 -= lr*self.grad_w1
        self.b1 -= lr*self.grad_b1

# Setup a model:
model = NetNumpy()

# Train the model:
for epoch in range(n_epoch):
    output = model.forward(data)
    model.backward()
    model.upadte()
```

