# Classification of MNIST using neural networks

- Model: **Linear(784,200) - Sigmoid - Linear(200,10) - Softmax - Cross entropy error**
- Initialization: Xavier (mean=0, stddev=sqrt(1/n))
- Optimizer: Gradient descent method
- Learning rate: 0.01
- Epochs: 10 (batch_size: 64, shuffle)

Given MNIST datasets,
```python
data, target = x_train, y_train
```

## Numpy
### numpy (1)
- Manual graident calculation using `numpy.ndarray`
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
- Manual gradient calculation (class version)
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

## Pytorch
### pytorch (1)
- Manual gradient calculation using `torch.Tensor`
```python
# Setup a model:
w1 = np.random.randn(784, 200)
b1 = np.zeros(200)
w2 = np.random.randn(200, 10)
b2 = np.zeros(10)

w1 = torch.from_numpy(w1).float()
b1 = torch.from_numpy(b1).float()
w2 = torch.from_numpy(w2).float()
b2 = torch.from_numpy(b2).float()

# Train the model:
for epoch in range(n_epoch):
    # Forward propagation:
    lin1 = torch.mm(data, w1) + b1
    sig1 = F.sigmoid(lin1)
    lin2 = torch.mm(sig1, w2) + b2
    output  = lin2

    # Backward progapation:
    grad_output = (output - onehot(target))/target.size(0)
    grad_lin2 = grad_output
    grad_w2 = torch.mm(sig1.t(), grad_lin2)
    grad_b2 = torch.sum(grad_lin2, 0)

    grad_sig1 = torch.mm(grad_lin2, w2.t())
    grad_lin1 = sig1*(1 - sig1)*grad_sig1
    grad_w1 = torch.mm(data.t(), grad_lin1)
    grad_b1 = torch.sum(grad_lin1, 0)

    # Update model parameters:
    w2 -= lr*grad_w2
    b2 -= lr*grad_b2
    w1 -= lr*grad_w1
    b1 -= lr*grad_b1
```

### pytorch (2)
- Automatic gradient: `torch.Tensor` with `requires_grad=True`
```python
# Setup a model:
w1 = torch.randn(784,200, requires_grad=True)
b1 = torch.zeros(200, requires_grad=True)
w2 = torch.randn(200,10, requires_grad=True)
b2 = torch.zeros(10, requires_grad=True)

# Train the model:
for i in range(n_batch):
    # Forward propagation:
    lin1 = torch.mm(data, w1) + b1
    sig1 = F.sigmoid(lin1)
    lin2 = torch.mm(sig1, w2) + b2
    output  = lin2
    loss = F.cross_entropy(output, target)

    # Backward progapation:
    loss.backward()

    # Update model parameters:
    for param in (w1, b1, w2, b2):
        param.data -= lr*param.grad.data
        param.grad.zero_()
```

### pytorch (3)
- Optimizer and GPU operation
```python
use_gpu = 1
device = torch.device("cuda") if use_gpu else torch.device("cpu")

data, target = data.to(device), target.to(device)

# Setup a model:
model = torch.nn.Sequential(torch.nn.Linear(784,200), torch.nn.Sigmoid(),
                            torch.nn.Linear(200,10)).to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

# Train the model:
for epoch in range(n_epoch):
    # Forwad propagation:
    output = model(data)
    loss = criterion(output, target)
    optimizer.zero_grad()

    # Backward propagation:
    loss.backward()

    # Update model parameters:
    optimizer.step()
```

## Tensorflow
- Setup a model & forward propagation
```python
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

# Setup a model:
w1 = tf.Variable(tf.random_normal([784, 200], stddev=0.1))
b1 = tf.Variable(tf.zeros([200]))
w2 = tf.Variable(tf.random_normal([200, 10], stddev=0.1))
b2 = tf.Variable(tf.zeros([10]))

# Forward propagation:
lin1 = tf.add(tf.matmul(x, w1), b1)
sig1 = tf.nn.sigmoid(lin1)
lin2 = tf.add(tf.matmul(sig1, w2), b2)
output = lin2
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
           logits=output, labels=y))
```

### tensorflow (1)
- Manual gradient calculation using tensors
``` python
# Backward propagation:
grad_output = (output-y)/tf.cast(tf.shape(y)[0], tf.float32)
grad_lin2 = grad_output
grad_w2 = tf.matmul(tf.transpose(sig1), grad_lin2)
grad_b2 = tf.reduce_sum(grad_lin2, 0)

grad_sig1 = tf.matmul(grad_lin2, tf.transpose(w2))
grad_lin1 = sig1*(1-sig1)*grad_sig1
grad_w1 = tf.matmul(tf.transpose(x), grad_lin1)
grad_b1 = tf.reduce_sum(grad_lin1, 0)

# Update model parameters:
update_w1 = w1.assign(w1 - lr*grad_w1)
update_b1 = b1.assign(b1 - lr*grad_b1)
update_w2 = w2.assign(w2 - lr*grad_w2)
update_b2 = b2.assign(b2 - lr*grad_b2)

# Evaluate the model:
correct = tf.equal(tf.argmax(output,1), tf.argmax(y,1))
acc = tf.reduce_mean(tf.cast(correct, tf.float32))

# Train the model
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(n_epoch):
        sess.run([update_w1, update_b1, update_w2, update_b2],
                 feed_dict={x:x_train[batch], y:y_train[batch]})
        loss_, acc_ = sess.run([loss, acc],
                 feed_dict={x:x_train[batch], y:y_train[batch]})
```

### tensorflow (2)
- Automatic gradient calculation with `tf.gradients`
```python
# Backward propagation:
grad_w1, grad_b1, grad_w2, grad_b2 = tf.gradients(loss, [w1, b1, w2, b2])

# Update model parameters:
update_w1 = w1.assign(w1 - lr*grad_w1)
update_b1 = b1.assign(b1 - lr*grad_b1)
update_w2 = w2.assign(w2 - lr*grad_w2)
update_b2 = b2.assign(b2 - lr*grad_b2)

# Evaluate the model:
correct = tf.equal(tf.argmax(output,1), tf.argmax(y,1))
acc = tf.reduce_mean(tf.cast(correct, tf.float32))

# Train the model
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(n_epoch):
        sess.run([update_w1, update_b1, update_w2, update_b2],
                feed_dict={x:x_train[batch], y:y_train[batch]})
        loss_batch, acc_batch  = sess.run([loss, acc],
                feed_dict={x:x_train[batch], y:y_train[batch]})
```

### tensorflow (3)
- Optimizer
```python
optimizer = tf.train.GradientDescentOptimizer(lr).minimize(loss)

# Evaluate the model:
correct = tf.equal(tf.argmax(output,1), tf.argmax(y,1))
acc = tf.reduce_mean(tf.cast(correct, tf.float32))

# Train the model
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(n_epoch):
        sess.run(optimizer, feed_dict={x:x_train[batch], y:y_train[batch]})
        loss_batch, acc_batch  = sess.run([loss, acc],
                feed_dict={x:x_train[batch], y:y_train[batch]})
```
