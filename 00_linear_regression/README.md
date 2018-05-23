# Linear regression

- Model: Linear(y = w x + b) - Mean squared error
- Learning rate: 0.01
- Epochs: 1000

### numpy
- Manual graident calculation using `numpy.ndarray`
```python
import numpy as np

# Setup a model
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
```

### pytorch (1)
- Manual gradient calculation using `torch.Tensor`
```python
import torch

# Setup a model
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
```

### pytorch (2)
- `torch.nn` and `torch.optim`
```python
import torch

# Setup a model
model = torch.nn.Linear(1,1)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

# Train the model:
for epoch in range(n_epoch):
    # Forward propagation:
    output = model(x_train)
    loss = criterion(output, y_train)
   
    optimizer.zero_grad()

    # Backward propagation#
    loss.backward()
    
    # update model parameters:
    optimizer.step()
```

### tensorflow
```python
import tensorflow as tf

x = tf.placeholder(tf.float32, [None,1])
y = tf.placeholder(tf.float32, [None,1])

# Setup a model
w1 = tf.Variable(tf.random_normal([1,1]))
b1 = tf.Variable(tf.random_normal([1]))

output = tf.add(tf.matmul(x, w1), b1)
loss = tf.reduce_mean(tf.square(output - y))

grad_y = 2*(output - y)/tf.cast(tf.shape(y)[0], tf.float32)
grad_w1 = tf.matmul(tf.transpose(x), grad_y)
grad_b1 = tf.reduce_sum(grad_y, 0)

update_w1 = w1.assign(w1 - lr*grad_w1)
update_b1 = b1.assign(b1 - lr*grad_b1)

# Train the model:
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(n_epoch):
        sess.run([update_w1, update_b1], feed_dict={x:x_train, y:y_train})
        loss_ = sess.run(loss, feed_dict={x:x_train, y:y_train})
```

### keras
```python
import keras

# Setup a model
model = keras.models.Sequential()
model.add(keras.layers.Dense(1, input_dim=1))
model.compile(loss='mse', optimizer=keras.optimizers.SGD(lr=lr))

# Train the model:
history = model.fit(x_train, y_train, epochs=n_epoch, verbose=0)

# Evaluate the trained model:
w1, b1 = model.get_weights()
loss = model.evaluate(x_train, y_train, verbose=0)
```
