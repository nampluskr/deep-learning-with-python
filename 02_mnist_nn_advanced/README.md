# Classification of MNIST using convolutional neural nets

- Model: Linear(784,200) - ReLU - Linear(200,200) - ReLU - Linear(200,10) - Softmax - Cross entropy error
  - 2 Hidden layer (nodes = 200, 200)
- Criterion: Cross entropy error
- Activation: relu
- Initialization: (Xavier or He)
- Optimizer: Adam
- Learning rate: 0.001
- Epochs: 10 (batch_size: 64, shuffle)

### numpy
```python
import numpy_nn_lib as nn

# Setup a model:
np.random.seed(0)
layers = [nn.Linear(784, 200, '1', activation='relu'), nn.Relu(),
          nn.Linear(200, 200, '2', activation='relu'), nn.Relu(),
          nn.Linear(200, 10, '3')]
criterion = nn.SoftmaxWithLoss()
model = nn.MultiNetNumpy(layers, criterion)
optimizer = nn.Adam(model, lr=lr)
```

### pytorch
```python
class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
                torch.nn.Linear(784,200), torch.nn.ReLU(),
                torch.nn.Linear(200,200), torch.nn.ReLU(),
                torch.nn.Linear(200,10))

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(-1, 784)
        return self.layers(x)

model = Net()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
```

### tensorflow
```python
# Setup a model:
tf.set_random_seed(0)
lin1 = tf.layers.dense(x, 200, activation=tf.nn.relu)
lin2 = tf.layers.dense(lin1, 200, activation=tf.nn.relu)
lin3 = tf.layers.dense(lin2, 10)
output = lin3

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
           logits=output, labels=y))
optimizer = tf.train.AdamOptimizer(lr).minimize(loss)
```


### keras
```python
# Setup a model:
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(200, activation='relu', input_shape=(784,)))
model.add(Dense(200, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.adam(lr=lr),
              metrics=['accuracy'])
```