# Classification of MNIST using convolutional neural nets

- Model: Layer1 - Layer2 - Layer3
  - Layer1: Convolution(N x 32 x 28 x 28) - Relu - Max Pooling(N x 32 x 14 x 14) - Dropout(0.5)
  - Layer2: Convolution(N x 64 x 14 x 14) - Relu - Max Pooling(N x 64 x 7 x 7) - Dropout(0.5)
  - Layer3: FC(64 x 7 x 7, 256) - Relu - Dropout(0.5) - FC(256, 10)
- Criterion: Cross entropy error
- Optimizer: Adam
- Learning rate: 0.001
- Epochs: 10 (batch_size: 64, shuffle)

### pytorch
```python
class CNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Sequential(
                torch.nn.Conv2d(1, 32, 3, padding=1),   # N x 32 x 28 x 28
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(2, 2))               # N x 64 x 14 x 14

        self.layer2 = torch.nn.Sequential(
                torch.nn.Conv2d(32, 64, 3, padding=1),  # N x 32 x 28 x 28
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(2,2))                # N x 64 x 7 x 7

        self.fc1 = torch.nn.Sequential(
                torch.nn.Linear(64*7*7, 256), torch.nn.ReLU())
        self.fc2 = torch.nn.Linear(256,10)

    def forward(self, x):
        x = F.dropout(self.layer1(x), training=self.training)
        x = F.dropout(self.layer2(x), training=self.training)
        x = x.view(-1, 64*7*7)
        x = F.dropout(self.fc1(x), training=self.training)
        return self.fc2(x)


class Evaluator:
    def __init__(self, model, criterion):
        self.model = model
        self.criterion = criterion

    def __call__(self, data, target):
        with torch.no_grad():
            output = self.model(data)
            loss = self.criterion(output, target).item()
            acc = torch.eq(F.softmax(output, 1).argmax(1),
                           target).float().mean().item()
        return loss, acc


# Setup a model:
model = CNN().to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
score = Evaluator(model, criterion)

# Train the model:
for epoch in range(n_epoch):
	# Train the model:
	model.train()
	for i, (data, target) in enumerate(train_loader):
		data, target = data.to(device), target.to(device)
		loss = criterion(model(data), target)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

	# Evaluate the model:
	model.eval()
	with torch.no_grad():
		for data, target in test_loader:
			data, target = data.to(device), target.to(device)
			loss_batch, acc_batch = score(data, target)
```

### tensorflow
```python
x = tf.placeholder(tf.float32, [None, 28, 28, 1])
y = tf.placeholder(tf.float32, [None, 10])
is_training = tf.placeholder(tf.bool)

# Setup a model:
tf.set_random_seed(0)
layer1 = tf.layers.conv2d(x, 32, [3, 3], activation=tf.nn.relu)
layer1 = tf.layers.max_pooling2d(layer1, [2, 2], [2, 2])
layer1 = tf.layers.dropout(layer1, 0.5, is_training)

layer2 = tf.layers.conv2d(layer1, 64, [3, 3], activation=tf.nn.relu)
layer2 = tf.layers.max_pooling2d(layer2, [2, 2], [2, 2])
layer2 = tf.layers.dropout(layer2, 0.5, is_training)

layer3 = tf.contrib.layers.flatten(layer2)
layer3 = tf.layers.dense(layer3, 256, activation=tf.nn.relu)
layer3 = tf.layers.dropout(layer3, 0.5, is_training)

output = tf.layers.dense(layer3, 10, activation=None)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
           logits=output, labels=y))
optimizer = tf.train.AdamOptimizer(lr).minimize(loss)

correct = tf.equal(tf.argmax(output,1), tf.argmax(y,1))
acc = tf.reduce_mean(tf.cast(correct, tf.float32))

# Train the model
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
	for epoch in range(n_epoch):
		data = {x:x_train[batch], y:y_train[batch], is_training:True}
		sess.run(optimizer, feed_dict=data)
		loss_batch, acc_batch  = sess.run([loss, acc], feed_dict=data)
```

### keras
```python
# Setup a model:
model = Sequential()
#model.add(keras.layers.Input(shape=(28, 28, 1)))
model.add(keras.layers.Conv2D(32, (3, 3), activation='relu',
          input_shape=(28, 28, 1)))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Dropout(0.5))

model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Dropout(0.5))

model.add(keras.layers.Flatten())
model.add(Dense(256, activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.adam(lr=lr),
              metrics=['accuracy'])

# Train the model
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=n_epoch,
                    verbose=verbose,
                    validation_data=(x_test, y_test))
```
