import numpy as np
import keras


# Set hyper-parameters:
n_epoch, lr = 100, 0.01

# Load data:
np.random.seed(0)
n_data = 1000
noise = np.random.normal(0, 1, (n_data, 1))
x_train = np.random.uniform(-10, 10, (n_data,1))
y_train = x_train*2 + 10 + noise

# Setup a model
model = keras.models.Sequential()
model.add(keras.layers.Dense(1, input_dim=1))
model.compile(loss='mse', optimizer=keras.optimizers.SGD(lr=lr))

# Train the model:
history = model.fit(x_train, y_train, epochs=n_epoch, verbose=0)

# Evaluate the trained model:
w1, b1 = model.get_weights()
loss = model.evaluate(x_train, y_train, verbose=0)

print("Epoch[%3d] >> Loss = %f" % (n_epoch, loss))
print(w1, b1)