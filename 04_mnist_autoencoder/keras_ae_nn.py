import sys, os
sys.path.append(os.pardir)

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
import common.mnist as mnist


# To prevent CUDA_ERROR_OUT_OF_MEMORY:
config = keras.backend.tf.ConfigProto()
config.gpu_options.allow_growth = True
session = keras.backend.tf.Session(config=config)

# Set hyper-parameters:
n_epoch, batch_size, lr = 10, 64, 0.0001
verbose = True

# Load data:
data, _, _, _ = mnist.load()

# Setup a model:
model = Sequential()
model.add(Dense(50, activation='relu', input_shape=(784,)))
model.add(Dense(10, activation='sigmoid'))
model.summary()

model.compile(loss='mean_squared_error',
              optimizer=optimizers.adam(lr=lr),
              metrics=['accuracy'])

# Train the model
history = model.fit(data, data,
                    batch_size=batch_size,
                    epochs=n_epoch,
                    verbose=verbose)

# Evaluate the trained model:
#loss_test, acc_test = model.evaluate(x_test, y_test, verbose=0)
#print("\nEpoch[{:3d}] > Test Loss: {:.3f} / Test Acc. {:.3f}".format(
#                    n_epoch, loss_test, acc_test))