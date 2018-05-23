import sys, os
sys.path.append(os.pardir)

import keras
import common.numpy_nn as np_nn


# To prevent CUDA_ERROR_OUT_OF_MEMORY:
config = keras.backend.tf.ConfigProto()
config.gpu_options.allow_growth = True
session = keras.backend.tf.Session(config=config)

# Set hyper-parameters:
n_epoch, batch_size, lr = 10, 64, 0.01
verbose = True

# Load data:
x_train, y_train, x_test, y_test = np_nn.mnist(one_hot=True)

# Setup a model:
model = keras.models.Sequential()
model.add(keras.layers.Dense(200, activation='sigmoid', input_shape=(784,)))
model.add(keras.layers.Dense(10, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.sgd(lr=lr),
              metrics=['accuracy'])

# Train the model
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=n_epoch,
                    verbose=verbose,
                    validation_data=(x_test, y_test))

# Evaluate the model after training:
loss_test, acc_test = model.evaluate(x_test, y_test, verbose=0)
print("\nEpoch[{:3d}] > Test Loss: {:.3f} / Test Acc. {:.3f}".format(
                    n_epoch, loss_test, acc_test))