import sys, os
sys.path.append(os.pardir)

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
#from keras.datasets import mnist
import common.mnist as mnist


# To prevent CUDA_ERROR_OUT_OF_MEMORY:
config = keras.backend.tf.ConfigProto()
config.gpu_options.allow_growth = True
session = keras.backend.tf.Session(config=config)

# Set hyper-parameters:
n_epoch, batch_size, lr = 10, 64, 0.001
verbose = True

# Load data:
x_train, y_train, x_test, y_test = mnist.load(flatten=False)
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

#(x_train, y_train), (x_test, y_test) = mnist.load_data()
#x_train = x_train.reshape(-1, 28, 28, 1)/255.
#y_train = keras.utils.to_categorical(y_train, 10)
#x_test = x_test.reshape(-1, 28, 28, 1)/255.
#y_test = keras.utils.to_categorical(y_test, 10)

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

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.adam(lr=lr),
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