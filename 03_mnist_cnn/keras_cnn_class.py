import sys, os
sys.path.append(os.pardir)

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
import common.mnist as mnist


class NeuralNet(Sequential):
    def __init__(self):
        super().__init__()
        self.add(keras.layers.Conv2D(32, (3, 3), activation='relu',
                  input_shape=(28, 28, 1)))
        self.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        self.add(keras.layers.Dropout(0.5))

        self.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
        self.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        self.add(keras.layers.Dropout(0.5))

        self.add(keras.layers.Flatten())
        self.add(Dense(256, activation='relu'))
        self.add(keras.layers.Dropout(0.5))
        self.add(Dense(10, activation='softmax'))

        self.compile(loss='categorical_crossentropy',
                      optimizer=optimizers.adam(lr=lr),
                      metrics=['accuracy'])


if __name__ == "__main__":

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

    # Setup a model:
    model = NeuralNet()
    model.summary()

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