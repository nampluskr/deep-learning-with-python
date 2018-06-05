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
        self.add(Dense(200, activation='relu', input_shape=(784,)))
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
    x_train, y_train, x_test, y_test = mnist.load()

    # Setup a model:
    model = NeuralNet()

    # Train the model
    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=n_epoch,
                        verbose=verbose,
                        validation_data=(x_test, y_test))

    # Evaluate the trained model:
    loss_test, acc_test = model.evaluate(x_test, y_test, verbose=0)
    print("\nEpoch[{:3d}] > Test Loss: {:.3f} / Test Acc. {:.3f}".format(
                        n_epoch, loss_test, acc_test))