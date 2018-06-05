import sys, os
sys.path.append(os.pardir)

import keras
from keras import models, layers, optimizers
import common.mnist as mnist


class CNN_Seq(models.Sequential):
    def __init__(self):
        super().__init__()
        self.add(layers.Conv2D(32, (3, 3), activation='relu',
                  input_shape=(28, 28, 1)))
        self.add(layers.MaxPooling2D(pool_size=(2, 2)))
        self.add(layers.Dropout(0.5))

        self.add(layers.Conv2D(64, (3, 3), activation='relu'))
        self.add(layers.MaxPooling2D(pool_size=(2, 2)))
        self.add(layers.Dropout(0.5))

        self.add(layers.Flatten())
        self.add(layers.Dense(256, activation='relu'))
        self.add(layers.Dropout(0.5))
        self.add(layers.Dense(10, activation='softmax'))

        self.compile(loss='categorical_crossentropy',
                      optimizer=optimizers.adam(lr=lr),
                      metrics=['accuracy'])


class CNN_Model(models.Model):
    def __init__(self):
        x = layers.Input(shape=(28, 28, 1))
        h = layers.Conv2D(32, (3, 3), activation='relu')(x)
        h = layers.MaxPooling2D(pool_size=(2, 2))(h)
        h = layers.Dropout(0.5)(h)

        h = layers.Conv2D(64, (3, 3), activation='relu')(h)
        h = layers.MaxPooling2D(pool_size=(2, 2))(h)
        h = layers.Dropout(0.5)(h)

        h = layers.Flatten()(h)
        h = layers.Dense(256, activation='relu')(h)
        h = layers.Dropout(0.5)(h)
        y = layers.Dense(10, activation='softmax')(h)

        super().__init__(x, y)
        self.compile(loss='categorical_crossentropy',
                      optimizer=optimizers.adam(lr=lr),
                      metrics=['accuracy'])


if __name__ == "__main__":

    # To prevent CUDA_ERROR_OUT_OF_MEMORY:
    config = keras.backend.tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = keras.backend.tf.Session(config=config)

    # Set hyper-parameters:
    n_epoch, batch_size, lr = 1, 64, 0.001
    verbose = True

    # Load data:
    x_train, y_train, x_test, y_test = mnist.load()
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    # Setup a model:
#    model = CNN_Seq()
    model = CNN_Model()
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