import sys, os
sys.path.append(os.pardir)

import numpy as np
from datetime import datetime
import numpy_nn_lib as nn
import common.numpy_nn as np_nn


if __name__ == "__main__":

    # Set hyper-parameters:
    n_epoch, batch_size, lr = 10, 64, 0.001
    shuffle, verbose = True, True

    # Load data:
    x_train, y_train, x_test, y_test = np_nn.mnist(one_hot=True)

    # Setup a model:
    np.random.seed(0)
    layers = [nn.Linear(784, 200, '1', activation='relu'), nn.Relu(),
              nn.Linear(200, 200, '2', activation='relu'), nn.Relu(),
              nn.Linear(200, 10, '3')]
    criterion = nn.SoftmaxWithLoss()
    model = nn.MultiNetNumpy(layers, criterion)
    optimizer = nn.Adam(model, lr=lr)

    # Train the model:
    n_data = x_train.shape[0]
    n_batch = n_data // batch_size + (1 if n_data % batch_size else 0)

    message = "Epoch[{:3d}] ({:3d}%) > Loss {:.3f} / Acc. {:.3f}"

    t_start = datetime.now()
    for epoch in range(n_epoch):
        index = np.arange(x_train.shape[0])
        if shuffle:
            np.random.shuffle(index)

        if verbose:
            print('')

        loss_train, acc_train = 0, 0
        for i in range(n_batch):
            batch = index[i*batch_size:(i+1)*batch_size]
            data, target = x_train[batch], y_train[batch]

            loss = model.loss(data, target)
            model.backward()
            optimizer.update()

            # Evaluate the model after an epoch:
            loss_batch = model.loss(data, target)
            acc_batch = model.score(data, target)
            loss_train += loss_batch
            acc_train += acc_batch

            if verbose and (i+1) % 100 == 0:
                print(message.format(epoch+1, int(100*(i+1)*batch_size/n_data),
                                     loss_batch, acc_batch))

        print(message.format(epoch+1, 100, loss_train/n_batch, acc_train/n_batch),
              "(Time {})".format(datetime.now() - t_start))

    # Evaluate the trained model:
    print("\nEpoch[{:3d}] > Test Loss: {:.3f} / Test Acc. {:.3f}".format(
                    epoch+1, model.loss(x_test, y_test), model.score(x_test, y_test)))