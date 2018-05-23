import sys, os
sys.path.append(os.pardir)

import numpy as np
from datetime import datetime
import common.numpy_nn as np_nn


if __name__ == "__main__":

    # Set hyper-parameters:
    n_epoch, batch_size, lr = 10, 64, 0.01
    shuffle, verbose = True, True

    # Load data:
    x_train, y_train, x_test, y_test = np_nn.mnist(one_hot=True)

    # Setup a model:
    np.random.seed(0)
    w1 = np.random.normal(0, 0.1, (784, 200))
    b1 = np.zeros(200)
    w2 = np.random.normal(0, 0.1, (200, 10))
    b2 = np.zeros(10)

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

            # Forward propagation:
            lin1 = np.dot(data, w1) + b1
            sig1 = np_nn.sigmoid(lin1)
            lin2 = np.dot(sig1, w2) + b2
            output  = lin2

            # Backward progapation:
            grad_output = (output - target)/target.shape[0]
            grad_lin2 = grad_output
            grad_w2 = np.dot(sig1.T, grad_lin2)
            grad_b2 = np.sum(grad_lin2, axis=0)

            grad_sig1 = np.dot(grad_lin2, w2.T)
            grad_lin1 = sig1*(1 - sig1)*grad_sig1
            grad_w1 = np.dot(data.T, grad_lin1)
            grad_b1 = np.sum(grad_lin1, axis=0)

            # Update model parameters:
            w2 -= lr*grad_w2
            b2 -= lr*grad_b2
            w1 -= lr*grad_w1
            b1 -= lr*grad_b1

            # Evaluate the model after an epoch:
            loss_batch = np_nn.cross_entropy(np_nn.softmax(output), target)
            acc_batch = (np_nn.softmax(output).argmax(1) == target.argmax(1)).mean()
            loss_train += loss_batch
            acc_train += acc_batch

            if verbose and (i+1) % 100 == 0:
                print(message.format(epoch+1, int(100*(i+1)*batch_size/n_data),
                                     loss_batch, acc_batch))

        print(message.format(epoch+1, 100, loss_train/n_batch, acc_train/n_batch),
              "(Time {})".format(datetime.now() - t_start))

    # Evaluate the trained model:
    sig1 = np_nn.sigmoid(np.dot(x_test, w1) + b1)
    pred = np_nn.softmax(np.dot(sig1, w2) + b2)

    loss_test = np_nn.cross_entropy(pred, y_test)
    acc_test = (pred.argmax(1) == y_test.argmax(1)).mean()

    print("\nEpoch[{:3d}] > Test Loss: {:.3f} / Test Acc. {:.3f}".format(
                    epoch+1, loss_test, acc_test))