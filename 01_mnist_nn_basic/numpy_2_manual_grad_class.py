import sys, os
sys.path.append(os.pardir)

import numpy as np
from datetime import datetime
import common.numpy_nn as np_nn
import common.mnist as mnist


class NetNumpy:
    def __init__(self):
        np.random.seed(0)
        self.w1 = np.random.randn(784, 200)*np.sqrt(1./784)
        self.b1 = np.zeros(200)
        self.w2 = np.random.randn(200, 10)*np.sqrt(1./200)
        self.b2 = np.zeros(10)

    def forward(self, x):
        self.data = x
        lin1 = np.dot(self.data, self.w1) + self.b1
        self.sig1 = np_nn.sigmoid(lin1)
        lin2 = np.dot(self.sig1, self.w2) + self.b2
        return lin2

    def backward(self, dout=1):
        grad_output = (self.output - target)/target.shape[0]
        grad_lin2 = grad_output
        self.grad_w2 = np.dot(self.sig1.T, grad_lin2)
        self.grad_b2 = np.sum(grad_lin2, axis=0)

        grad_sig1 = np.dot(grad_lin2, self.w2.T)
        grad_lin1 = self.sig1*(1 - self.sig1)*grad_sig1
        self.grad_w1 = np.dot(self.data.T, grad_lin1)
        self.grad_b1 = np.sum(grad_lin1, axis=0)

    def update(self, lr):
        self.w2 -= lr*self.grad_w2
        self.b2 -= lr*self.grad_b2
        self.w1 -= lr*self.grad_w1
        self.b1 -= lr*self.grad_b1

    def loss(self, x, t):
        self.output = self.forward(x)
        return np_nn.cross_entropy(np_nn.softmax(self.output), t)

    def score(self, x, t):
        y = self.forward(x)
        return (np_nn.softmax(y).argmax(1) == t.argmax(1)).mean()


if __name__ == "__main__":

    # Set hyper-parameters:
    n_epoch, batch_size, lr = 10, 64, 0.01
    shuffle, verbose = True, True

    # Load data:
    x_train, y_train, x_test, y_test = mnist.load()

    # Setup a model:
    model = NetNumpy()

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
            model.update(lr=lr)

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