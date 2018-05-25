import sys, os
sys.path.append(os.pardir)

import numpy as np
from datetime import datetime
import common.numpy_nn as np_nn
import common.mnist as mnist


class Convolution:
    def __init__(self, ch_in, ch_out, name,
                 kernel_size=(3,3), stride=1, padding=1):
        self.name = 'conv' + name
        self.w = np.random.randn(ch_out, ch_in, *kernel_size)
        self.b = np.zeros(ch_out)
        self.h_kernel, self.w_kernel = kernel_size
        self.stride, self.padding = stride, padding

    def forward(self, x):
        self.x = x
        n_data, ch_in, h_in, w_in = x.shape
        h_out = (h_in + 2*self.padding - self.h_kernel)//self.stride + 1
        w_out = (w_in + 2*self.padding - self.w_kernel)//self.stride + 1

        self.x_dim2 = np_nn.im2col(x, self.h_kernel, self.w_kernel,
                                   self.stride, self.padding)
        self.w_dim2 = self.w.reshape(self.w.shape[0], -1)

        out = np.dot(self.x_dim2, self.w_dim2.T)
        out = out.reshape(n_data, h_out, w_out, -1).transpose(0,3,1,2)

        return out

    def backward(self, dy):
        dy = dy.transpose(0,2,3,1).reshape(-1, self.w.shape[0])

        self.dw = np.dot(self.x_dim2.T, dy)
        self.dw = self.dw.transpose(1,0).reshape(*self.w.shape)
        self.db = np.sum(dy, axis=0)

        dx_dim2 = np.dot(dy, self.w_dim2)
        dx = np_nn.col2im(dx_dim2, self.x.shape, self.h_kernel, self.w_kernel,
                          self.stride, self.padding)
        return dx



class MaxPooling:
    def __init__(self, kernel_size=(2,2), stride=2, padding=0):
        self.name = 'pooling'
        self.h_kernel, self.w_kernel = kernel_size
        self.stride, self.padding = stride, padding

    def forward(self, x):
        self.x = x
        n_data, ch_in, h_in, w_in = x.shape
        h_out = (h_in - self.h_kernel)//self.stride + 1
        w_out = (h_in - self.h_kernel)//self.stride + 1

        x_dim2 = np_nn.im2col(x, self.h_kernel, self.w_kernel, self.stride, self.padding)
        x_dim2 = x_dim2.reshape(-1, self.h_kernel*self.w_kernel)

        self.arg_max = np.argmax(x_dim2, axis=1)
        out = np.max(x_dim2, axis=1)
        out = out.reshape(n_data, h_out, w_out, -1).transpose(0,3,1,2)

        return out

    def backward(self, dy):
        dy = dy.transpose(0,2,3,1)
        dmax = np.zeros((dy.size, self.h_kernel*self.w_kernel))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dy.flatten()
        dmax = dmax.reshape(dy.shape + (self.h_kernel*self.w_kernel,))

        dx_dim2 = dmax.reshape(np.prod(dmax.shape),-1)
        dx = np_nn.col2im(dx_dim2, self.x.shape, self.h_kernel, self.w_kernel,
                          self.stride, self.padding)

        return dx


class Dropout:
    def __init__(self, ratio=0.5):
        self.name = 'dropout'
        self.ratio = ratio
        self.is_training = False

    def forward(self, x):
        if self.is_training:
            self.mask = np.random.rand(*x.shape) > self.ratio
            return x*self.mask
        else:
            return x

    def backward(self, dout):
        return dout*self.mask


class Flatten:
    def __init__(self):
        self.name = 'flatten'

    def forward(self, x):
        self.x = x
        return x.reshape(x.shape[0], -1)

    def backward(self, dy):
        return dy.reshape(*self.x.shape)


if __name__ == "__main__":

    # Set hyper-parameters:
    n_epoch, batch_size, lr = 1, 64, 0.001
    shuffle, verbose = True, True

    # Load data:
    x_train, y_train, x_test, y_test = mnist.load(onehot=True, flatten=False)

    # Setup a model:
    np.random.seed(0)
    layers = [Convolution(1, 32, name='1', kernel_size=(3,3)),
              np_nn.Relu(), MaxPooling(), Dropout(),
              Convolution(32, 64, name='2', kernel_size=(3,3)),
              MaxPooling(), Dropout(), Flatten(),
              np_nn.Linear(64*7*7, 256, name='3', activation='relu'),
              np_nn.Relu(), Dropout(),
              np_nn.Linear(256,10, name='4', activation='relu')]

    criterion = np_nn.SoftmaxWithLoss()
    model = np_nn.MultiNetNumpy(layers, criterion)
    optimizer = np_nn.Adam(model, lr=lr)

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

        model.train()
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
    model.eval()
    n_data = x_test.shape[0]
    n_batch = n_data // batch_size + (1 if n_data % batch_size else 0)

    loss_test, acc_test = 0, 0
    for i in range(n_batch):
        batch = np.arange(n_data)[i*batch_size:(i+1)*batch_size]
        data, target = x_test[batch], y_test[batch]

        loss_batch = model.loss(data, target)
        acc_batch = model.score(data, target)
        loss_test += loss_batch
        acc_test += acc_batch

    print("\nEpoch[{:3d}] > Test Loss: {:.3f} / Test Acc. {:.3f}".format(
                    n_epoch, loss_test/n_batch, acc_test/n_batch))