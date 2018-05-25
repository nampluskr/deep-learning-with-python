import sys, os
sys.path.append(os.pardir)

import numpy as np
from datetime import datetime
import common.numpy_nn as np_nn


class Convolution:
    def __init__(self, ch_in, ch_out, kernel_size=(3,3), stride=1, padding=1):
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
        self.ratio = ratio

    def forward(self, x, is_training=True):
        self.mask = np.random.rand(*x.shape) > self.ratio if is_training \
                    else np.ones(*x.shape)

        return x*self.mask

    def backward(self, dout):
        return dout*self.mask


if __name__ == "__main__":

    # Set hyper-parameters:
    n_epoch, batch_size, lr = 10, 64, 0.001
    shuffle, verbose = True, True

    # Load data:
    x_train, y_train, x_test, y_test = np_nn.mnist(one_hot=True,
                                                   flatten=False)

    conv = Convolution(1, 32, (3,3), 1, 1)
    pool = MaxPooling((2,2), 2, 0)

    input_data = x_train[:100]
    print("Input   >>", input_data.shape)

    out = conv.forward(input_data)
    print("Conv    >>", out.shape)

    out = pool.forward(out)
    print("Pooling >>", out.shape)


    dout = np.ones_like(out)
    dout = pool.backward(dout)
    print(dout.shape)

    dout = conv.backward(dout)
    print(dout.shape)
