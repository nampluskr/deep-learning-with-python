import numpy as np
from scipy.special import expit as sigmoid, logsumexp


def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_data.shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1

    img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    return col


def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]


#########################################################################
# Functions:
#########################################################################
def softmax(x):
    logsum = logsumexp(x,1).reshape(-1,1) if x.ndim == 2 else logsumexp(x)
    return np.exp(x - logsum)


def cross_entropy(y, t):
    batch_size = y.shape[0] if y.ndim == 2 else 1
    return -np.sum(t*np.log(y+1.0E-8))/batch_size


#########################################################################
# Multi-layer neural net:
#########################################################################
class MultiNetNumpy:
    def __init__(self, layers, error):
        self.layers = layers
        self.error = error
        self.info()

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, dout=1):
        dy = self.error.backward(dout=1)
        for layer in reversed(self.layers):
            dy = layer.backward(dy)

    def loss(self, x, t):
        return self.error.forward(self.forward(x), t)

    def score(self, x, t):
        return self.error.score(self.forward(x), t)

    def info(self):
        names = [l.name.title() for l in self.layers] + [self.error.name]
        print("Model >>", ' - '.join(names))

    def train(self):
        for layer in self.layers:
            if layer.name == ('dropout'):
                layer.is_training = True

    def eval(self):
        for layer in self.layers:
            if layer.name == ('dropout'):
                layer.is_training = False


#########################################################################
# Layers:
#########################################################################
class Linear:
    def __init__(self, n, m, name, activation='sigmoid'):
        self.name = 'linear' + name
        sigma = {'sigmoid':np.sqrt(1./n), 'relu':np.sqrt(2./n)}
        self.w = np.random.normal(0, sigma[activation.lower()], (n, m))
        self.b = np.zeros(m)

    def forward(self, x):
        self.x = x.copy()
        return np.dot(x, self.w) + self.b

    def backward(self, dy):
        self.dw = np.dot(self.x.T, dy)
        self.db = np.sum(dy, axis=0)
        return np.dot(dy, self.w.T)


class Sigmoid:
    def __init__(self):
        self.name = 'sigmoid'

    def forward(self, x):
        self.y = sigmoid(x)
        return self.y

    def backward(self, dy):
        return self.y*(1-self.y)*dy


class Relu:
    def __init__(self):
        self.name = 'relu'

    def forward(self, x):
        self.x = x
        return np.where(x>0, x, 0)

    def backward(self, dy):
        return np.where(self.x>0, dy, 0)


class SoftmaxWithLoss:
    def __init__(self):
        self.name = "Softmax - Cross Entropy"

    def forward(self, y, t):
        self.y, self.t = softmax(y), t
        return cross_entropy(self.y, t)

    def backward(self, dout=1):
        return dout*(self.y-self.t)/self.t.shape[0]

    def score(self, y, t):
        return (y.argmax(1) == t.argmax(1)).mean()


#########################################################################
# Optimizers:
#########################################################################
class GradientDescent:
    def __init__(self, model, lr):
        self.model = model
        self.lr = lr
        self.name = "GradientDescent"

    def update(self):
        for layer in reversed(self.model.layers):
            if layer.name.startswith('linear') or layer.name.startswith('conv'):
                layer.w -= self.lr*layer.dw
                layer.b -= self.lr*layer.db


class Momentum:
    def __init__(self, model, lr, momentum=0.9):
        self.name = "Momentrum"
        self.model = model
        self.lr = lr
        self.momentum = momentum
        self.v1, self.v2 = {}, {}

    def update(self):
        for layer in reversed(self.model.layers):
            if layer.name.startswith('linear') or layer.name.startswith('conv'):
                name = layer.name

                if name not in self.v1.keys():
                    self.v1[name] = np.zeros_like(layer.w)
                    self.v2[name] = np.zeros_like(layer.b)

                self.v1[name] = self.momentum*self.v1[name] - self.lr*layer.dw
                self.v2[name] = self.momentum*self.v2[name] - self.lr*layer.db

                layer.w += self.v1[name]
                layer.b += self.v2[name]


class AdaGrad:
    def __init__(self, model, lr):
        self.name = "Adagrad"
        self.model = model
        self.lr = lr
        self.h1, self.h2 = {}, {}

    def update(self):
        for layer in reversed(self.model.layers):
            if layer.name.startswith('linear') or layer.name.startswith('conv'):
                name = layer.name

                if name not in self.h1.keys():
                    self.h1[name] = np.zeros_like(layer.w)
                    self.h2[name] = np.zeros_like(layer.b)

                self.h1[name] += layer.dw**2
                self.h2[name] += layer.db**2

                layer.w -= self.lr*layer.dw/np.sqrt(self.h1[name] + 1.0E-8)
                layer.b -= self.lr*layer.db/np.sqrt(self.h2[name] + 1.0E-8)


class Adam:
    def __init__(self, model, lr, beta1=0.9, beta2=0.999):
        self.name = "Adam"
        self.model = model
        self.lr = lr
        self.beta1, self.beta2 = beta1, beta2
        self.i, self.m1, self.m2, self.v1, self.v2 = {}, {}, {}, {}, {}

    def update(self):
        for layer in reversed(self.model.layers):
            if layer.name.startswith('linear') or layer.name.startswith('conv'):
                name = layer.name

                if name not in self.i.keys():
                    self.i[name] = 0
                    self.m1[name] = np.zeros_like(layer.w)
                    self.v1[name] = np.zeros_like(layer.w)
                    self.m2[name] = np.zeros_like(layer.b)
                    self.v2[name] = np.zeros_like(layer.b)

                self.i[name] += 1; i = self.i[name]
                lr_ = self.lr*np.sqrt(1-self.beta2**i)/(1-self.beta1**i)

                self.m1[name] += (1 - self.beta1)*(layer.dw - self.m1[name])
                self.v1[name] += (1 - self.beta2)*(layer.dw**2 - self.v1[name])

                self.m2[name] += (1 - self.beta1)*(layer.db - self.m2[name])
                self.v2[name] += (1 - self.beta2)*(layer.db**2 - self.v2[name])

                layer.w -= lr_*self.m1[name]/(np.sqrt(self.v1[name]) + 1.0E-8)
                layer.b -= lr_*self.m2[name]/(np.sqrt(self.v2[name]) + 1.0E-8)
