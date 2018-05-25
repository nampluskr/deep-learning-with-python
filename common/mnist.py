import pickle
import numpy as np


def load(flatten=True, onehot=True):
    mnist = pickle.load(open('..\data\mnist.pkl', 'rb'))
    x_train = mnist['train_img']/255.
    x_test  = mnist['test_img']/255.
    y_train = mnist['train_label']
    y_test  = mnist['test_label']

    if onehot:
        y_train = onehot_encode(y_train, 10)
        y_test = onehot_encode(y_test, 10)

    if not flatten:
        x_train = x_train.reshape(-1, 1, 28, 28)
        x_test = x_test.reshape(-1, 1, 28, 28)

    return x_train, y_train, x_test, y_test


def onehot_encode(y, size):
    y_onehot = np.zeros((y.shape[0], size), dtype=int)
    for i, row in enumerate(y_onehot):
        row[int(y[i])] = 1.0
    return y_onehot

if __name__ == "__main__":

    x_train, y_train, x_test, y_test = load()

    print("data", x_train.shape)
    print("target", y_train.shape)