import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
from datetime import datetime


def onehot_encode(y, size):
    y_onehot = np.zeros((y.shape[0], size), dtype=int)
    for i, row in enumerate(y_onehot):
        row[int(y[i])] = 1.0
    return y_onehot


if __name__ == "__main__":

    # Set hyper-parameters:
    n_epoch, batch_size, lr = 10, 64, 0.01
    shuffle, verbose = True, True

    # Load data:
    mnist = fetch_mldata('MNIST original')
    x_train, x_test, y_train, y_test = train_test_split(
            mnist.data/255., mnist.target, test_size=10000)

    y_train = onehot_encode(y_train, 10)
    y_test  = onehot_encode(y_test, 10)

    # Setup a model:
    clf = MLPClassifier(solver='sgd', activation='logistic',
                        hidden_layer_sizes=(200,), max_iter=n_epoch,
                        random_state=0, learning_rate_init=lr,
                        shuffle=shuffle, batch_size=batch_size)

    # Train the model:
    t_start = datetime.now()
    clf.fit(x_train, y_train)

    message = "Epoch[{:3d}] ({:3d}%) > Loss {:.3f} / Acc. {:.3f}"
    print(message.format(clf.n_iter_, 100, clf.loss_, clf.score(x_train, y_train),
          "(Time {})".format(datetime.now() - t_start)))

    # Evaluate the model after training:
    print("Epoch[{:3d}] > Test Loss: {:.3f} / Test Acc. {:.3f}".format(
          clf.n_iter_, clf.score(x_test, y_test), clf.score(x_test, y_test)))