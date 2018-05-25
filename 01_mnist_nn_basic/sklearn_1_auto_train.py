import sys, os
sys.path.append(os.pardir)

from sklearn.neural_network import MLPClassifier
from datetime import datetime
import common.numpy_nn as np_nn


if __name__ == "__main__":

    # Set hyper-parameters:
    n_epoch, batch_size, lr = 10, 64, 0.01
    shuffle, verbose = True, True

    # Load data:
    x_train, y_train, x_test, y_test = np_nn.mnist(onehot=True)

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