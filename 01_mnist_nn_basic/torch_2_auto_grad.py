import sys, os
sys.path.append(os.pardir)

import numpy as np
import torch
import torch.nn.functional as F
from datetime import datetime
import common.mnist as mnist


if __name__ == "__main__":

    # Set hyper-parameters:
    n_epoch, batch_size, lr = 10, 64, 0.01
    shuffle, verbose = True, True

    # Load data:
    x_train, y_train, x_test, y_test = mnist.load(onehot=False)

    x_train = torch.from_numpy(x_train).float()
    y_train = torch.from_numpy(y_train).long()
    x_test = torch.from_numpy(x_test).float()
    y_test = torch.from_numpy(y_test).long()

    # Setup a model:
    torch.manual_seed(0)
    w1 = torch.randn(784,200, requires_grad=True)
    b1 = torch.zeros(200, requires_grad=True)
    w2 = torch.randn(200,10, requires_grad=True)
    b2 = torch.zeros(10, requires_grad=True)

    # Train the model:
    n_data = x_train.shape[0]
    n_batch = n_data // batch_size + (1 if n_data % batch_size else 0)

    message = "Epoch[{:3d}] ({:3d}%) > Loss {:.3f} / Acc. {:.3f}"

    t_start = datetime.now()
    for epoch in range(n_epoch):
        index = torch.randperm(n_data) if shuffle else torch.arange(
                n_data, dtype=torch.int64)

        if shuffle:
            np.random.shuffle(index)

        if verbose:
            print('')

        loss_train, acc_train = 0, 0
        for i in range(n_batch):
            batch = index[i*batch_size:(i+1)*batch_size]
            data, target = x_train[batch], y_train[batch]

            # Forward propagation:
            lin1 = torch.mm(data, w1) + b1
            sig1 = F.sigmoid(lin1)
            lin2 = torch.mm(sig1, w2) + b2
            output  = lin2
            loss = F.cross_entropy(output, target)

            # Backward progapation:
            loss.backward()

            # Update weights and bias:
            for param in (w1, b1, w2, b2):
                param.data -= lr*param.grad.data
                param.grad.zero_()

            # Evaluate the model after an epoch:
            with torch.no_grad():
                loss_batch = F.cross_entropy(output, target).item()
                acc_batch = torch.eq(F.softmax(output,1).argmax(1),
                                     target).float().mean().item()
                loss_train += loss_batch
                acc_train += acc_batch

            if verbose and (i+1) % 100 == 0:
                print(message.format(epoch+1, int(100*(i+1)*batch_size/n_data),
                                     loss_batch, acc_batch))

        print(message.format(epoch+1, 100, loss_train/n_batch, acc_train/n_batch),
              "(Time {})".format(datetime.now() - t_start))

    # Evaluate the model after training:
    with torch.no_grad():
        sig1 = F.sigmoid(torch.mm(x_test, w1) + b1)
        output = torch.mm(sig1, w2) + b2

        loss_test = F.cross_entropy(output, y_test).item()
        acc_test = torch.eq(F.softmax(output,1).argmax(1),
                            y_test).float().mean().item()

    print("\nEpoch[{:3d}] > Test Loss: {:.3f} / Test Acc. {:.3f}".format(
                    epoch+1, loss_test, acc_test))