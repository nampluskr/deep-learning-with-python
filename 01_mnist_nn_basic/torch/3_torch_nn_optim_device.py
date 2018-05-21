import pickle
import numpy as np
import torch
import torch.nn.functional as F
from datetime import datetime


def onehot(y):
    y_onehot = torch.zeros(y.size(0), 10)
    return y_onehot.scatter(1, y.view(-1,1), 1)


if __name__ == "__main__":

    use_gpu = 1
    device = torch.device("cuda") if use_gpu else torch.device("cpu")

    # Set hyper-parameters:
    n_epoch, batch_size, lr = 10, 64, 0.01
    shuffle, verbose = True, True

    # Load data:
    mnist = pickle.load(open('..\..\data\mnist.pkl', 'rb'))
    x_train = mnist['train_img']/255.
    y_train = mnist['train_label']
    x_test  = mnist['test_img']/255.
    y_test  = mnist['test_label']

    x_train = torch.from_numpy(x_train).float().to(device)
    y_train = torch.from_numpy(y_train).long().to(device)
    x_test = torch.from_numpy(x_test).float().to(device)
    y_test = torch.from_numpy(y_test).long().to(device)

    # Setup a model:
    torch.manual_seed(0)

    model = torch.nn.Sequential(torch.nn.Linear(784,200), torch.nn.Sigmoid(),
                                torch.nn.Linear(200,10)).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

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
            data, target = x_train[batch].to(device), y_train[batch].to(device)

            output = model(data)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

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
        data, target = x_test.to(device), y_test.to(device)
        output = model(data)
        loss_test = criterion(output, target)
        acc_test = torch.eq(F.softmax(output,1).argmax(1),
                            target).float().mean().item()

    print("\nEpoch[{:3d}] > Test Loss: {:.3f} / Test Acc. {:.3f}".format(
                    epoch+1, loss_test, acc_test))