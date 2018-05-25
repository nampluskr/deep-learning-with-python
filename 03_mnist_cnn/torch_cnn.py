import sys, os
sys.path.append(os.pardir)

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
import common.mnist as mnist


class CNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Sequential(
                torch.nn.Conv2d(1, 32, (3,3), padding=1),  # N, 32, 28, 28
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(2, 2))                  # N, 32, 14, 14

        self.layer2 = torch.nn.Sequential(
                torch.nn.Conv2d(32, 64, (3,3), padding=1), # N, 64, 14, 14
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(2,2))                   # N, 64, 7, 7

        self.fc1 = torch.nn.Sequential(
                torch.nn.Linear(64*7*7, 256), torch.nn.ReLU())
        self.fc2 = torch.nn.Linear(256,10)

    def forward(self, x):
        x = F.dropout(self.layer1(x), training=self.training)
        x = F.dropout(self.layer2(x), training=self.training)
        x = x.view(-1, 64*7*7)
        x = F.dropout(self.fc1(x), training=self.training)
        return self.fc2(x)


class MNIST(Dataset):
    def __init__(self, train=True):
        self.train = train
        x_train, y_train, x_test, y_test = mnist.load(onehot=False,
                                                      flatten=False)

        self.x_train = torch.from_numpy(x_train).float()
        self.y_train = torch.from_numpy(y_train).long()
        self.x_test = torch.from_numpy(x_test).float()
        self.y_test = torch.from_numpy(y_test).long()

        self.len = self.x_train.size(0) if train else self.x_test.size(0)

    def __getitem__(self, index):
        if self.train:
            return self.x_train[index], self.y_train[index]
        else:
            return self.x_test[index], self.y_test[index]

    def __len__(self):
        return self.len


class Evaluator:
    def __init__(self, model, criterion):
        self.model = model
        self.criterion = criterion

    def __call__(self, data, target):
        with torch.no_grad():
            output = self.model(data)
            loss = self.criterion(output, target).item()
            acc = torch.eq(F.softmax(output, 1).argmax(1),
                           target).float().mean().item()
        return loss, acc


if __name__ == "__main__":

    use_gpu = 1
    device = torch.device("cuda") if use_gpu else torch.device("cpu")

    # Set hyper-parameters:
    n_epoch, batch_size, lr = 10, 64, 0.001
    shuffle, verbose = True, True

    # Load data:
    train_loader = DataLoader(MNIST(train=True), batch_size=batch_size,
                              shuffle=shuffle)
    test_loader = DataLoader(MNIST(train=False), batch_size=batch_size,
                             shuffle=False)

    # Setup a model:
    torch.manual_seed(0)

    model = CNN().to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    score = Evaluator(model, criterion)

    # Train the model:
    message = "Epoch[{:3d}] ({:3d}%) > Loss {:.4f} / Acc. {:.4f}"

    t_start = datetime.now()
    for epoch in range(n_epoch):
        if verbose:
            print('')

        model.train()
        loss_train, acc_train, n_batch = 0, 0, len(train_loader)
        for i, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            loss = criterion(model(data), target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Evaluate the model after an epoch:
            with torch.no_grad():
                loss_batch, acc_batch = score(data, target)
                loss_train += loss_batch
                acc_train += acc_batch

            if verbose and (i+1) % 100 == 0:
                print(message.format(epoch+1,
                      int(100*(i+1)*batch_size/len(train_loader.dataset)),
                      loss_batch, acc_batch))

        print(message.format(epoch+1, 100, loss_train/n_batch, acc_train/n_batch),
              "(Time {})".format(datetime.now() - t_start))

        # Evaluate the model after training:
        model.eval()
        with torch.no_grad():
            loss_test, acc_test, n_batch = 0, 0, len(test_loader)
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)

                loss_batch, acc_batch = score(data, target)
                loss_test += loss_batch
                acc_test += acc_batch

            print("\nEpoch[{:3d}] > Test Loss: {:.4f} / Test Acc. {:.4f}".format(
                        epoch+1, loss_test/n_batch, acc_test/n_batch))