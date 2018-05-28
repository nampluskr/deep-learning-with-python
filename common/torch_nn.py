import torch
from torch.utils.data import Dataset, DataLoader
import common.mnist as mnist


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


class Accuracy:
    def __init__(self, model, criterion):
        self.model = model
        self.criterion = criterion

    def __call__(self, data, target):
        with torch.no_grad():
            output = self.model(data)
            loss = self.criterion(output, target).item()
            acc = torch.eq(torch.nn.functional.softmax(output, 1).argmax(1),
                           target).float().mean().item()

        return loss, acc


if __name__ == "__main__":

    # Set hyper-parameters:
    n_epoch, batch_size, lr = 10, 64, 0.001
    shuffle, verbose = True, True

    # Load data:
    train_loader = DataLoader(MNIST(train=True),
                              batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(MNIST(train=False),
                             batch_size=batch_size, shuffle=False)

    print(len(train_loader.dataset), train_loader.batch_size)