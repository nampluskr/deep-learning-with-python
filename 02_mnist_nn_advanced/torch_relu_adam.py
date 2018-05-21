import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from datetime import datetime


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
                torch.nn.Linear(784,200), torch.nn.ReLU(),
                torch.nn.Linear(200,200), torch.nn.ReLU(),
                torch.nn.Linear(200,10))

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(-1, 784)
        return self.layers(x)


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
    train_dataset = datasets.MNIST(root='../data/', train=True,
                                   transform=transforms.ToTensor())
    test_dataset = datasets.MNIST(root='../data/', train=False,
                                  transform=transforms.ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Setup a model:
    torch.manual_seed(0)

    model = Net().to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    score = Evaluator(model, criterion)

    # Train the model:
    message = "Epoch[{:3d}] ({:3d}%) > Loss {:.3f} / Acc. {:.3f}"

    t_start = datetime.now()
    for epoch in range(n_epoch):
        if verbose:
            print('')

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
    with torch.no_grad():
        loss_test, acc_test, n_batch = 0, 0, len(test_loader)
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            loss_batch, acc_batch = score(data, target)
            loss_test += loss_batch
            acc_test += acc_batch

        print("\nEpoch[{:3d}] > Test Loss: {:.3f} / Test Acc. {:.3f}".format(
                    epoch+1, loss_test/n_batch, acc_test/n_batch))