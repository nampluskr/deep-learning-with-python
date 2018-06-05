import sys, os
sys.path.append(os.pardir)

import torch
from torch.utils.data import DataLoader
from datetime import datetime
import common.torch_nn as tch_nn
import matplotlib.pyplot as plt


class Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Sequential(
                torch.nn.Linear(28*28, 50),
                torch.nn.ReLU())

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.layer(x)


class Decoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Sequential(
                torch.nn.Linear(50, 28*28),
                torch.nn.Sigmoid())

    def forward(self, x):
        y = self.layer(x)
        return y.view(-1, 1, 28, 28)


if __name__ == "__main__":

    use_gpu = 1
    device = torch.device("cuda") if use_gpu else torch.device("cpu")

    # Set hyper-parameters:
    n_epoch, batch_size, lr = 10, 64, 0.0001
    shuffle, verbose = True, True

    # Load data:
    train_loader = DataLoader(tch_nn.MNIST(train=True),
                              batch_size=batch_size, shuffle=shuffle)

    # Setup a model:
    torch.manual_seed(0)

    encoder = Encoder().to(device)
    decoder = Decoder().to(device)
    parameters = list(encoder.parameters()) + list(decoder.parameters())
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(parameters, lr=lr)
#    score = Evaluator(model, criterion)

    # Train the model:
    message = "Epoch[{:3d}] ({:3d}%) > Loss {:.4f}"

    t_start = datetime.now()
    for epoch in range(n_epoch):
        if verbose:
            print('')

        encoder.train()
        decoder.train()
        loss_train, n_batch = 0, len(train_loader)
        for i, (data, _) in enumerate(train_loader):
            data = data.to(device)

            output = encoder(data)
            output = decoder(output)
            loss = criterion(output, data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Evaluate the model after an epoch:
            loss_train += loss
            if verbose and (i+1) % 100 == 0:
                print(message.format(epoch+1,
                      int(100*(i+1)*batch_size/len(train_loader.dataset)), loss))

        print(message.format(epoch+1, 100, loss_train/n_batch),
              "(Time {})".format(datetime.now() - t_start))

    input_img = torch.squeeze(data).detach()
    output_img = torch.squeeze(output).detach()
    for i in range(data.size(0)):
        fig, (ax1, ax2) = plt.subplots(1,2)
        ax1.imshow(input_img[i], cmap='gray')
        ax1.set_xticks([]); ax1.set_yticks([])
        ax2.imshow(output_img[i], cmap='gray')
        ax2.set_xticks([]); ax2.set_yticks([])
        plt.show()