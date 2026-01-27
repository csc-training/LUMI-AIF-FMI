import os

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from torch_blue import vi

from tqdm import tqdm


data_root = f"/scratch/{os.environ['SLURM_JOB_ACCOUNT']}/data/"


# TODO turun this into a VIModule
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )
    def forward(self, x):
        return self.f(x)

def train(dl, model, loss_fn, optimizer, device=torch.device("cuda:0"), num_samples=10):
    model = model.to(device)
    model.train()
    for step, batch in (enumerate(pbar:=tqdm(dl))):
        x, y = batch
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        # TODO you can pass a `samples` parameter to a VIModule to set how many forward passes it does
        pred = model(x)
        loss = loss_fn(pred, y)
        pbar.set_description(f"Loss: {(loss.item()):>7f}")
        loss.backward()
        optimizer.step()

def test(dl, model, loss_fn, predictive_distribution=None, device=torch.device("cuda:0"), num_samples=10):
    model = model.to(device)
    model.eval()
    loss, correct = 0.0, 0.0
    with torch.no_grad():
        for step, batch in enumerate(dl):
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            samples = model(x)
            loss += loss_fn(samples, y).item()
            if predictive_distribution is not None:
                pred = predictive_distribution.predictive_parameters_from_samples(samples)
            else:
                pred = pred = samples
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    loss /= len(dl)
    correct /= (len(dl.dataset))
    print(f"Test accuracy: {(100 * correct):>0.1f}%, Avg loss: {loss:>8f}")

def main():
    epochs = 5
    learning_rate = 1e-4
    batch_size = 64
    train_ds = datasets.FashionMNIST(
        root=data_root,
        train=True,
        download=False,
        transform=ToTensor(),
    )
    test_ds = datasets.FashionMNIST(
        root=data_root,
        train=False,
        download=False,
        transform=ToTensor(),
    )


    train_dl = DataLoader(train_ds, batch_size=batch_size)
    test_dl = DataLoader(test_ds, batch_size=batch_size)

    model = Model()

    # NOTE return log probs of weights needed by KL loss
    model.return_log_probs = True
    print(model)

    # TODO define a categorical predictive distribution for classification
    predictive_distribution = None
    # TODO switch the loss_fn to a Kullback-Leibler loss
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        print(f"Epoch {epoch}:")
        train(train_dl, model, loss_fn, optimizer)
        test(train_dl, model, loss_fn, predictive_distribution)

    # NOTE save checkpoint

if __name__ == "__main__":
    main()
