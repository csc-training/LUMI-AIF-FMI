import os

import torch
from torch import nn
import torch.nn.functional as F

from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
from torch_geometric.datasets import ModelNet
from torch_geometric import nn as gn

from tqdm import tqdm


data_root = f"/scratch/{os.environ['SLURM_JOB_ACCOUNT']}/data/ModelNet"


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        # TODO add graph convolution layers
        # TODO add feedfoward layers

    def forward(self, data):
        # TODO extract vertex positions, edge indices, and batch indices from the batch
        # TODO forward through the convolutions, alternating with activations
        # TODO apply a global pooling information (if it is not GPU supported, transfer needed data back and forth from the device)
        # TODO forward through the feedworward layers
        # TODO return result
        pass


def train(dl, model, loss_fn, optimizer, device=torch.device("cuda:0")):
    model = model.to(device)
    model.train()
    for step, data in (enumerate(pbar:=tqdm(dl))):
        data = data.to(device)

        optimizer.zero_grad()
        pred = model(data)
        loss = loss_fn(pred, data.y)
        pbar.set_description(f"Loss: {(loss.item()):>7f}")
        loss.backward()
        optimizer.step()


def test(dl, model, loss_fn, device=torch.device("cuda:0")):
    model = model.to(device)
    model.eval()
    loss, correct = 0.0, 0.0
    with torch.no_grad():
        for step, data in enumerate(dl):
            data = data.to(device)
            pred = model(data)
            loss += loss_fn(pred, data.y).item()
            correct += (pred.argmax(1) == data.y).type(torch.float).sum().item()
    loss /= len(dl)
    correct /= (len(dl.dataset))
    print(f"Test accuracy: {(100 * correct):>0.1f}%, Avg loss: {loss:>8f}")
    # print(f"output shape: {samples.size()}")


def main():

    epochs = 5
    learnint_rate = 1e-4
    batch_size = 64
    device = torch.device("cuda:0")


    model = Model()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    pre_transform=T.Compose([T.NormalizeScale(), T.FaceToEdge(),])
    transform=T.Compose([
        T.RandomJitter(0.01),
        T.RandomRotate(15, axis=0),
        T.RandomRotate(15, axis=1),
        T.RandomRotate(15, axis=2)
        ])

    train_ds = ModelNet(
        root=data_root,
        name="10",
        train=True,
        pre_transform=pre_transform,
        transform=transform)
    test_ds = ModelNet(
        root=data_root,
        name="10",
        train=False,
        pre_transform=pre_transform,
        transform=transform)


    train_dl = DataLoader(train_ds, batch_size=12, shuffle=True, num_workers=6)
    test_dl = DataLoader(test_ds, batch_size=12, shuffle=False, num_workers=6)

    for epoch in range(epochs):
        print(f"Epoch {epoch}:")
        train(train_dl, model, loss_fn, optimizer)
        test(train_dl, model, loss_fn)

if __name__ == "__main__":
    main()
