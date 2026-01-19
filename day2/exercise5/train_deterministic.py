from pathlib import Path

import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
from tqdm import tqdm


class HousingDataset(Dataset):
    def __init__(self, root: Path = Path("./"), drop_proximity: bool = True):
        self.df = pd.read_csv(Path(root) / Path("housing.csv.zip")).dropna()
        self.proximity = True
        if drop_proximity:
            self.df = self.df.drop("ocean_proximity", axis="columns")
            self.proximity = False


    def __getitem__(self, idx: int) -> (torch.Tensor, torch.Tensor):
        if self.proximity:
            raise NotImplementedError
        else:
            x = torch.tensor(self.df.iloc[idx].loc["longitude":"median_income"].to_numpy(), dtype=torch.float32)
            y = torch.tensor(self.df.iloc[idx].loc["median_house_value"], dtype=torch.float32) / 10000

            return x, y

    def __len__(self) -> int:
        return len(self.df)

    def tokenize_proximity(self):
        raise NotImplementedError
        # TODO Check how many possible values ocean_proximity has and one-hot encode them.
        # TODO The returned input then has to be either a tuple of tensors:
        # TODO (floating point features: float, ocean_proximity: int) or a dictionary of tensors.


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        n1 = torch.nn.BatchNorm1d(8)
        l1 = torch.nn.Linear(8, 16)
        a1 = torch.nn.ReLU()
        l2 = torch.nn.Linear(16, 16)
        a2 = torch.nn.ReLU()
        l3 = torch.nn.Linear(16, 1)
        self.f = torch.nn.Sequential(l1, a1, l2, a2, l3)

    def forward(self, x):
        # TODO then add an embedding module to your network (see torch.nn.Embedding)
        # TODO concatenate or add the embedding to the input vector
        # TODO does including the ocean_proximity information improve performance?
        return torch.flatten(self.f(x))


def main():
    epochs = 2
    learning_rate = 1e-4
    batch_size = 64
    device = torch.device("cuda:0")
    ds = HousingDataset()
    # TODO split into training and test data
    dl = DataLoader(ds, batch_size=batch_size)
    model = Model()
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.MSELoss()
    model.to(device)

    for epoch in range(epochs):
        # for step, batch in enumerate(tqdm(dl)):
        for step, batch in enumerate(dl):
            x, y = batch
            x = x.to(device)
            y = y.to(device)

            optim.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            optim.step()
            print(loss/(torch.sum(y)/batch_size))
        print(loss/(torch.sum(y)/batch_size))

if __name__ == "__main__":
    main()
