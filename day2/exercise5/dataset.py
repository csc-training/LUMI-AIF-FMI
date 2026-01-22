from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset


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

