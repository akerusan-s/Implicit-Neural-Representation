import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np


class OdeDataset(Dataset):

    def __init__(self, x, timestamp):
        self.x = torch.tensor(x, dtype=torch.float)
        self.timestamp = torch.tensor(timestamp, dtype=torch.float, requires_grad=True).reshape(-1)

    def __len__(self):
        return self.x.shape[0] - 1

    def __getitem__(self, idx):
        return self.timestamp[idx], self.timestamp[idx + 1], self.x[idx], self.x[idx + 1]


def get_dataset(x_noised, timestamp):
    return OdeDataset(x_noised, timestamp)


def get_dataloader(x_noised, timestamp):
    return DataLoader(
        get_dataset(x_noised, timestamp),
        batch_size=1
    )


def get_data(path):
    data = np.load(path)
    return data["ts"], data["X"], data["X_dot"], data["X_noised"]
