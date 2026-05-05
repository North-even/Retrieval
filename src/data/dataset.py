import random
import pandas as pd
import torch
from torch.utils.data import Dataset


class MFDataset(Dataset):
    def __init__(self, csv_path, num_items):
        self.df = pd.read_csv(csv_path)
        self.num_items = num_items

        self.user_items = (
            self.df.groupby("user_idx")["movie_idx"]
            .apply(set)
            .to_dict()
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        user = int(row["user_idx"])
        pos_item = int(row["movie_idx"])

        while True:
            neg_item = random.randint(0, self.num_items - 1)
            if neg_item not in self.user_items[user]:
                break

        return (
            torch.tensor(user, dtype=torch.long),
            torch.tensor(pos_item, dtype=torch.long),
            torch.tensor(neg_item, dtype=torch.long),
        )