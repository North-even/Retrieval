import json
import random

import pandas as pd
import torch
from torch.utils.data import Dataset


class TwoTowerDataset(Dataset):
    def __init__(
        self,
        train_csv_path,
        movie_genres_path,
        idx2movie_path,
        num_items,
        max_history_len=20,
        max_genre_len=6,
    ):
        self.df = pd.read_csv(train_csv_path)
        self.num_items = num_items
        self.max_history_len = max_history_len
        self.max_genre_len = max_genre_len

        with open(movie_genres_path, "r", encoding="utf-8") as f:
            movie_genres = json.load(f)

        self.movie_genres = {
            int(k): v for k, v in movie_genres.items()
        }

        with open(idx2movie_path, "r", encoding="utf-8") as f:
            idx2movie = json.load(f)

        self.idx2movie = {
            int(k): int(v) for k, v in idx2movie.items()
        }

        self.user_items = (
            self.df.groupby("user_idx")["movie_idx"]
            .apply(set)
            .to_dict()
        )

        self.user_histories = (
            self.df.sort_values(["user_idx", "timestamp"])
            .groupby("user_idx")["movie_idx"]
            .apply(list)
            .to_dict()
        )

    def __len__(self):
        return len(self.df)

    def _sample_negative(self, user):
        while True:
            neg_item = random.randint(0, self.num_items - 1)
            if neg_item not in self.user_items[user]:
                return neg_item

    def _get_history(self, user, pos_item):
        history = self.user_histories[user]

        history = [item for item in history if item != pos_item]

        if len(history) >= self.max_history_len:
            history = history[-self.max_history_len:]
        else:
            history = [0] * (self.max_history_len - len(history)) + history

        return history

    def _get_genres(self, movie_idx):
        original_movie_id = self.idx2movie[movie_idx]
        genres = self.movie_genres.get(original_movie_id, [])

        # genre id + 1，保留 0 作为 padding
        genres = [g + 1 for g in genres]

        if len(genres) >= self.max_genre_len:
            genres = genres[:self.max_genre_len]
        else:
            genres = genres + [0] * (self.max_genre_len - len(genres))

        return genres

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        user = int(row["user_idx"])
        pos_item = int(row["movie_idx"])
        neg_item = self._sample_negative(user)

        history = self._get_history(user, pos_item)
        pos_genres = self._get_genres(pos_item)
        neg_genres = self._get_genres(neg_item)

        return (
            torch.tensor(user, dtype=torch.long),
            torch.tensor(history, dtype=torch.long),
            torch.tensor(pos_item, dtype=torch.long),
            torch.tensor(neg_item, dtype=torch.long),
            torch.tensor(pos_genres, dtype=torch.long),
            torch.tensor(neg_genres, dtype=torch.long),
        )