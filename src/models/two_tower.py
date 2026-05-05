import torch
import torch.nn as nn


class TwoTowerModel(nn.Module):
    def __init__(
        self,
        num_users,
        num_items,
        num_genres,
        embed_dim=64
    ):
        super().__init__()

        # user id embedding
        self.user_embedding = nn.Embedding(
            num_users,
            embed_dim
        )

        # item id embedding
        self.item_embedding = nn.Embedding(
            num_items,
            embed_dim
        )

        # genre embedding
        # +1 because 0 is padding
        self.genre_embedding = nn.Embedding(
            num_genres + 1,
            embed_dim,
            padding_idx=0
        )

        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        nn.init.normal_(self.genre_embedding.weight, std=0.01)

    def encode_user(
        self,
        user,
        history
    ):
        """
        user: [B]
        history: [B, H]
        """

        user_emb = self.user_embedding(user)  # [B,d]

        history_emb = self.item_embedding(history)  # [B,H,d]

        history_emb = history_emb.mean(dim=1)  # [B,d]

        user_vec = user_emb + history_emb

        return user_vec

    def encode_item(
        self,
        item,
        genres
    ):
        """
        item: [B]
        genres: [B,G]
        """

        item_emb = self.item_embedding(item)  # [B,d]

        genre_emb = self.genre_embedding(genres)  # [B,G,d]

        genre_emb = genre_emb.mean(dim=1)  # [B,d]

        item_vec = item_emb + genre_emb

        return item_vec

    def forward(
        self,
        user,
        history,
        pos_item,
        neg_item,
        pos_genres,
        neg_genres
    ):
        user_vec = self.encode_user(
            user,
            history
        )

        pos_vec = self.encode_item(
            pos_item,
            pos_genres
        )

        neg_vec = self.encode_item(
            neg_item,
            neg_genres
        )

        pos_score = (
            user_vec * pos_vec
        ).sum(dim=1)

        neg_score = (
            user_vec * neg_vec
        ).sum(dim=1)

        return pos_score, neg_score

    def bpr_loss(
        self,
        pos_score,
        neg_score
    ):
        loss = -torch.log(
            torch.sigmoid(
                pos_score - neg_score
            ) + 1e-8
        )

        return loss.mean()