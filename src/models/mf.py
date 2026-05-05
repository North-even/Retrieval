import torch
import torch.nn as nn

class MFModel(nn.Module):
    def __init__(self, num_users, num_items, embed_dim=64):
        super().__init__()

        self.user_embedding = nn.Embedding(num_users, embed_dim)
        self.item_embedding = nn.Embedding(num_items, embed_dim)

        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)

    def forward(self, user, pos_item, neg_item):
        """
        user: [B]
        pos_item: [B]
        neg_item: [B]
        """

        user_emb = self.user_embedding(user)        # [B,d]
        pos_emb = self.item_embedding(pos_item)     # [B,d]
        neg_emb = self.item_embedding(neg_item)     # [B,d]

        pos_score = (user_emb * pos_emb).sum(dim=1)  # [B]
        neg_score = (user_emb * neg_emb).sum(dim=1)  # [B]

        return pos_score, neg_score

    def bpr_loss(self, pos_score, neg_score):
        loss = -torch.log(torch.sigmoid(pos_score - neg_score) + 1e-8)
        return loss.mean()

