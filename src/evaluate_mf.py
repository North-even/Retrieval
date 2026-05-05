import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.models.mf import MFModel


def recall_at_k(ranked_items, target_item, k):
    return int(target_item in ranked_items[:k])


def ndcg_at_k(ranked_items, target_item, k):
    topk = ranked_items[:k]

    if target_item not in topk:
        return 0.0

    rank = topk.index(target_item)
    return 1.0 / np.log2(rank + 2)


def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    output_dir = project_root / "outputs" / "mf_baseline"
    model_path = output_dir / "best.pt"

    data_dir = project_root / "data" / "processed"
    train_path = data_dir / "train.csv"
    valid_path = data_dir / "valid.csv"

    checkpoint = torch.load(model_path, map_location=device)

    num_users = checkpoint["num_users"]
    num_items = checkpoint["num_items"]
    embed_dim = checkpoint["embed_dim"]

    model = MFModel(
        num_users=num_users,
        num_items=num_items,
        embed_dim=embed_dim,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    train_df = pd.read_csv(train_path)
    valid_df = pd.read_csv(valid_path)

    # 用户训练集中已交互过的 item，用于评估时过滤
    train_user_items = (
        train_df.groupby("user_idx")["movie_idx"]
        .apply(set)
        .to_dict()
    )

    recall_10 = []
    recall_20 = []
    ndcg_10 = []
    ndcg_20 = []

    with torch.no_grad():
        item_ids = torch.arange(num_items, dtype=torch.long, device=device)
        item_emb = model.item_embedding(item_ids)  # [num_items, dim]

        for _, row in tqdm(valid_df.iterrows(), total=len(valid_df), desc="Evaluating"):
            user = int(row["user_idx"])
            target_item = int(row["movie_idx"])

            user_tensor = torch.tensor([user], dtype=torch.long, device=device)
            user_emb = model.user_embedding(user_tensor)  # [1, dim]

            scores = torch.matmul(user_emb, item_emb.T).squeeze(0)  # [num_items]

            # 过滤训练集中已经交互过的 item，避免推荐“看过的电影”
            seen_items = train_user_items.get(user, set())

            if len(seen_items) > 0:
                seen_items_tensor = torch.tensor(
                    list(seen_items),
                    dtype=torch.long,
                    device=device
                )
                scores[seen_items_tensor] = -1e9

            ranked_items = torch.argsort(scores, descending=True).cpu().numpy().tolist()

            recall_10.append(recall_at_k(ranked_items, target_item, 10))
            recall_20.append(recall_at_k(ranked_items, target_item, 20))
            ndcg_10.append(ndcg_at_k(ranked_items, target_item, 10))
            ndcg_20.append(ndcg_at_k(ranked_items, target_item, 20))

    result = {
        "Recall@10": np.mean(recall_10),
        "Recall@20": np.mean(recall_20),
        "NDCG@10": np.mean(ndcg_10),
        "NDCG@20": np.mean(ndcg_20),
    }

    print("\nEvaluation Results")
    for k, v in result.items():
        print(f"{k}: {v:.6f}")

    result_path = output_dir / "eval_result.txt"
    with open(result_path, "w", encoding="utf-8") as f:
        for k, v in result.items():
            f.write(f"{k}: {v:.6f}\n")

    print(f"\nSaved results to {result_path}")


if __name__ == "__main__":
    evaluate()