import sys
from pathlib import Path
import json

import faiss
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.models.two_tower import TwoTowerModel


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {int(k): v for k, v in data.items()}


def get_genres(movie_idx, idx2movie, movie_genres, max_genre_len):
    original_movie_id = idx2movie[movie_idx]
    genres = movie_genres.get(original_movie_id, [])

    genres = [g + 1 for g in genres]

    if len(genres) >= max_genre_len:
        genres = genres[:max_genre_len]
    else:
        genres = genres + [0] * (max_genre_len - len(genres))

    return genres


def build_user_histories(train_df, max_history_len):
    user_histories = (
        train_df.sort_values(["user_idx", "timestamp"])
        .groupby("user_idx")["movie_idx"]
        .apply(list)
        .to_dict()
    )

    padded = {}

    for user, history in user_histories.items():
        if len(history) >= max_history_len:
            history = history[-max_history_len:]
        else:
            history = [0] * (max_history_len - len(history)) + history

        padded[int(user)] = history

    return padded


def recall_at_k(ranked_items, target_item, k):
    return int(target_item in ranked_items[:k])


def ndcg_at_k(ranked_items, target_item, k):
    topk = ranked_items[:k]

    if target_item not in topk:
        return 0.0

    rank = topk.index(target_item)
    return 1.0 / np.log2(rank + 2)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_dir = project_root / "data" / "processed"
    output_dir = project_root / "outputs" / "two_tower"

    model_path = output_dir / "best.pt"

    train_path = data_dir / "train.csv"
    valid_path = data_dir / "valid.csv"
    movie_genres_path = data_dir / "movie_genres.json"
    idx2movie_path = data_dir / "idx2movie.json"

    checkpoint = torch.load(model_path, map_location=device)

    num_users = checkpoint["num_users"]
    num_items = checkpoint["num_items"]
    num_genres = checkpoint["num_genres"]
    embed_dim = checkpoint["embed_dim"]
    max_history_len = checkpoint["max_history_len"]
    max_genre_len = checkpoint["max_genre_len"]

    model = TwoTowerModel(
        num_users=num_users,
        num_items=num_items,
        num_genres=num_genres,
        embed_dim=embed_dim,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    train_df = pd.read_csv(train_path)
    valid_df = pd.read_csv(valid_path)

    movie_genres = load_json(movie_genres_path)
    idx2movie = load_json(idx2movie_path)

    user_histories = build_user_histories(train_df, max_history_len)

    train_user_items = (
        train_df.groupby("user_idx")["movie_idx"]
        .apply(set)
        .to_dict()
    )

    # ---------- Encode all item vectors ----------
    all_item_ids = torch.arange(num_items, dtype=torch.long, device=device)

    all_item_genres = [
        get_genres(
            movie_idx=i,
            idx2movie=idx2movie,
            movie_genres=movie_genres,
            max_genre_len=max_genre_len,
        )
        for i in range(num_items)
    ]

    all_item_genres = torch.tensor(all_item_genres, dtype=torch.long, device=device)

    with torch.no_grad():
        item_vecs = model.encode_item(
            item=all_item_ids,
            genres=all_item_genres,
        )

    item_vecs = item_vecs.detach().cpu().numpy().astype("float32")

    # ---------- Build FAISS index ----------
    # Inner product index: score = user_vec dot item_vec
    index = faiss.IndexFlatIP(embed_dim)
    index.add(item_vecs)

    print(f"FAISS index built. num_items={index.ntotal}")

    recall_10 = []
    recall_20 = []
    ndcg_10 = []
    ndcg_20 = []

    search_k = 100

    with torch.no_grad():
        for _, row in tqdm(valid_df.iterrows(), total=len(valid_df), desc="FAISS Evaluating"):
            user = int(row["user_idx"])
            target_item = int(row["movie_idx"])

            user_tensor = torch.tensor([user], dtype=torch.long, device=device)

            history = user_histories.get(user, [0] * max_history_len)
            history_tensor = torch.tensor([history], dtype=torch.long, device=device)

            user_vec = model.encode_user(
                user=user_tensor,
                history=history_tensor,
            )

            user_vec = user_vec.detach().cpu().numpy().astype("float32")

            scores, indices = index.search(user_vec, search_k)

            candidates = indices[0].tolist()

            # Filter seen items after retrieval
            seen_items = train_user_items.get(user, set())
            ranked_items = [item for item in candidates if item not in seen_items]

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

    print("\nFAISS Retrieval Results")
    for k, v in result.items():
        print(f"{k}: {v:.6f}")

    result_path = output_dir / "faiss_eval_result.txt"
    with open(result_path, "w", encoding="utf-8") as f:
        for k, v in result.items():
            f.write(f"{k}: {v:.6f}\n")

    print(f"\nSaved results to {result_path}")


if __name__ == "__main__":
    main()