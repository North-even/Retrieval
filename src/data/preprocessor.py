from pathlib import Path

import pandas as pd


def load_ratings(data_dir: Path) -> pd.DataFrame:
    rating_path = data_dir / "ratings.dat"

    ratings = pd.read_csv(
        rating_path,
        sep="::",
        engine="python",
        names=["user_id", "movie_id", "rating", "timestamp"],
        encoding="latin-1",
    )

    return ratings

def build_implicit_feedback(ratings: pd.DataFrame, min_rating: int = 4) -> pd.DataFrame:
    positives = ratings[ratings["rating"] >= min_rating].copy()

    positives = positives[["user_id", "movie_id", "timestamp"]]
    positives = positives.sort_values(["user_id", "timestamp"])

    return positives

def encode_ids(df: pd.DataFrame):
    user_ids = sorted(df["user_id"].unique())
    movie_ids = sorted(df["movie_id"].unique())

    user2idx = {u: i for i, u in enumerate(user_ids)}
    movie2idx = {m: i for i, m in enumerate(movie_ids)}

    df["user_idx"] = df["user_id"].map(user2idx)
    df["movie_idx"] = df["movie_id"].map(movie2idx)

    return df, user2idx, movie2idx

def split_by_user_time(df: pd.DataFrame):
    train_list = []
    valid_list = []
    test_list = []

    for user_idx, group in df.groupby("user_idx"):
        group = group.sort_values("timestamp")

        if len(group) < 3:
            train_list.append(group)
            continue

        train_list.append(group.iloc[:-2])
        valid_list.append(group.iloc[-2:-1])
        test_list.append(group.iloc[-1:])

    train_df = pd.concat(train_list).reset_index(drop=True)
    valid_df = pd.concat(valid_list).reset_index(drop=True)
    test_df = pd.concat(test_list).reset_index(drop=True)

    return train_df, valid_df, test_df

def main():
    project_root = Path(__file__).resolve().parents[2]
    data_dir = project_root / "data"
    output_dir = project_root / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    ratings = load_ratings(data_dir)
    positives = build_implicit_feedback(ratings, min_rating=4)
    positives, user2idx, movie2idx = encode_ids(positives)

    train_df, valid_df, test_df = split_by_user_time(positives)

    train_df.to_csv(output_dir / "train.csv", index=False)
    valid_df.to_csv(output_dir / "valid.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)

    print("Preprocessing finished.")
    print(f"num_users: {len(user2idx)}")
    print(f"num_items: {len(movie2idx)}")
    print(f"num_positive_interactions: {len(positives)}")
    print(f"train: {len(train_df)}")
    print(f"valid: {len(valid_df)}")
    print(f"test: {len(test_df)}")


if __name__ == "__main__":
    main()