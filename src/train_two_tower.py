import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.data.tower_dataset import TwoTowerDataset
from src.models.two_tower import TwoTowerModel


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_dir = project_root / "data" / "processed"
    train_path = data_dir / "train.csv"
    movie_genres_path = data_dir / "movie_genres.json"
    idx2movie_path = data_dir / "idx2movie.json"

    output_dir = project_root / "outputs" / "two_tower"
    output_dir.mkdir(parents=True, exist_ok=True)

    num_users = 6038
    num_items = 3533
    num_genres = 18

    embed_dim = 64
    batch_size = 1024
    lr = 1e-3
    epochs = 10
    max_history_len = 20
    max_genre_len = 6

    train_dataset = TwoTowerDataset(
        train_csv_path=train_path,
        movie_genres_path=movie_genres_path,
        idx2movie_path=idx2movie_path,
        num_items=num_items,
        max_history_len=max_history_len,
        max_genre_len=max_genre_len,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    model = TwoTowerModel(
        num_users=num_users,
        num_items=num_items,
        num_genres=num_genres,
        embed_dim=embed_dim,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_loss = float("inf")
    best_epoch = 0

    log_path = output_dir / "train_log.txt"

    with open(log_path, "w", encoding="utf-8") as log_file:
        log_file.write(
            f"model=TwoTower\n"
            f"num_users={num_users}\n"
            f"num_items={num_items}\n"
            f"num_genres={num_genres}\n"
            f"embed_dim={embed_dim}\n"
            f"batch_size={batch_size}\n"
            f"lr={lr}\n"
            f"epochs={epochs}\n"
            f"max_history_len={max_history_len}\n"
            f"max_genre_len={max_genre_len}\n\n"
        )

        for epoch in range(1, epochs + 1):
            model.train()

            total_loss = 0.0
            total_steps = 0

            progress = tqdm(train_loader, desc=f"Epoch {epoch}")

            for (
                user,
                history,
                pos_item,
                neg_item,
                pos_genres,
                neg_genres,
            ) in progress:
                user = user.to(device)
                history = history.to(device)
                pos_item = pos_item.to(device)
                neg_item = neg_item.to(device)
                pos_genres = pos_genres.to(device)
                neg_genres = neg_genres.to(device)

                pos_score, neg_score = model(
                    user=user,
                    history=history,
                    pos_item=pos_item,
                    neg_item=neg_item,
                    pos_genres=pos_genres,
                    neg_genres=neg_genres,
                )

                loss = model.bpr_loss(pos_score, neg_score)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_steps += 1

                progress.set_postfix(loss=f"{loss.item():.4f}")

            avg_loss = total_loss / total_steps

            msg = f"epoch={epoch}, train_loss={avg_loss:.6f}"
            print(msg)
            log_file.write(msg + "\n")
            log_file.flush()

            if avg_loss < best_loss:
                best_loss = avg_loss
                best_epoch = epoch

                save_path = output_dir / "best.pt"

                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "model": "TwoTower",
                        "num_users": num_users,
                        "num_items": num_items,
                        "num_genres": num_genres,
                        "embed_dim": embed_dim,
                        "batch_size": batch_size,
                        "lr": lr,
                        "epoch": epoch,
                        "train_loss": avg_loss,
                        "max_history_len": max_history_len,
                        "max_genre_len": max_genre_len,
                    },
                    save_path,
                )

                print(f"Best model saved to {save_path}")

        summary = (
            f"\nTraining finished.\n"
            f"Best epoch: {best_epoch}\n"
            f"Best train loss: {best_loss:.6f}\n"
        )

        print(summary)
        log_file.write(summary)


if __name__ == "__main__":
    train()