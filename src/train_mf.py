import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.data.dataset import MFDataset
from src.models.mf import MFModel

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_dir = project_root / "data" / "processed"
    train_path = data_dir / "train.csv"

    output_dir = project_root / "outputs" / "mf_baseline"
    output_dir.mkdir(parents=True, exist_ok=True)

    num_users = 6038
    num_items = 3533
    embed_dim = 64
    batch_size = 1024
    lr = 1e-3
    epochs = 10

    train_dataset = MFDataset(
        csv_path=train_path,
        num_items=num_items
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    model = MFModel(
        num_users=num_users,
        num_items=num_items,
        embed_dim=embed_dim
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_loss = float("inf")

    log_path = output_dir / "train_log.txt"

    with open(log_path, "w", encoding="utf-8") as log_file:
        log_file.write(
            f"model=MF\n"
            f"num_users={num_users}\n"
            f"num_items={num_items}\n"
            f"embed_dim={embed_dim}\n"
            f"batch_size={batch_size}\n"
            f"lr={lr}\n"
            f"epochs={epochs}\n\n"
        )

        for epoch in range(1, epochs + 1):
            model.train()

            total_loss = 0.0
            total_steps = 0

            progress = tqdm(train_loader, desc=f"Epoch{epoch}")


            for user, pos_item, neg_item in progress:
                user = user.to(device)
                pos_item = pos_item.to(device)
                neg_item = neg_item.to(device)

                pos_score, neg_score = model(user, pos_item, neg_item)
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

                save_path = output_dir / "best.pt"
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "model": "MF",
                        "num_users": num_users,
                        "num_items": num_items,
                        "embed_dim": embed_dim,
                        "batch_size": batch_size,
                        "lr": lr,
                        "epoch": epoch,
                        "train_loss": avg_loss,
                    },
                    save_path
                )

                print(f"Best model saved to {save_path}")

        summary = (
            f"\nTraining finished.\n"
            f"Best train loss: {best_loss:.6f}\n"
        )

        print(summary)
        log_file.write(summary)

if __name__ == "__main__":
    train()