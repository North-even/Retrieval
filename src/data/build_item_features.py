from pathlib import Path
import pandas as pd
import json


def main():
    project_root = Path(__file__).resolve().parents[2]

    data_dir = project_root / "data"
    output_dir = data_dir / "processed"

    movies_path = data_dir / "movies.dat"

    movies = pd.read_csv(
        movies_path,
        sep="::",
        engine="python",
        names=["movie_id", "title", "genres"],
        encoding="latin-1"
    )

    all_genres = set()

    for g in movies["genres"]:
        genre_list = g.split("|")
        all_genres.update(genre_list)

    genre2idx = {
        genre: idx
        for idx, genre in enumerate(sorted(all_genres))
    }

    print(f"num_genres:{len(genre2idx)}")

    movie_genre_dict = {}

    for _, row in movies.iterrows():
        movie_id = row["movie_id"]

        genre_ids = [
            genre2idx[g]
            for g in row["genres"].split("|")
        ]

        movie_genre_dict[int(movie_id)] = genre_ids

    with open(output_dir / "movie_genres.json", "w") as f:
        json.dump(movie_genre_dict, f)

    with open(output_dir / "genre_vocab.json", "w") as f:
        json.dump(genre2idx, f)

    print("saved movie genre features")

if __name__ == "__main__":
    main()