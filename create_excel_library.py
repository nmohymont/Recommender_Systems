import pandas as pd
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

possible_ratings_paths = [
    BASE_DIR / "data" / "small" / "evidence" / "ratings.csv",
    PROJECT_DIR / "data" / "small" / "evidence" / "ratings.csv",
]

possible_movies_paths = [
    BASE_DIR / "data" / "small" / "content" / "movies.csv",
    PROJECT_DIR / "data" / "small" / "content" / "movies.csv",
]


def find_existing_path(possible_paths, file_name):
    for path in possible_paths:
        if path.exists():
            return path

    recursive_results = list(PROJECT_DIR.rglob(file_name))

    if len(recursive_results) > 0:
        return recursive_results[0]

    raise FileNotFoundError(f"Impossible de trouver {file_name}")


ratings_path = find_existing_path(possible_ratings_paths, "ratings.csv")
movies_path = find_existing_path(possible_movies_paths, "movies.csv")

print("Ratings path:", ratings_path)
print("Movies path:", movies_path)

ratings = pd.read_csv(ratings_path)
movies = pd.read_csv(movies_path)

movie_stats = (
    ratings
    .groupby("movieId")
    .agg(
        rating_count=("rating", "count"),
        rating_mean=("rating", "mean")
    )
    .reset_index()
)

popular_movies = movie_stats.merge(
    movies,
    on="movieId",
    how="left"
)

popular_movies = popular_movies.sort_values(
    by=["rating_count", "rating_mean"],
    ascending=[False, False]
)

library = popular_movies.head(100).copy()

library = library[
    [
        "movieId",
        "title",
        "genres",
        "rating_count",
        "rating_mean"
    ]
]

library["n_watched"] = ""
library["wishlist"] = ""
library["recent"] = ""
library["top10"] = ""

output_path = BASE_DIR / "library_ilies.xlsx"

library.to_excel(output_path, index=False)

print("Excel créé :", output_path)