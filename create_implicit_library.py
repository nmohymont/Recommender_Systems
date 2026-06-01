import pandas as pd
from pathlib import Path
from datetime import datetime


BASE_DIR = Path(__file__).resolve().parent

library_path = BASE_DIR / "library_ilies.xlsx"
output_path = BASE_DIR / "implicit_ratings_ilies.csv"

NEW_USER_ID = -1


def compute_implicit_rating(row):
    n_watched = row["n_watched"]
    wishlist = row["wishlist"]
    recent = row["recent"]
    top10 = row["top10"]

    # On limite n_watched à 5 pour éviter qu'un film vu 20 fois écrase tout
    n_watched_capped = min(n_watched, 5)

    score = (
        0.5
        + 0.8 * n_watched_capped
        + 0.9 * wishlist
        + 0.5 * recent
        + 1.5 * top10
    )

    # On force le score dans l'échelle MovieLens : 0.5 à 5.0
    score = max(0.5, min(5.0, score))

    return round(score, 2)


library = pd.read_excel(library_path)

# Remplacer les cellules vides par 0
for col in ["n_watched", "wishlist", "recent", "top10"]:
    library[col] = library[col].fillna(0).astype(int)

# Garder seulement les films où tu as donné au moins une information
active_library = library[
    (library["n_watched"] > 0)
    | (library["wishlist"] == 1)
    | (library["recent"] == 1)
    | (library["top10"] == 1)
].copy()

# Calculer les notes implicites
active_library["rating"] = active_library.apply(compute_implicit_rating, axis=1)

# Créer le format compatible MovieLens / Surprise
implicit_ratings = active_library[["movieId", "rating"]].copy()
implicit_ratings.insert(0, "userId", NEW_USER_ID)
implicit_ratings["timestamp"] = int(datetime.now().timestamp())

implicit_ratings.to_csv(output_path, index=False)

print("Fichier créé :", output_path)
print("Nombre de films utilisés pour le profil :", len(implicit_ratings))
print()
print(implicit_ratings.head(20))