"""
recommender_building.py
=======================
Script de construction offline du système de recommandation.

Ce script effectue deux tâches :

1. IMPLICIT LIBRARY
   Lit le fichier library_ilies.xlsx, convertit les comportements implicites
   (n_watched, wishlist, recent, top10) en ratings MovieLens-compatibles,
   et les ajoute au fichier ratings.csv pour créer un profil personnel (userId=-1).

2. MODEL TRAINING & PICKLE
   Entraîne tous les modèles du hybrid sur le trainset complet (avec implicites)
   et les sauvegarde en fichiers pickle dans data/small/recs/.
   Ces pickles sont ensuite chargés par recommender.py pour les recommandations.

Usage
-----
    # Étape 1 uniquement (créer le profil implicite)
    python recommender_building.py --step implicit

    # Étape 2 uniquement (entraîner et sauvegarder les modèles)
    python recommender_building.py --step train

    # Les deux étapes (recommandé pour une première installation)
    python recommender_building.py
    python recommender_building.py --step all

Fichiers produits
-----------------
    data/small/evidence/ratings_with_implicit_ilies.csv
    data/small/recs/svd.pkl
    data/small/recs/user_based.pkl
    data/small/recs/item_based.pkl
    data/small/recs/content_based.pkl
"""

import argparse
import pickle
from datetime import datetime
from pathlib import Path

import pandas as pd

from constants import Constant as C
from loaders import load_library, load_ratings
from models_test import (
    SVDModel,
    UserBasedKNN,
    ItemBasedKNN,
    ContentBased,
)
from surprise import Dataset, Reader


# ===========================================================================
# Configuration
# ===========================================================================

NEW_USER_ID      = -1
LIBRARY_FILENAME = "library_ilies.xlsx"


# ===========================================================================
# PARTIE 1 — Implicit Library
# ===========================================================================

def compute_implicit_rating(row) -> float:
    """
    Convertit les signaux comportementaux implicites en rating MovieLens.

    Formule :
        score = 0.5
              + 0.8 * min(n_watched, 5)   # vu plusieurs fois = apprécié
              + 0.9 * wishlist            # veut le revoir = intérêt fort
              + 0.5 * recent             # vu récemment = mémorable
              + 1.5 * top10              # favori absolu = note maximale

    Le score est clampé dans l'échelle MovieLens [0.5, 5.0].

    Justification des poids (cf. Practical Recommender Systems, ch. 4.5) :
        - top10 a le poids le plus fort (1.5) car c'est le signal le plus fort
        - n_watched est cappé à 5 pour éviter qu'un film vu 20x écrase tout
        - wishlist et recent donnent un signal modéré mais fiable
    """
    n_watched = int(row["n_watched"])
    wishlist  = int(row["wishlist"])
    recent    = int(row["recent"])
    top10     = int(row["top10"])

    n_watched_capped = min(n_watched, 5)

    score = (
        0.5
        + 0.8 * n_watched_capped
        + 0.9 * wishlist
        + 0.5 * recent
        + 1.5 * top10
    )

    return round(
        max(C.RATINGS_SCALE[0], min(C.RATINGS_SCALE[1], score)),
        2
    )


def load_implicit_library() -> pd.DataFrame:
    """Charge et valide le fichier Excel de la library implicite."""
    library = load_library(LIBRARY_FILENAME)

    required = [C.ITEM_ID_COL, C.LABEL_COL, C.GENRES_COL,
                "n_watched", "wishlist", "recent", "top10"]
    missing = [col for col in required if col not in library.columns]

    if missing:
        raise ValueError(
            f"Colonnes manquantes dans {LIBRARY_FILENAME} : {missing}"
        )

    for col in ["n_watched", "wishlist", "recent", "top10"]:
        library[col] = library[col].fillna(0).astype(int)

    return library


def build_implicit_ratings(
    library: pd.DataFrame,
    user_id: int = NEW_USER_ID
) -> pd.DataFrame:
    """Convertit la library Excel en ratings compatibles MovieLens."""
    active = library[
        (library["n_watched"] > 0)
        | (library["wishlist"] == 1)
        | (library["recent"]   == 1)
        | (library["top10"]    == 1)
    ].copy()

    active[C.RATING_COL] = active.apply(compute_implicit_rating, axis=1)

    implicit = active[[C.ITEM_ID_COL, C.RATING_COL]].copy()
    implicit.insert(0, C.USER_ID_COL, user_id)
    implicit[C.TIMESTAMP_COL] = int(datetime.now().timestamp())

    return implicit


def append_implicit_ratings_to_movielens(
    implicit_ratings: pd.DataFrame
) -> tuple:
    """Fusionne les ratings implicites avec le fichier ratings.csv original."""
    original_path = C.EVIDENCE_PATH / C.RATINGS_FILENAME
    output_path   = C.EVIDENCE_PATH / C.RATINGS_WITH_IMPLICIT_FILENAME

    original = pd.read_csv(original_path)

    combined = pd.concat([original, implicit_ratings], ignore_index=True)
    combined.to_csv(output_path, index=False)

    return combined, output_path


def run_implicit_library():
    """Lance la construction du profil implicite complet."""
    print("\n" + "=" * 55)
    print("STEP 1 — Building implicit library")
    print("=" * 55)

    print("Loading library...")
    library = load_implicit_library()

    print("Computing implicit ratings...")
    implicit_ratings = build_implicit_ratings(library)

    print("Appending to MovieLens dataset...")
    combined, output_path = append_implicit_ratings_to_movielens(
        implicit_ratings
    )

    print()
    print(f"✓ Implicit profile created (userId={NEW_USER_ID})")
    print(f"  Ratings generated : {len(implicit_ratings)}")
    print(f"  Output file       : {output_path}")
    print()
    print(implicit_ratings.head(10).to_string(index=False))


# ===========================================================================
# PARTIE 2 — Model Training & Pickle
# ===========================================================================

MODELS_TO_TRAIN = {
    "svd": (
        SVDModel,
        {"n_factors": 75, "n_epochs": 50, "lr_all": 0.005,
         "reg_all": 0.08, "random_state": 42}
    ),
    "user_based": (
        UserBasedKNN,
        {"k": 80, "min_k": 3}
    ),
    "item_based": (
        ItemBasedKNN,
        {"k": 60, "min_k": 3}
    ),
    "content_based": (
        ContentBased,
        {"features_method": "V3", "alpha": 24.0}
    ),
}


def build_trainset(use_implicit: bool = True):
    """Construit le trainset Surprise complet."""
    ratings = load_ratings(surprise_format=False, use_implicit=use_implicit)
    reader  = Reader(rating_scale=C.RATINGS_SCALE)
    data    = Dataset.load_from_df(ratings[C.USER_ITEM_RATINGS], reader)
    return data.build_full_trainset()


def save_model(model, name: str):
    """Sauvegarde un modèle entraîné en pickle dans data/small/recs/."""
    C.RECS_PATH.mkdir(parents=True, exist_ok=True)
    path = C.RECS_PATH / f"{name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"  ✓ Saved → {path}")
    return path


def load_model(name: str):
    """Charge un modèle pickle depuis data/small/recs/."""
    path = C.RECS_PATH / f"{name}.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"Pickle not found: {path}\n"
            "Run recommender_building.py --step train first."
        )
    with open(path, "rb") as f:
        return pickle.load(f)


def run_training(use_implicit: bool = True):
    """Entraîne tous les modèles et les sauvegarde en pickle."""
    print("\n" + "=" * 55)
    print("STEP 2 — Training & saving models")
    print("=" * 55)

    print("Building trainset...")
    trainset = build_trainset(use_implicit=use_implicit)
    print(
        f"  Users : {trainset.n_users} | "
        f"Items : {trainset.n_items} | "
        f"Ratings : {trainset.n_ratings}"
    )

    for name, (model_class, params) in MODELS_TO_TRAIN.items():
        print(f"\nTraining [{name}]...")
        model = model_class(**params)
        model.fit(trainset)
        save_model(model, name)

    print("\n✓ All models saved in", C.RECS_PATH)


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Build implicit library and/or train models."
    )
    parser.add_argument(
        "--step",
        choices=["implicit", "train", "all"],
        default="all",
        help=(
            "implicit : only build the implicit library\n"
            "train    : only train and save models\n"
            "all      : both (default)"
        )
    )
    args = parser.parse_args()

    if args.step in ("implicit", "all"):
        run_implicit_library()

    if args.step in ("train", "all"):
        run_training(use_implicit=True)

    print("\n✓ recommender_building.py completed.")


if __name__ == "__main__":
    main()