"""
recommender_building.py
=======================
Script de construction offline du système de recommandation.

Tâches
------
1. IMPLICIT LIBRARY
   Convertit library_ilies.xlsx en ratings MovieLens-compatibles
   et les ajoute au dataset (userId = -1).

2. SVD TUNING avec Optuna
   Trouve les meilleurs hyperparamètres SVD et met à jour configs.py.

3. MODEL TRAINING & PICKLE
   Entraîne tous les modèles du hybrid et les sauvegarde dans recs/.

Usage
-----
    python recommender_building.py                  # tout faire
    python recommender_building.py --step implicit  # profil implicite seulement
    python recommender_building.py --step tune      # tuning SVD seulement
    python recommender_building.py --step train     # training + pickle seulement
"""

import argparse
import json
import pickle
import warnings
warnings.filterwarnings("ignore")

from datetime import datetime
from pathlib import Path

import pandas as pd

from constants import Constant as C
from loaders import load_library, load_ratings
from models_test import SVDModel, UserBasedKNN, ItemBasedKNN, ContentBased
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split


# ===========================================================================
# Configuration
# ===========================================================================

NEW_USER_ID      = -1
LIBRARY_FILENAME = "library_ilies.xlsx"

MODELS_TO_TRAIN = {
    "svd":          (SVDModel,      {"n_factors": 75, "n_epochs": 50,
                                     "lr_all": 0.005, "reg_all": 0.08,
                                     "random_state": 42}),
    "user_based":   (UserBasedKNN,  {"k": 80, "min_k": 3}),
    "item_based":   (ItemBasedKNN,  {"k": 60, "min_k": 3}),
    "content_based":(ContentBased,  {"features_method": "V3", "alpha": 24.0}),
}


# ===========================================================================
# PARTIE 1 — Implicit Library
# ===========================================================================

def compute_implicit_rating(row) -> float:
    """
    Convertit les signaux comportementaux en rating MovieLens [0.5, 5.0].

    Formule (cf. Practical Recommender Systems, ch. 4.5) :
        score = 0.5
              + 0.8 * min(n_watched, 5)
              + 0.9 * wishlist
              + 0.5 * recent
              + 1.5 * top10
    """
    score = (
        0.5
        + 0.8 * min(int(row["n_watched"]), 5)
        + 0.9 * int(row["wishlist"])
        + 0.5 * int(row["recent"])
        + 1.5 * int(row["top10"])
    )
    return round(max(C.RATINGS_SCALE[0], min(C.RATINGS_SCALE[1], score)), 2)


def load_implicit_library() -> pd.DataFrame:
    library  = load_library(LIBRARY_FILENAME)
    required = [C.ITEM_ID_COL, C.LABEL_COL, C.GENRES_COL,
                "n_watched", "wishlist", "recent", "top10"]
    missing  = [col for col in required if col not in library.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes dans {LIBRARY_FILENAME} : {missing}")
    for col in ["n_watched", "wishlist", "recent", "top10"]:
        library[col] = library[col].fillna(0).astype(int)
    return library


def build_implicit_ratings(library, user_id=NEW_USER_ID) -> pd.DataFrame:
    active = library[
        (library["n_watched"] > 0) | (library["wishlist"] == 1) |
        (library["recent"]   == 1) | (library["top10"]    == 1)
    ].copy()
    active[C.RATING_COL] = active.apply(compute_implicit_rating, axis=1)
    implicit = active[[C.ITEM_ID_COL, C.RATING_COL]].copy()
    implicit.insert(0, C.USER_ID_COL, user_id)
    implicit[C.TIMESTAMP_COL] = int(datetime.now().timestamp())
    return implicit


def append_implicit_ratings(implicit_ratings) -> tuple:
    original = pd.read_csv(C.EVIDENCE_PATH / C.RATINGS_FILENAME)
    combined = pd.concat([original, implicit_ratings], ignore_index=True)
    output   = C.EVIDENCE_PATH / C.RATINGS_WITH_IMPLICIT_FILENAME
    combined.to_csv(output, index=False)
    return combined, output


def run_implicit_library():
    print("\n" + "="*55)
    print("STEP 1 — Building implicit library")
    print("="*55)
    library          = load_implicit_library()
    implicit_ratings = build_implicit_ratings(library)
    combined, output = append_implicit_ratings(implicit_ratings)
    print(f"  userId          : {NEW_USER_ID}")
    print(f"  Ratings generated: {len(implicit_ratings)}")
    print(f"  Output file     : {output}")
    print()
    print(implicit_ratings.head(10).to_string(index=False))


# ===========================================================================
# PARTIE 2 — SVD Tuning avec Optuna
# ===========================================================================

def run_svd_tuning(n_trials=30):
    """
    Optimise les hyperparamètres SVD avec Optuna et met à jour
    les paramètres dans MODELS_TO_TRAIN.
    """
    print("\n" + "="*55)
    print("STEP 2 — SVD Tuning with Optuna")
    print("="*55)

    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("  Optuna not installed. pip install optuna")
        print("  Skipping SVD tuning — using default params.")
        return

    ratings  = load_ratings(surprise_format=False, use_implicit=False)
    reader   = Reader(rating_scale=C.RATINGS_SCALE)
    data     = Dataset.load_from_df(ratings[C.USER_ITEM_RATINGS], reader)
    trainset, testset = train_test_split(data, test_size=0.25, random_state=1)

    def objective(trial):
        params = {
            "n_factors": trial.suggest_categorical("n_factors", [20, 50, 75, 100, 150]),
            "n_epochs":  trial.suggest_int("n_epochs", 20, 80),
            "lr_all":    trial.suggest_float("lr_all",  0.001, 0.02, log=True),
            "reg_all":   trial.suggest_float("reg_all", 0.01,  0.15, log=True),
            "random_state": 42
        }
        model = SVD(**params)
        model.fit(trainset)
        preds = model.test(testset)
        return accuracy.rmse(preds, verbose=False)

    print(f"  Running {n_trials} trials...")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best        = study.best_params
    best["random_state"] = 42
    best_rmse   = round(study.best_value, 4)

    print(f"\n  Best params (RMSE={best_rmse}):")
    for k, v in best.items():
        print(f"    {k}: {v}")

    # Mettre à jour MODELS_TO_TRAIN avec les meilleurs params
    MODELS_TO_TRAIN["svd"] = (SVDModel, best)

    # Sauvegarder les meilleurs params
    C.EVALUATION_PATH.mkdir(parents=True, exist_ok=True)
    params_path = C.EVALUATION_PATH / "svd_best_params.json"
    with open(params_path, "w") as f:
        json.dump({"params": best, "rmse": best_rmse}, f, indent=2)
    print(f"\n  Saved to {params_path}")
    print("\n  Update configs.py 'svd_default' with these params for evaluator.py")


# ===========================================================================
# PARTIE 3 — Model Training & Pickle
# ===========================================================================

def build_trainset(use_implicit=True):
    ratings = load_ratings(surprise_format=False, use_implicit=use_implicit)
    reader  = Reader(rating_scale=C.RATINGS_SCALE)
    data    = Dataset.load_from_df(ratings[C.USER_ITEM_RATINGS], reader)
    return data.build_full_trainset()


def save_model(model, name: str) -> Path:
    C.RECS_PATH.mkdir(parents=True, exist_ok=True)
    path = C.RECS_PATH / f"{name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"  ✓ Saved → {path}")
    return path


def run_training(use_implicit=True):
    print("\n" + "="*55)
    print("STEP 3 — Training & saving models")
    print("="*55)

    # Charger les meilleurs params SVD si disponibles
    params_path = C.EVALUATION_PATH / "svd_best_params.json"
    if params_path.exists():
        with open(params_path) as f:
            svd_data = json.load(f)
        MODELS_TO_TRAIN["svd"] = (SVDModel, svd_data["params"])
        print(f"  Using tuned SVD params (RMSE={svd_data['rmse']})")
    else:
        print("  Using default SVD params (run --step tune for better results)")

    print("\n  Building trainset...")
    trainset = build_trainset(use_implicit=use_implicit)
    print(f"  {trainset.n_users} users | "
          f"{trainset.n_items} items | "
          f"{trainset.n_ratings} ratings")

    for name, (model_class, params) in MODELS_TO_TRAIN.items():
        print(f"\n  Training [{name}]...")
        model = model_class(**params)
        model.fit(trainset)
        save_model(model, name)

    print(f"\n  ✓ All models saved in {C.RECS_PATH}")


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--step",
        choices=["implicit", "tune", "train", "all"],
        default="all",
        help="implicit | tune | train | all (default)"
    )
    parser.add_argument(
        "--n_trials", type=int, default=30,
        help="Number of Optuna trials for SVD tuning (default: 30)"
    )
    args = parser.parse_args()

    if args.step in ("implicit", "all"):
        run_implicit_library()

    if args.step in ("tune", "all"):
        run_svd_tuning(n_trials=args.n_trials)

    if args.step in ("train", "all"):
        run_training(use_implicit=True)

    print("\n✓ recommender_building.py completed.")


if __name__ == "__main__":
    main()