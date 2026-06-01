"""
============
Pipeline d'évaluation de tous les modèles définis dans configs.py.

Métriques calculées
-------------------
- RMSE          : Root Mean Squared Error (prédiction de rating)
- MAE           : Mean Absolute Error (prédiction de rating)
- Precision@K   : fraction de recommandations pertinentes dans le top-K
- Recall@K      : fraction de films pertinents retrouvés dans le top-K
- Coverage      : proportion du catalogue recommandé au moins une fois
- Diversity     : diversité intra-liste moyenne (distance Jaccard sur genres)

Usage
-----
    python evaluator.py

Output
------
    - Tableau récapitulatif affiché dans le terminal
    - Fichier CSV exporté dans data/small/evaluation/report_YYYY_MM_DD.csv
"""

from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd

from surprise import accuracy, Dataset, Reader
from surprise.model_selection import train_test_split

from constants import Constant as C
from configs_test import EvalConfig
from loaders import load_ratings, load_items, export_evaluation_report


# ===========================================================================
# Métriques top-N
# ===========================================================================

def precision_recall_at_k(predictions, k=10, threshold=4.0):
    """
    Calcule Precision@K et Recall@K pour chaque utilisateur puis moyenne.

    Precision@K = |{items pertinents dans top-K}| / K
    Recall@K    = |{items pertinents dans top-K}| / |{items pertinents}|

    Un item est "pertinent" si son vrai rating >= threshold.
    """
    user_est_true = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions, recalls = {}, {}

    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        n_rel         = sum(1 for _, tr in user_ratings if tr >= threshold)
        n_rec_k       = sum(1 for est, _ in user_ratings[:k] if est >= threshold)
        n_rel_and_rec = sum(
            1 for est, tr in user_ratings[:k]
            if est >= threshold and tr >= threshold
        )

        precisions[uid] = n_rel_and_rec / n_rec_k if n_rec_k else 0
        recalls[uid]    = n_rel_and_rec / n_rel   if n_rel   else 0

    precision = sum(precisions.values()) / len(precisions) if precisions else 0
    recall    = sum(recalls.values())    / len(recalls)    if recalls    else 0

    return precision, recall


def get_top_n(predictions, n=10):
    """Retourne le top-N par utilisateur à partir des prédictions Surprise."""
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))
    for uid in top_n:
        top_n[uid].sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = top_n[uid][:n]
    return top_n


# ===========================================================================
# Métriques diversité / couverture
# ===========================================================================

def catalog_coverage(top_n: dict, all_item_ids: set) -> float:
    """
    Coverage = |items recommandés uniques| / |catalogue total|

    Mesure la capacité du modèle à couvrir l'ensemble du catalogue.
    Un modèle avec coverage faible recommande toujours les mêmes films.
    """
    recommended = set()
    for user_recs in top_n.values():
        for iid, _ in user_recs:
            recommended.add(int(iid))
    if not all_item_ids:
        return 0.0
    return len(recommended) / len(all_item_ids)


def genre_distance(genres_a: str, genres_b: str) -> float:
    """
    Distance de Jaccard entre deux ensembles de genres.
    0 = mêmes genres, 1 = aucun genre en commun.
    """
    set_a = set(str(genres_a).split("|"))
    set_b = set(str(genres_b).split("|"))
    union = set_a | set_b
    if not union:
        return 0.0
    return 1 - len(set_a & set_b) / len(union)


def intra_list_diversity(top_n: dict, items: pd.DataFrame) -> float:
    """
    Diversité intra-liste moyenne.

    Pour chaque utilisateur, calcule la distance moyenne entre toutes les
    paires de films recommandés (basée sur les genres). Puis moyenne globale.

    Un score proche de 1 = recommandations très diversifiées en termes de genres.
    Un score proche de 0 = recommandations très homogènes.
    """
    diversities = []

    for user_recs in top_n.values():
        movie_ids = [
            int(iid) for iid, _ in user_recs
            if int(iid) in items.index
        ]
        if len(movie_ids) < 2:
            continue

        distances = [
            genre_distance(
                items.loc[movie_ids[i], C.GENRES_COL],
                items.loc[movie_ids[j], C.GENRES_COL]
            )
            for i in range(len(movie_ids))
            for j in range(i + 1, len(movie_ids))
        ]

        if distances:
            diversities.append(np.mean(distances))

    return float(np.mean(diversities)) if diversities else 0.0


# ===========================================================================
# Évaluation d'un modèle
# ===========================================================================

def evaluate_model(model_name: str, model_class, model_params: dict) -> dict:
    """
    Entraîne et évalue un modèle sur un train/test split 75/25.

    Retourne un dict avec toutes les métriques.
    """
    # Chargement des données
    ratings = load_ratings(surprise_format=False, use_implicit=False)
    reader  = Reader(rating_scale=C.RATINGS_SCALE)
    data    = Dataset.load_from_df(ratings[C.USER_ITEM_RATINGS], reader)

    trainset, testset = train_test_split(
        data,
        test_size=EvalConfig.test_size,
        random_state=1
    )

    # Entraînement
    model = model_class(**model_params)
    model.fit(trainset)

    # Prédictions
    predictions = model.test(testset)

    # Métriques de prédiction
    rmse = accuracy.rmse(predictions, verbose=False)
    mae  = accuracy.mae(predictions,  verbose=False)

    # Métriques top-N
    precision, recall = precision_recall_at_k(
        predictions,
        k=EvalConfig.top_n_value,
        threshold=EvalConfig.relevance_threshold
    )

    top_n = get_top_n(predictions, n=EvalConfig.top_n_value)

    # Métriques diversité / couverture
    items        = load_items()
    all_item_ids = set(items.index)
    coverage     = catalog_coverage(top_n, all_item_ids)
    diversity    = intra_list_diversity(top_n, items)

    return {
        "model":                    model_name,
        "rmse":                     round(rmse,      4),
        "mae":                      round(mae,       4),
        f"precision@{EvalConfig.top_n_value}": round(precision, 4),
        f"recall@{EvalConfig.top_n_value}":    round(recall,    4),
        "coverage":                 round(coverage,  4),
        "diversity":                round(diversity, 4),
    }


# ===========================================================================
# Main
# ===========================================================================

def main():
    print("=" * 65)
    print("MLSMM2156 — Evaluation Pipeline")
    print(f"Models   : {len(EvalConfig.models)}")
    print(f"Test size: {EvalConfig.test_size * 100:.0f}%")
    print(f"Top-N    : {EvalConfig.top_n_value}")
    print(f"Threshold: {EvalConfig.relevance_threshold}")
    print("=" * 65)

    results = []

    for model_name, model_class, model_params in EvalConfig.models:
        print(f"\n→ Evaluating [{model_name}]...")
        try:
            result = evaluate_model(model_name, model_class, model_params)
            results.append(result)
            print(
                f"  RMSE={result['rmse']:.4f}  "
                f"MAE={result['mae']:.4f}  "
                f"P@{EvalConfig.top_n_value}={result[f'precision@{EvalConfig.top_n_value}']:.4f}  "
                f"R@{EvalConfig.top_n_value}={result[f'recall@{EvalConfig.top_n_value}']:.4f}  "
                f"Cov={result['coverage']:.4f}  "
                f"Div={result['diversity']:.4f}"
            )
        except Exception as e:
            print(f"  ✗ Error: {e}")

    if not results:
        print("\nNo results to display.")
        return

    report = pd.DataFrame(results)

    print("\n" + "=" * 65)
    print("RESULTS SUMMARY")
    print("=" * 65)
    print(report.to_string(index=False))

    export_evaluation_report(report)
    print(f"\nReport exported to {C.EVALUATION_PATH}/")


if __name__ == "__main__":
    main()