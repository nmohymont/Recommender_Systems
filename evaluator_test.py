"""
evaluator_test.py
=================
Pipeline d'évaluation de tous les modèles définis dans configs_test.py.

Métriques calculées
-------------------
- RMSE      : Root Mean Squared Error (prédiction de rating)
- MAE       : Mean Absolute Error (prédiction de rating)
- nDCG@K    : Normalized Discounted Cumulative Gain (qualité du ranking)
- Diversity : diversité intra-liste moyenne (distance Jaccard sur genres)
- Novelty   : nouveauté moyenne des recommandations (basée sur la popularité)

Usage
-----
    python evaluator_test.py                    # évaluation standard
    python evaluator_test.py --tune-svd         # tuning Optuna SVD avant évaluation
    python evaluator_test.py --tune-svd --n-trials 50
"""

from collections import defaultdict
from datetime import datetime

import argparse
import numpy as np
import pandas as pd
import json


from surprise import accuracy, Dataset, Reader, SVD
from surprise.model_selection import train_test_split

from constants import Constant as C
from configs_test import EvalConfig
from loaders import load_ratings, load_items, export_evaluation_report


# ===========================================================================
# nDCG@K
# ===========================================================================

def ndcg_at_k(predictions, k=10, threshold=4.0):
    """
    Normalized Discounted Cumulative Gain @K.

    DCG@K  = Σ (2^rel_i - 1) / log2(i+1)
    nDCG@K = DCG@K / IDCG@K  (normalisé entre 0 et 1)
    """
    user_est_true = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    ndcg_scores = []
    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        top_k = user_ratings[:k]
        dcg = sum(
            (1 if true_r >= threshold else 0) / np.log2(i + 2)
            for i, (_, true_r) in enumerate(top_k)
        )
        ideal = sorted(user_ratings, key=lambda x: x[1], reverse=True)[:k]
        idcg = sum(
            (1 if true_r >= threshold else 0) / np.log2(i + 2)
            for i, (_, true_r) in enumerate(ideal)
        )
        ndcg_scores.append(dcg / idcg if idcg > 0 else 0.0)

    return float(np.mean(ndcg_scores)) if ndcg_scores else 0.0


# ===========================================================================
# Diversity
# ===========================================================================

def get_top_n(predictions, n=10):
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))
    for uid in top_n:
        top_n[uid].sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = top_n[uid][:n]
    return top_n


def genre_distance(genres_a: str, genres_b: str) -> float:
    set_a = set(str(genres_a).split("|"))
    set_b = set(str(genres_b).split("|"))
    union = set_a | set_b
    if not union:
        return 0.0
    return 1 - len(set_a & set_b) / len(union)


def intra_list_diversity(top_n: dict, items: pd.DataFrame) -> float:
    """
    Diversité intra-liste moyenne.
    Distance Jaccard moyenne entre toutes les paires de films recommandés.
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
# Novelty
# ===========================================================================

def novelty(top_n: dict, item_popularity: dict) -> float:
    """
    Nouveauté moyenne : novelty(i) = -log2(popularité(i))
    """
    scores = []
    for user_recs in top_n.values():
        user_novelty = []
        for iid, _ in user_recs:
            pop = item_popularity.get(int(iid), 1e-10)
            user_novelty.append(-np.log2(pop))
        if user_novelty:
            scores.append(np.mean(user_novelty))
    return float(np.mean(scores)) if scores else 0.0


def compute_item_popularity(ratings: pd.DataFrame) -> dict:
    """Popularité = proportion d'utilisateurs ayant noté le film."""
    n_users = ratings[C.USER_ID_COL].nunique()
    counts  = ratings.groupby(C.ITEM_ID_COL)[C.USER_ID_COL].count()
    return (counts / n_users).to_dict()


# ===========================================================================
# Optuna SVD Tuning
# ===========================================================================

def tune_svd_optuna(n_trials: int = 30) -> dict:
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("  ✗ Optuna not installed. Run: pip install optuna")
        return {}

    import json

    print("\n" + "=" * 65)
    print("SVD Hyperparameter Tuning — Optuna")
    print(f"Trials   : {n_trials}")
    print(f"Split    : {EvalConfig.test_size * 100:.0f}% test")
    print("=" * 65)

    ratings  = load_ratings(surprise_format=False, use_implicit=False)
    reader   = Reader(rating_scale=C.RATINGS_SCALE)
    data     = Dataset.load_from_df(ratings[C.USER_ITEM_RATINGS], reader)
    trainset, testset = train_test_split(
        data, test_size=EvalConfig.test_size, random_state=1
    )

    def objective(trial):
        # On applique les contraintes demandées
        params = {
            # On cherche uniquement des petits facteurs (explicabilité préservée)
            "n_factors": trial.suggest_int("n_factors", 5, 55),
            
            # On laisse Optuna trouver le bon moment pour s'arrêter
            "n_epochs":  trial.suggest_int("n_epochs", 15, 100),
            
            "lr_all":    0.005,
            # On ajuste la régularisation globale
            "reg_all":   trial.suggest_float("reg_all", 0.02, 0.15, log=True),
            "random_state": 42,
        }
        model = SVD(**params)
        model.fit(trainset)
        preds = model.test(testset)
        return accuracy.rmse(preds, verbose=False)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best      = study.best_params
    best["random_state"] = 42
    best_rmse = round(study.best_value, 4)

    print(f"\n  Best params (RMSE={best_rmse}):")
    for k, v in best.items():
        if k != "random_state":
            print(f"    {k}: {v}")

    C.EVALUATION_PATH.mkdir(parents=True, exist_ok=True)
    params_path = C.EVALUATION_PATH / "svd_best_params.json"
    with open(params_path, "w") as f:
        json.dump({"params": best, "rmse": best_rmse}, f, indent=2)
    print(f"\n  Saved → {params_path}")

    return best

# ===========================================================================
# Évaluation d'un modèle
# ===========================================================================

def evaluate_model(model_name: str, model_class, model_params: dict) -> dict:
    ratings = load_ratings(surprise_format=False, use_implicit=False)
    reader  = Reader(rating_scale=C.RATINGS_SCALE)
    data    = Dataset.load_from_df(ratings[C.USER_ITEM_RATINGS], reader)
 
    trainset, testset = train_test_split(
        data,
        test_size=EvalConfig.test_size,
        random_state=1
    )
 
    model = model_class(**model_params)
    model.fit(trainset)
    predictions = model.test(testset)
 
    rmse = accuracy.rmse(predictions, verbose=False)
    mae  = accuracy.mae(predictions,  verbose=False)
 
    ndcg = ndcg_at_k(
        predictions,
        k=EvalConfig.top_n_value,
        threshold=EvalConfig.relevance_threshold
    )
 
    top_n           = get_top_n(predictions, n=EvalConfig.top_n_value)
    items           = load_items()
    item_popularity = compute_item_popularity(ratings)
    diversity       = intra_list_diversity(top_n, items)
    nov             = novelty(top_n, item_popularity)
 
    return {
        "model":     model_name,
        "rmse":      round(rmse,      4),
        "mae":       round(mae,       4),
        f"ndcg@{EvalConfig.top_n_value}": round(ndcg, 4),
        "diversity": round(diversity, 4),
        "novelty":   round(nov,       4),
    }
 
 


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune-svd",  action="store_true",
                        help="Run Optuna SVD tuning before evaluation")
    parser.add_argument("--n-trials", type=int, default=30,
                        help="Number of Optuna trials (default: 30)")
    args = parser.parse_args()

    # Optuna SVD tuning si demandé
    if args.tune_svd:
        best_params = tune_svd_optuna(n_trials=args.n_trials)
        
    params_path = C.EVALUATION_PATH / "svd_best_params.json"
    loaded_params = {}
    
    if params_path.exists():
        try:
            with open(params_path, "r") as f:
                saved_data = json.load(f)
                # On récupère le sous-dictionnaire "params" créé par ta fonction de tuning
                loaded_params = saved_data.get("params", {})
            print(f"-> Successfully loaded tuned SVD parameters from JSON.")
        except Exception as e:
            print(f"-> Warning: Could not read JSON parameters ({e}). Using defaults.")
    else:
        print("-> No tuned parameters found. Running SVD with library defaults.")

    # Évaluation standard
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

        # Si c'est le modèle d'optuna et que des paramètres ont été chargés, on les injecte
        if model_name == "svd_optuna_tuning" and loaded_params:
            model_params = loaded_params

        try:
            result = evaluate_model(model_name, model_class, model_params)
            results.append(result)
            print(
                f"  RMSE={result['rmse']:.4f}  "
                f"MAE={result['mae']:.4f}  "
                f"nDCG@{EvalConfig.top_n_value}={result[f'ndcg@{EvalConfig.top_n_value}']:.4f}  "
                f"Div={result['diversity']:.4f}  "
                f"Nov={result['novelty']:.4f}"
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