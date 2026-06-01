from surprise import KNNBasic

from models_test import (
    ModelBaselineMean,
    UserBasedJaccardKNN,
    UserBasedKNN,
    ItemBasedKNN,
    ContentBased,
    SVDModel,
)


class EvalConfig:

    models = [

        # ------------------------------------------------------------------
        # 1. BASELINE
        # ------------------------------------------------------------------
        (
            "baseline_mean",
            ModelBaselineMean,
            {}
        ),

        # ------------------------------------------------------------------
        # 2. USER-BASED — Jaccard (custom, non disponible dans Surprise)
        # ------------------------------------------------------------------
        (
            "user_based_jaccard",
            UserBasedJaccardKNN,
            {
                "k": 40,
                "min_k": 3
            }
        ),

        # ------------------------------------------------------------------
        # 3. USER-BASED — KNNBaseline Pearson (Surprise)
        #    Comparaison directe avec Jaccard pour justifier le choix custom
        # ------------------------------------------------------------------
        (
            "user_based_pearson_baseline",
            UserBasedKNN,
            {
                "k": 80,
                "min_k": 3
            }
        ),

        # ------------------------------------------------------------------
        # 4. ITEM-BASED — KNNBaseline Pearson (Surprise)
        # ------------------------------------------------------------------
        (
            "item_based_pearson_baseline",
            ItemBasedKNN,
            {
                "k": 60,
                "min_k": 3
            }
        ),

        # ------------------------------------------------------------------
        # 5. CONTENT-BASED — Ridge V3 (features riches)
        #    Meilleur content-based : RMSE 0.727 sur hackathon
        # ------------------------------------------------------------------
        (
            "content_based_ridge_v3",
            ContentBased,
            {
                "features_method": "V3",
                "alpha": 24.0
            }
        ),

        # ------------------------------------------------------------------
        # 6. LATENT FACTOR — FunkSVD
        # ------------------------------------------------------------------
        (
            "svd",
            SVDModel,
            {
                "n_factors": 75,
                "n_epochs": 50,
                "lr_all": 0.005,
                "reg_all": 0.08,
                "random_state": 42
            }
        ),
    ]

    # ----------------------------------------------------------------------
    # Métriques d'évaluation
    # ----------------------------------------------------------------------

    # Métriques de prédiction (train/test split)
    split_metrics = ["rmse", "mae"]

    # Seuil pour precision@k et recall@k (rating >= threshold = "bon film")
    top_n_value = 10
    relevance_threshold = 4.0

    # Proportion du test set
    test_size = 0.25