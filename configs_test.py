"""
configs.py
==========
Configuration des modèles à évaluer via evaluator.py.

Pour lancer l'évaluation :
    python evaluator.py

Note : les modèles item-based, content-based et SVD sont commentés
temporairement pour se concentrer sur les résultats user-based.
Les décommenter quand nécessaire.
"""

from surprise import KNNBasic, KNNBaseline, SVD

from models_test import (
    ModelBaselineMean,
    UserBasedJaccardKNN,
    UserBasedITRKNN,
    UserBasedKNN,
    ItemBasedKNN,
    ItemBasedPearsonKNN,
    ContentBased,
    SVDModel,
)


class EvalConfig:

    models = [

        # ------------------------------------------------------------------
        # 1. BASELINE
        # ------------------------------------------------------------------
        #(
        #    "baseline_mean",
        #    ModelBaselineMean,
        #    {}
        #),

        # ------------------------------------------------------------------
        # 2. USER-BASED
        # Trois similarités comparées :
        #   - Jaccard  (custom, non disponible dans Surprise)
        #   - ITR      (custom, non disponible dans Surprise)
        #   - Pearson  (Surprise) avec tuning de k
        # ------------------------------------------------------------------
        #(
        #    "user_based_jaccard_k40",
        #    UserBasedJaccardKNN,
        #    {"k": 40, "min_k": 3}
        #),
        #(
        #   "user_based_itr_k40",
        #    UserBasedITRKNN,
        #    {"k": 40, "min_k": 3}
        #),
        #(
        #    "user_based_pearson_k40",
        #    UserBasedKNN,
        #    {"k": 40, "min_k": 3}
        #),
        #(
        #    "user_based_pearson_k60",
        #    UserBasedKNN,
        #    {"k": 60, "min_k": 3}
        #),
        #(
        #    "user_based_pearson_k80",
        #    UserBasedKNN,
        #    {"k": 80, "min_k": 3}
        #),
        #(
        #    "user_based_pearson_k100",
        #    UserBasedKNN,
        #    {"k": 100, "min_k": 3}
        #),

        # ------------------------------------------------------------------
        # 3. ITEM-BASED — commenté temporairement
        # ------------------------------------------------------------------
        (
            "item_based_cosine_k40",
            ItemBasedKNN,
            {"k": 40, "min_k": 3}
        ),
        
        (
            "item_based_cosine_k60",
            ItemBasedKNN,
            {"k": 60, "min_k": 3}
        ),
        
        (
            "item_based_cosine_k80",
            ItemBasedKNN,
            {"k": 80, "min_k": 3}
        ),
        
        (
            "item_based_pearson_k60",
            ItemBasedPearsonKNN,
            {"k": 60, "min_k": 3}
        ),

        (
            "item_based_cosine_k20",
            ItemBasedKNN,
            {"k": 20, "min_k": 3}
        ),
        
        (
            "item_based_cosine_k40",
            ItemBasedKNN,
            {"k": 40, "min_k": 3}
        ),
        
        (
            "item_based_cosine_k60",
            ItemBasedKNN,
            {"k": 60, "min_k": 3}
        ),
        
        (
            "item_based_cosine_k80",
            ItemBasedKNN,
            {"k": 80, "min_k": 3}
        ),
        
        (
            "item_based_cosine_k100",
            ItemBasedKNN,
            {"k": 100, "min_k": 3}
        ),
        
        (
            "item_based_pearson_k20",
            ItemBasedPearsonKNN,
            {"k": 20, "min_k": 3}
        ),
        
        (
            "item_based_pearson_k40",
            ItemBasedPearsonKNN,
            {"k": 40, "min_k": 3}
        ),
        
        (
            "item_based_pearson_k60",
            ItemBasedPearsonKNN,
            {"k": 60, "min_k": 3}
        ),
        
        (
            "item_based_pearson_k80",
            ItemBasedPearsonKNN,
            {"k": 80, "min_k": 3}
        ),
        (
            "item_based_pearson_k100",
            ItemBasedPearsonKNN,
            {"k": 100, "min_k": 3}
        ),

        # ------------------------------------------------------------------
        # 4. CONTENT-BASED — commenté temporairement
        # ------------------------------------------------------------------
        # (
        #    "content_v1",
        #    ContentBased,
        #    {"features_method": "V1", "alpha": 24.0}
        #),
        #(
        #    "content_v2",
        #    ContentBased,
        #    {"features_method": "V2", "alpha": 24.0}
        #),
        #(
        #    "content_v3_alpha10",
        #    ContentBased,
        #    {"features_method": "V3", "alpha": 10.0}
        #),
        #(
        #    "content_v3_alpha24",
        #    ContentBased,
        #    {"features_method": "V3", "alpha": 24.0}
        #),
        #(
        #    "content_v3_alpha50",
        #    ContentBased,
        #    {"features_method": "V3", "alpha": 50.0}
        #),

        #(
        #    "content_v3_alpha1",
        #    ContentBased,
        #    {"features_method": "V3", "alpha": 1.0}
        #    ),
        
        #(
        #    "content_v3_alpha5",
        #    ContentBased,
        #    {"features_method": "V3", "alpha": 5.0}
        #),
        
        #(
        #    "content_v3_alpha10",
        #    ContentBased,
        #    {"features_method": "V3", "alpha": 10.0}
        #),

        #(
        #    "content_v3_alpha01",
        #    ContentBased,
        #    {"features_method": "V3", "alpha": 0.1}
        #),

        # ------------------------------------------------------------------
        # 5. LATENT FACTOR — commenté temporairement
        # ------------------------------------------------------------------
        
        #(
        #    "svd_default",
        #    SVDModel,
        #    {
        #        "n_factors": 75,
        #        "n_epochs":  50,
        #        "lr_all":    0.005,
        #        "reg_all":   0.08,
        #        "random_state": 42
        #    }
        #),
        
        #(

        #    "svd_tuned",
        #    SVDModel,
        #    {
        #        "n_factors":    100,
        #        "n_epochs":     80,
        #        "lr_all":       0.007565017481617904,
        #        "reg_all":      0.10761653019253145,
        #        "random_state": 42
        #    }
        #),
    
    ]

    # ----------------------------------------------------------------------
    # Métriques
    # ----------------------------------------------------------------------
    split_metrics       = ["rmse", "mae"]
    top_n_value         = 10
    relevance_threshold = 4.0
    test_size           = 0.25