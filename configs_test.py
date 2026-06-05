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

from models_test import (
    ModelBaselineMean,
    UserBasedITRKNN,
    UserBasedCosineKNN,
    UserBasedKNN,
    ItemBasedKNN,
    ItemBasedCosineKNN,
    ContentBased,
    SVDModel,
)

class EvalConfig:
    models = [
        (
        "svd_optuna_10trials",
        SVDModel,
        {"n_factors": 25, "n_epochs": 35, "lr_all": 0.005, "reg_all": 0.1008, "random_state": 42}
    ),
    (
        "svd_optuna_20trials",
        SVDModel,
        {"n_factors": 25, "n_epochs": 38, "lr_all": 0.005, "reg_all": 0.0644, "random_state": 42}
    ),
    (
        "svd_optuna_30trials",
        SVDModel,
        {"n_factors": 10, "n_epochs": 50, "lr_all": 0.005, "reg_all": 0.0653, "random_state": 42}
    ),
    (
        "svd_optuna_50trials",
        SVDModel,
        {"n_factors": 13, "n_epochs": 48, "lr_all": 0.005, "reg_all": 0.0676, "random_state": 42}
    ),
    (
        "svd_optuna_100trials",
        SVDModel,
        {"n_factors": 20, "n_epochs": 48, "lr_all": 0.005, "reg_all": 0.0705, "random_state": 42}
    ),



        # ITR union — tuning k (min_k=5 fixé)
    # ("user_itr_k20_mink5",  UserBasedITRKNN, {"k": 20,  "min_k": 5}),
    # ("user_itr_k40_mink5",  UserBasedITRKNN, {"k": 40,  "min_k": 5}),
    # ("user_itr_k60_mink5",  UserBasedITRKNN, {"k": 60,  "min_k": 5}),
    # ("user_itr_k80_mink5",  UserBasedITRKNN, {"k": 80,  "min_k": 5}),
    # ("user_itr_k100_mink5", UserBasedITRKNN, {"k": 100, "min_k": 5}),

    # Cosine KNNWithMeans — tuning k
# ("user_cosine_wm_k20", UserBasedCosineKNN, {"k": 20, "min_k": 3, "min_support": 1}),
# ("user_cosine_wm_k40", UserBasedCosineKNN, {"k": 40, "min_k": 3, "min_support": 1}),
# ("user_cosine_wm_k60", UserBasedCosineKNN, {"k": 60, "min_k": 3, "min_support": 1}),
# ("user_cosine_wm_k80", UserBasedCosineKNN, {"k": 80, "min_k": 3, "min_support": 1}),
# ("user_cosine_wm_k100", UserBasedCosineKNN, {"k": 100, "min_k": 3, "min_support": 1}),

# # Pearson KNNWithMeans — tuning k
# ("user_pearson_wm_k20", UserBasedKNN, {"k": 20, "min_k": 3, "min_support": 1}),
# ("user_pearson_wm_k40", UserBasedKNN, {"k": 40, "min_k": 3, "min_support": 1}),
# ("user_pearson_wm_k60", UserBasedKNN, {"k": 60, "min_k": 3, "min_support": 1}),
# ("user_pearson_wm_k80", UserBasedKNN, {"k": 80, "min_k": 3, "min_support": 1}),
# ("user_pearson_wm_k100", UserBasedKNN, {"k": 100, "min_k": 3, "min_support": 1}),

# Cosine KNNWithMeans — k=80 (meilleur nDCG), tuning min_k
# ("user_cosine_wm_k80_mink1",  UserBasedCosineKNN, {"k": 80, "min_k": 1,  "min_support": 1}),
# ("user_cosine_wm_k80_mink2",  UserBasedCosineKNN, {"k": 80, "min_k": 2,  "min_support": 1}),
# ("user_cosine_wm_k80_mink3",  UserBasedCosineKNN, {"k": 80, "min_k": 3,  "min_support": 1}),
# ("user_cosine_wm_k80_mink5",  UserBasedCosineKNN, {"k": 80, "min_k": 5,  "min_support": 1}),
# ("user_cosine_wm_k80_mink7",  UserBasedCosineKNN, {"k": 80, "min_k": 7,  "min_support": 1}),
# ("user_cosine_wm_k80_mink10", UserBasedCosineKNN, {"k": 80, "min_k": 10, "min_support": 1}),
# ("user_cosine_wm_k80_mink15", UserBasedCosineKNN, {"k": 80, "min_k": 15, "min_support": 1}),
# ("user_cosine_wm_k80_mink20", UserBasedCosineKNN, {"k": 80, "min_k": 20, "min_support": 1}),

# # Pearson KNNWithMeans — k=20 (meilleur nDCG), tuning min_k
# ("user_pearson_wm_k20_mink1",  UserBasedKNN, {"k": 20, "min_k": 1,  "min_support": 1}),
# ("user_pearson_wm_k20_mink2",  UserBasedKNN, {"k": 20, "min_k": 2,  "min_support": 1}),
# ("user_pearson_wm_k20_mink3",  UserBasedKNN, {"k": 20, "min_k": 3,  "min_support": 1}),
# ("user_pearson_wm_k20_mink5",  UserBasedKNN, {"k": 20, "min_k": 5,  "min_support": 1}),
# ("user_pearson_wm_k20_mink7",  UserBasedKNN, {"k": 20, "min_k": 7,  "min_support": 1}),
# ("user_pearson_wm_k20_mink10", UserBasedKNN, {"k": 20, "min_k": 10, "min_support": 1}),
# ("user_pearson_wm_k20_mink15", UserBasedKNN, {"k": 20, "min_k": 15, "min_support": 1}),
# ("user_pearson_wm_k20_mink20", UserBasedKNN, {"k": 20, "min_k": 20, "min_support": 1}), 

# Cosine KNNWithMeans — k=80, min_k=5, tuning min_support
# ("user_cosine_wm_k80_mink5_ms1",  UserBasedCosineKNN, {"k": 80, "min_k": 5, "min_support": 1}),
# ("user_cosine_wm_k80_mink5_ms2",  UserBasedCosineKNN, {"k": 80, "min_k": 5, "min_support": 2}),
# ("user_cosine_wm_k80_mink5_ms3",  UserBasedCosineKNN, {"k": 80, "min_k": 5, "min_support": 3}),
# ("user_cosine_wm_k80_mink5_ms5",  UserBasedCosineKNN, {"k": 80, "min_k": 5, "min_support": 5}),
# ("user_cosine_wm_k80_mink5_ms10", UserBasedCosineKNN, {"k": 80, "min_k": 5, "min_support": 10}),
# ("user_cosine_wm_k80_mink5_ms20", UserBasedCosineKNN, {"k": 80, "min_k": 5, "min_support": 20}),

# # Pearson KNNWithMeans — k=20, min_k=5, tuning min_support
# ("user_pearson_wm_k20_mink5_ms1",  UserBasedKNN, {"k": 20, "min_k": 5, "min_support": 1}),
# ("user_pearson_wm_k20_mink5_ms2",  UserBasedKNN, {"k": 20, "min_k": 5, "min_support": 2}),
# ("user_pearson_wm_k20_mink5_ms3",  UserBasedKNN, {"k": 20, "min_k": 5, "min_support": 3}),
# ("user_pearson_wm_k20_mink5_ms5",  UserBasedKNN, {"k": 20, "min_k": 5, "min_support": 5}),
# ("user_pearson_wm_k20_mink5_ms10", UserBasedKNN, {"k": 20, "min_k": 5, "min_support": 10}),
# ("user_pearson_wm_k20_mink5_ms20", UserBasedKNN, {"k": 20, "min_k": 5, "min_support": 20}),



    # # USER-BASED ITR — tuning min_k exhaustif
    # (
    #     "user_itr_mink1",
    #     UserBasedITRKNN,
    #     {"k": 40, "min_k": 1}
    # ),
    # (
    #     "user_itr_mink2",
    #     UserBasedITRKNN,
    #     {"k": 40, "min_k": 2}
    # ),
    # (
    #     "user_itr_mink3",
    #     UserBasedITRKNN,
    #     {"k": 40, "min_k": 3}
    # ),
    # (
    #     "user_itr_mink5",
    #     UserBasedITRKNN,
    #     {"k": 40, "min_k": 5}
    # ),
    # (
    #     "user_itr_mink7",
    #     UserBasedITRKNN,
    #     {"k": 40, "min_k": 7}
    # ),
    # (
    #     "user_itr_mink10",
    #     UserBasedITRKNN,
    #     {"k": 40, "min_k": 10}
    # ),
    # (
    #     "user_itr_mink15",
    #     UserBasedITRKNN,
    #     {"k": 40, "min_k": 15}
    # ),
    # (
    #     "user_itr_mink20",
    #     UserBasedITRKNN,
    #     {"k": 40, "min_k": 20}
    # ),

     # ------------------------------------------------------------------
        # USER-BASED ITR — custom, absent de Surprise
        # ------------------------------------------------------------------

        # (
        #     "user_itr_k40_mink3",
        #     UserBasedITRKNN,
        #     {"k": 40, "min_k": 3}
        # ),
        
        # (
        #     "user_itr_k40_mink5",
        #     UserBasedITRKNN,
        #     {"k": 40, "min_k": 5}
        # ),

        # (
        #     "user_itr_k40_mink1",
        #     UserBasedITRKNN,
        #     {"k": 40, "min_k": 1}
        # ),
        
        # (
        #     "user_itr_k40_mink3",
        #     UserBasedITRKNN,
        #     {"k": 40, "min_k": 3}
        # ),
        
        # (
        #     "user_itr_k40_mink5",
        #     UserBasedITRKNN,
        #     {"k": 40, "min_k": 5}
        # ),
        
        # (
        #     "user_itr_k20_mink3",
        #     UserBasedITRKNN,
        #     {"k": 20, "min_k": 3}
        # ),
        
        # (
        #     "user_itr_k60_mink1",
        #     UserBasedITRKNN,
        #     {"k": 60, "min_k": 1}
        # ),

        # (
        #     "user_itr_k60_mink2",
        #     UserBasedITRKNN,
        #     {"k": 60, "min_k": 2}
        # ),

        # (
        #     "user_itr_k60_mink3",
        #     UserBasedITRKNN,
        #     {"k": 60, "min_k": 3}
        # ),

        # (
        #     "user_itr_k60_mink5",
        #     UserBasedITRKNN,
        #     {"k": 60, "min_k": 5}
        # ),


        # (
        #     "user_itr_k60_mink7",
        #     UserBasedITRKNN,
        #     {"k": 60, "min_k": 7}
        # ),

        # (
        #     "user_itr_k60_mink10",
        #     UserBasedITRKNN,
        #     {"k": 60, "min_k": 10}
        # ),

        # (
        #     "user_itr_k60_mink15",
        #     UserBasedITRKNN,
        #     {"k": 60, "min_k": 15}
        # ),

        # (
        #     "user_itr_k60_mink20",
        #     UserBasedITRKNN,
        #     {"k": 60, "min_k": 20}
        # ),

        # ------------------------------------------------------------------
        # 3. ITEM-BASED 
        # ------------------------------------------------------------------
        # ------------------------------------------------------------------
        # ITEM-BASED — Cosine classique (KNNBasic)
        # ------------------------------------------------------------------
        # (
        #     "item_cosine_k20",
        #     ItemBasedCosineKNN,
        #     {"k": 20, "min_k": 3}
        # ),
        # (
        #     "item_cosine_k40",
        #     ItemBasedCosineKNN,
        #     {"k": 40, "min_k": 3}
        # ),
        # (
        #     "item_cosine_k60",
        #     ItemBasedCosineKNN,
        #     {"k": 60, "min_k": 3}
        # ),
        # (
        #     "item_cosine_k80",
        #     ItemBasedCosineKNN,
        #     {"k": 80, "min_k": 3}
        # ),
        # (
        #     "item_cosine_k100",
        #     ItemBasedCosineKNN,
        #     {"k": 100, "min_k": 3}
        # ),

        # ------------------------------------------------------------------
        # ITEM-BASED — Adjusted Cosine (KNNWithMeans)
        # ------------------------------------------------------------------
        # (
        #     "item_adjcosine_k20",
        #     ItemBasedKNN,
        #     {"k": 20, "min_k": 3}
        # ),
        # (
        #     "item_adjcosine_k40",
        #     ItemBasedKNN,
        #     {"k": 40, "min_k": 3}
        # ),
        # (
        #     "item_adjcosine_k60",
        #     ItemBasedKNN,
        #     {"k": 60, "min_k": 3}
        # ),
        # (
        #     "item_adjcosine_k80",
        #     ItemBasedKNN,
        #     {"k": 80, "min_k": 3}
        # ),
        # (
        #     "item_adjcosine_k100",
        #     ItemBasedKNN,
        #     {"k": 100, "min_k": 3}
        # ),

        # ITEM-BASED Adjusted Cosine — tuning min_k (k=60 meilleur nDCG)
    # (
    #     "item_adjcosine_k60_mink1",
    #     ItemBasedKNN,
    #     {"k": 60, "min_k": 1}
    # ),
    # (
    #     "item_adjcosine_k60_mink2",
    #     ItemBasedKNN,
    #     {"k": 60, "min_k": 2}
    # ),
    # (
    #     "item_adjcosine_k60_mink3",
    #     ItemBasedKNN,
    #     {"k": 60, "min_k": 3}
    # ),
    # (
    #     "item_adjcosine_k60_mink5",
    #     ItemBasedKNN,
    #     {"k": 60, "min_k": 5}
    # ),
    # (
    #     "item_adjcosine_k60_mink7",
    #     ItemBasedKNN,
    #     {"k": 60, "min_k": 7}
    # ),
    # (
    #     "item_adjcosine_k60_mink10",
    #     ItemBasedKNN,
    #     {"k": 60, "min_k": 10}
    # ),

    # ITEM-BASED Cosine classique — tuning min_k (k=80)
    # (
    #     "item_cosine_k80_mink1",
    #     ItemBasedCosineKNN,
    #     {"k": 80, "min_k": 1}
    # ),
    # (
    #     "item_cosine_k80_mink2",
    #     ItemBasedCosineKNN,
    #     {"k": 80, "min_k": 2}
    # ),
    # (
    #     "item_cosine_k80_mink3",
    #     ItemBasedCosineKNN,
    #     {"k": 80, "min_k": 3}
    # ),
    # (
    #     "item_cosine_k80_mink5",
    #     ItemBasedCosineKNN,
    #     {"k": 80, "min_k": 5}
    # ),
    # (
    #     "item_cosine_k80_mink7",
    #     ItemBasedCosineKNN,
    #     {"k": 80, "min_k": 7}
    # ),
    # (
    #     "item_cosine_k80_mink10",
    #     ItemBasedCosineKNN,
    #     {"k": 80, "min_k": 10}
    # ),

        # ------------------------------------------------------------------
        # 4. CONTENT-BASED — commenté temporairement
        # ------------------------------------------------------------------
        # CONTENT-BASED — V3 avec différents alpha
    # (
    #     "content_v1",
    #     ContentBased,
    #     {"features_method": "V1", "alpha": 1.0}
    # ),
    # (
    #     "content_v2",
    #     ContentBased,
    #     {"features_method": "V2", "alpha": 1.0}
    # ),
    # (
    #     "content_v3_alpha01",
    #     ContentBased,
    #     {"features_method": "V3", "alpha": 0.1}
    # ),
    # (
    #     "content_v3_alpha1",
    #     ContentBased,
    #     {"features_method": "V3", "alpha": 1.0}
    # ),
    # (
    #     "content_v3_alpha5",
    #     ContentBased,
    #     {"features_method": "V3", "alpha": 5.0}
    # ),
    # (
    #     "content_v3_alpha10",
    #     ContentBased,
    #     {"features_method": "V3", "alpha": 10.0}
    # ),
    # (
    #     "content_v3_alpha24",
    #     ContentBased,
    #     {"features_method": "V3", "alpha": 24.0}
    # ),

    # (
    # "content_v4_alpha1",
    # ContentBased,
    # {"features_method": "V4", "alpha": 1.0}
#),

        # ------------------------------------------------------------------
        # 5. LATENT FACTOR — commenté temporairement
        # ------------------------------------------------------------------

        # SVD — default vs tuned Optuna

    # (
    #     "svd_default",
    #     SVDModel,
    #     {
    #         "n_factors":    5,
    #         "n_epochs":     15,
    #         "lr_all":       0.005,
    #         "reg_all":      0.08,
    #         "random_state": 42
    #     }
    # ),
    # (
    #     "svd_tuned",
    #     SVDModel,
    #     {
    #         "n_factors":    100,
    #         "n_epochs":     80,
    #         "lr_all":       0.007565017481617904,
    #         "reg_all":      0.10761653019253145,
    #         "random_state": 42
    #     }
    # ),

    # # # SVD — tuning n_factors
    # (
    #     "svd_factors50",
    #     SVDModel,
    #     {
    #         "n_factors":    50,
    #         "n_epochs":     80,
    #         "lr_all":       0.007565017481617904,
    #         "reg_all":      0.10761653019253145,
    #         "random_state": 42
    #     }
    # ),
    # (
    #     "svd_factors150",
    #     SVDModel,
    #     {
    #         "n_factors":    150,
    #         "n_epochs":     80,
    #         "lr_all":       0.007565017481617904,
    #         "reg_all":      0.10761653019253145,
    #         "random_state": 42
    #     }
    # ),
    # (
    #     "svd_factors200",
    #     SVDModel,
    #     {
    #         "n_factors":    200,
    #         "n_epochs":     80,
    #         "lr_all":       0.007565017481617904,
    #         "reg_all":      0.10761653019253145,
    #         "random_state": 42
    #     }
    # ),
        
    #     (
    #        "svd_default",
    #        SVDModel,
    #        {
    #            "n_factors": 75,
    #            "n_epochs":  50,
    #            "lr_all":    0.005,
    #            "reg_all":   0.08,
    #            "random_state": 42
    #        }
    #     ),
        
    #     (

    #        "svd_tuned",
    #        SVDModel,
    #        {
    #            "n_factors":    100,
    #            "n_epochs":     80,
    #            "lr_all":       0.007565017481617904,
    #            "reg_all":      0.10761653019253145,
    #            "random_state": 42
    #        }
    #     ),
    
    ]

    # ----------------------------------------------------------------------
    # Métriques
    # ----------------------------------------------------------------------
    split_metrics       = ["rmse", "mae"]
    top_n_value         = 10
    relevance_threshold = 4.0
    test_size           = 0.20