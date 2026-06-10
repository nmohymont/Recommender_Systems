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
    HybridModel,
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

    # ==================================================================
        # 2. USER-BASED COLLABORATIVE FILTERING
        # ==================================================================
 
        # ------------------------------------------------------------------
        # 2a. Cosine (KNNWithMeans) — tuning k (min_k=3, min_support=1)
        # Résultat : k=80 optimal sur nDCG@10
        # ------------------------------------------------------------------
        # ("user_cosine_k20",  UserBasedCosineKNN, {"k": 20,  "min_k": 3, "min_support": 1}),
        # ("user_cosine_k40",  UserBasedCosineKNN, {"k": 40,  "min_k": 3, "min_support": 1}),
        # ("user_cosine_k60",  UserBasedCosineKNN, {"k": 60,  "min_k": 3, "min_support": 1}),
        # ("user_cosine_k80",  UserBasedCosineKNN, {"k": 80,  "min_k": 3, "min_support": 1}),
        # ("user_cosine_k100", UserBasedCosineKNN, {"k": 100, "min_k": 3, "min_support": 1}),
 
        # ------------------------------------------------------------------
        # 2b. Cosine — tuning min_k (k=80 fixé, min_support=1)
        # Résultat : min_k=5 meilleur compromis RMSE / nDCG@10
        # ------------------------------------------------------------------
        # ("user_cosine_k80_mink1",  UserBasedCosineKNN, {"k": 80, "min_k": 1,  "min_support": 1}),
        # ("user_cosine_k80_mink2",  UserBasedCosineKNN, {"k": 80, "min_k": 2,  "min_support": 1}),
        # ("user_cosine_k80_mink3",  UserBasedCosineKNN, {"k": 80, "min_k": 3,  "min_support": 1}),
        # ("user_cosine_k80_mink5",  UserBasedCosineKNN, {"k": 80, "min_k": 5,  "min_support": 1}),
        # ("user_cosine_k80_mink7",  UserBasedCosineKNN, {"k": 80, "min_k": 7,  "min_support": 1}),
        # ("user_cosine_k80_mink10", UserBasedCosineKNN, {"k": 80, "min_k": 10, "min_support": 1}),
        # ("user_cosine_k80_mink15", UserBasedCosineKNN, {"k": 80, "min_k": 15, "min_support": 1}),
        # ("user_cosine_k80_mink20", UserBasedCosineKNN, {"k": 80, "min_k": 20, "min_support": 1}),
 
        # ------------------------------------------------------------------
        # 2c. Cosine — tuning min_support (k=80, min_k=5 fixés)
        # Résultat : min_support=3 optimal (RMSE + nDCG@10)
        # ------------------------------------------------------------------
        # ("user_cosine_k80_mink5_ms1",  UserBasedCosineKNN, {"k": 80, "min_k": 5, "min_support": 1}),
        # ("user_cosine_k80_mink5_ms2",  UserBasedCosineKNN, {"k": 80, "min_k": 5, "min_support": 2}),
        # ("user_cosine_k80_mink5_ms3",  UserBasedCosineKNN, {"k": 80, "min_k": 5, "min_support": 3}),
        # ("user_cosine_k80_mink5_ms5",  UserBasedCosineKNN, {"k": 80, "min_k": 5, "min_support": 5}),
        # ("user_cosine_k80_mink5_ms10", UserBasedCosineKNN, {"k": 80, "min_k": 5, "min_support": 10}),
        # ("user_cosine_k80_mink5_ms20", UserBasedCosineKNN, {"k": 80, "min_k": 5, "min_support": 20}),
 
        # ------------------------------------------------------------------
        # 2d. Pearson (KNNWithMeans) — tuning k (min_k=3, min_support=1)
        # Résultat : k=20 optimal sur nDCG@10
        # ------------------------------------------------------------------
        # ("user_pearson_k20",  UserBasedKNN, {"k": 20,  "min_k": 3, "min_support": 1}),
        # ("user_pearson_k40",  UserBasedKNN, {"k": 40,  "min_k": 3, "min_support": 1}),
        # ("user_pearson_k60",  UserBasedKNN, {"k": 60,  "min_k": 3, "min_support": 1}),
        # ("user_pearson_k80",  UserBasedKNN, {"k": 80,  "min_k": 3, "min_support": 1}),
        # ("user_pearson_k100", UserBasedKNN, {"k": 100, "min_k": 3, "min_support": 1}),
 
        # ------------------------------------------------------------------
        # 2e. Pearson — tuning min_k (k=20 fixé, min_support=1)
        # Résultat : min_k=5 meilleur compromis
        # ------------------------------------------------------------------
        # ("user_pearson_k20_mink1",  UserBasedKNN, {"k": 20, "min_k": 1,  "min_support": 1}),
        # ("user_pearson_k20_mink2",  UserBasedKNN, {"k": 20, "min_k": 2,  "min_support": 1}),
        # ("user_pearson_k20_mink3",  UserBasedKNN, {"k": 20, "min_k": 3,  "min_support": 1}),
        # ("user_pearson_k20_mink5",  UserBasedKNN, {"k": 20, "min_k": 5,  "min_support": 1}),
        # ("user_pearson_k20_mink7",  UserBasedKNN, {"k": 20, "min_k": 7,  "min_support": 1}),
        # ("user_pearson_k20_mink10", UserBasedKNN, {"k": 20, "min_k": 10, "min_support": 1}),
        # ("user_pearson_k20_mink15", UserBasedKNN, {"k": 20, "min_k": 15, "min_support": 1}),
        # ("user_pearson_k20_mink20", UserBasedKNN, {"k": 20, "min_k": 20, "min_support": 1}),
 
        # ------------------------------------------------------------------
        # 2f. Pearson — tuning min_support (k=20, min_k=5 fixés)
        # Résultat : min_support=3 optimal
        # ------------------------------------------------------------------
        # ("user_pearson_k20_mink5_ms1",  UserBasedKNN, {"k": 20, "min_k": 5, "min_support": 1}),
        # ("user_pearson_k20_mink5_ms2",  UserBasedKNN, {"k": 20, "min_k": 5, "min_support": 2}),
        # ("user_pearson_k20_mink5_ms3",  UserBasedKNN, {"k": 20, "min_k": 5, "min_support": 3}),
        # ("user_pearson_k20_mink5_ms5",  UserBasedKNN, {"k": 20, "min_k": 5, "min_support": 5}),
        # ("user_pearson_k20_mink5_ms10", UserBasedKNN, {"k": 20, "min_k": 5, "min_support": 10}),
        # ("user_pearson_k20_mink5_ms20", UserBasedKNN, {"k": 20, "min_k": 5, "min_support": 20}),
 
        # ------------------------------------------------------------------
        # 2g. ITR (custom — non disponible dans Surprise) — tuning k (min_k=5)
        # Résultat : k=60 optimal sur nDCG@10 et RMSE
        # ------------------------------------------------------------------
        # ("user_itr_k20_mink5",  UserBasedITRKNN, {"k": 20,  "min_k": 5}),
        # ("user_itr_k40_mink5",  UserBasedITRKNN, {"k": 40,  "min_k": 5}),
        # ("user_itr_k60_mink5",  UserBasedITRKNN, {"k": 60,  "min_k": 5}),
        # ("user_itr_k80_mink5",  UserBasedITRKNN, {"k": 80,  "min_k": 5}),
        # ("user_itr_k100_mink5", UserBasedITRKNN, {"k": 100, "min_k": 5}),
 
        # ------------------------------------------------------------------
        # 2h. ITR — tuning min_k (k=60 fixé)
        # Résultat : min_k=5 meilleur compromis RMSE / nDCG@10
        # ------------------------------------------------------------------
        # ("user_itr_k60_mink1",  UserBasedITRKNN, {"k": 60, "min_k": 1}),
        # ("user_itr_k60_mink2",  UserBasedITRKNN, {"k": 60, "min_k": 2}),
        # ("user_itr_k60_mink3",  UserBasedITRKNN, {"k": 60, "min_k": 3}),
        # ("user_itr_k60_mink5",  UserBasedITRKNN, {"k": 60, "min_k": 5}),
        # ("user_itr_k60_mink7",  UserBasedITRKNN, {"k": 60, "min_k": 7}),
        # ("user_itr_k60_mink10", UserBasedITRKNN, {"k": 60, "min_k": 10}),
        # ("user_itr_k60_mink15", UserBasedITRKNN, {"k": 60, "min_k": 15}),
        # ("user_itr_k60_mink20", UserBasedITRKNN, {"k": 60, "min_k": 20}),
 
        # ------------------------------------------------------------------
        # Configurations finales retenues (rapport — Table 2 & 6)
        # ------------------------------------------------------------------
        # ("user_cosine_best",  UserBasedCosineKNN, {"k": 80, "min_k": 5, "min_support": 3}),
        # ("user_pearson_best", UserBasedKNN,       {"k": 20, "min_k": 5, "min_support": 3}),
        # ("user_itr_best",     UserBasedITRKNN,    {"k": 60, "min_k": 5}),
 
 
        # ==================================================================
        # 3. ITEM-BASED COLLABORATIVE FILTERING
        # ==================================================================
 
        # ------------------------------------------------------------------
        # 3a. Adjusted Cosine (KNNWithMeans) — tuning k (min_k=3)
        # Résultat : k=100 optimal (RMSE + novelty), k=60 pic nDCG@10
        # ------------------------------------------------------------------
        # ("item_adjcosine_k20",  ItemBasedKNN, {"k": 20,  "min_k": 3}),
        # ("item_adjcosine_k40",  ItemBasedKNN, {"k": 40,  "min_k": 3}),
        # ("item_adjcosine_k60",  ItemBasedKNN, {"k": 60,  "min_k": 3}),
        # ("item_adjcosine_k80",  ItemBasedKNN, {"k": 80,  "min_k": 3}),
        # ("item_adjcosine_k100", ItemBasedKNN, {"k": 100, "min_k": 3}),
 
        # ------------------------------------------------------------------
        # 3b. Adjusted Cosine — tuning min_k (k=100 fixé)
        # Résultat : impact négligeable — stabilité item-item matrix
        # ------------------------------------------------------------------
        # ("item_adjcosine_k100_mink1",  ItemBasedKNN, {"k": 100, "min_k": 1}),
        # ("item_adjcosine_k100_mink2",  ItemBasedKNN, {"k": 100, "min_k": 2}),
        # ("item_adjcosine_k100_mink3",  ItemBasedKNN, {"k": 100, "min_k": 3}),
        # ("item_adjcosine_k100_mink5",  ItemBasedKNN, {"k": 100, "min_k": 5}),
        # ("item_adjcosine_k100_mink7",  ItemBasedKNN, {"k": 100, "min_k": 7}),
        # ("item_adjcosine_k100_mink10", ItemBasedKNN, {"k": 100, "min_k": 10}),
 
        # ------------------------------------------------------------------
        # Configuration finale retenue (rapport — Table 7 & 8)
        # ------------------------------------------------------------------
        # ("item_adjcosine_best", ItemBasedKNN, {"k": 100, "min_k": 3}),
 
 
        # ==================================================================
        # 4. CONTENT-BASED FILTERING (Ridge par utilisateur)
        # ==================================================================
 
        # ------------------------------------------------------------------
        # 4a. V3 (metadata + TF-IDF tags + genome + visual + rating signals)
        # Tuning alpha — Résultat : alpha=1.0 optimal (RMSE + nDCG@10)
        # ------------------------------------------------------------------
        # ("content_v3_alpha001", ContentBased, {"features_method": "V3", "alpha": 0.01}),
        # ("content_v3_alpha01",  ContentBased, {"features_method": "V3", "alpha": 0.1}),
        # ("content_v3_alpha05",  ContentBased, {"features_method": "V3", "alpha": 0.5}),
        # ("content_v3_alpha1",   ContentBased, {"features_method": "V3", "alpha": 1.0}),
        # ("content_v3_alpha5",   ContentBased, {"features_method": "V3", "alpha": 5.0}),
        # ("content_v3_alpha10",  ContentBased, {"features_method": "V3", "alpha": 10.0}),
        # ("content_v3_alpha25",  ContentBased, {"features_method": "V3", "alpha": 25.0}),
 
        # ------------------------------------------------------------------
        # 4b. V4 (V3 + LDA topics sur overviews TMDB)
        # Tuning alpha — Résultat : alpha=1.0 retenu (diversité + nDCG@10)
        # ------------------------------------------------------------------
        # ("content_v4_alpha001", ContentBased, {"features_method": "V4", "alpha": 0.01}),
        # ("content_v4_alpha01",  ContentBased, {"features_method": "V4", "alpha": 0.1}),
        # ("content_v4_alpha05",  ContentBased, {"features_method": "V4", "alpha": 0.5}),
        # ("content_v4_alpha1",   ContentBased, {"features_method": "V4", "alpha": 1.0}),
        # ("content_v4_alpha5",   ContentBased, {"features_method": "V4", "alpha": 5.0}),
        # ("content_v4_alpha10",  ContentBased, {"features_method": "V4", "alpha": 10.0}),
        # ("content_v4_alpha25",  ContentBased, {"features_method": "V4", "alpha": 25.0}),
 
        # ------------------------------------------------------------------
        # 4c. V5 (V3 + BERT embeddings sentence-transformers sur overviews)
        # Tuning alpha — Résultat : alpha=25.0 retenu (maximise novelty)
        # ------------------------------------------------------------------
        # ("content_v5_alpha001", ContentBased, {"features_method": "V5", "alpha": 0.01}),
        # ("content_v5_alpha01",  ContentBased, {"features_method": "V5", "alpha": 0.1}),
        # ("content_v5_alpha05",  ContentBased, {"features_method": "V5", "alpha": 0.5}),
        # ("content_v5_alpha1",   ContentBased, {"features_method": "V5", "alpha": 1.0}),
        # ("content_v5_alpha5",   ContentBased, {"features_method": "V5", "alpha": 5.0}),
        # ("content_v5_alpha10",  ContentBased, {"features_method": "V5", "alpha": 10.0}),
        # ("content_v5_alpha25",  ContentBased, {"features_method": "V5", "alpha": 25.0}),
 
        # ------------------------------------------------------------------
        # Configurations finales retenues (rapport — Table 10 & 11)
        # ------------------------------------------------------------------
        # ("content_v3_best", ContentBased, {"features_method": "V3", "alpha": 1.0}),
        # ("content_v4_best", ContentBased, {"features_method": "V4", "alpha": 1.0}),
        # ("content_v5_best", ContentBased, {"features_method": "V5", "alpha": 25.0}),
 
 
        # ==================================================================
        # 5. LATENT FACTOR — FunkSVD (Optuna tuning)
        # ==================================================================
        # Lancer : python evaluator.py --tune-svd --n-trials 40
        # Résultat optimal (40 trials) : n_factors=35, n_epochs=98, reg_all=0.0934
        # nDCG@10=0.8155, RMSE=0.8777
        # ------------------------------------------------------------------
        # ("svd_10trials",  SVDModel, {"n_factors": 27, "n_epochs": 80,  "reg_all": 0.085,  "random_state": 42}),
        # ("svd_20trials",  SVDModel, {"n_factors": 47, "n_epochs": 92,  "reg_all": 0.1065, "random_state": 42}),
        # ("svd_30trials",  SVDModel, {"n_factors": 35, "n_epochs": 89,  "reg_all": 0.1147, "random_state": 42}),
        # ("svd_40trials",  SVDModel, {"n_factors": 35, "n_epochs": 98,  "reg_all": 0.0934, "random_state": 42}),
        # ("svd_50trials",  SVDModel, {"n_factors": 53, "n_epochs": 81,  "reg_all": 0.0836, "random_state": 42}),
        # ("svd_100trials", SVDModel, {"n_factors": 44, "n_epochs": 79,  "reg_all": 0.0976, "random_state": 42}),
 
        # ------------------------------------------------------------------
        # Configuration finale retenue (rapport — Table 9)
        # ------------------------------------------------------------------
        # ("svd_best", SVDModel, {"n_factors": 35, "n_epochs": 98, "reg_all": 0.0934, "random_state": 42}),
 
 
        # ==================================================================
        # 6. HYBRID MODEL — combinaison pondérée SVD + Content V3 + ITR
        # Poids retenus : w_svd=0.45, w_content=0.30, w_user=0.25
        # Justification : SVD meilleur nDCG@10, V3 meilleur RMSE + cold-start,
        #                 ITR meilleure diversité
        # ==================================================================
        # ("hybrid_040_030_030", HybridModel, {"w_svd": 0.40, "w_content": 0.30, "w_user": 0.30}),
        # ("hybrid_045_025_030", HybridModel, {"w_svd": 0.45, "w_content": 0.25, "w_user": 0.30}),
        # ("hybrid_050_025_025", HybridModel, {"w_svd": 0.50, "w_content": 0.25, "w_user": 0.25}),
 
        # ------------------------------------------------------------------
        # Configuration finale retenue (rapport — Table 12)
        # ------------------------------------------------------------------
        ("hybrid_best", HybridModel, {"w_svd": 0.45, "w_content": 0.30, "w_user": 0.25}),
       
    ]

    # ----------------------------------------------------------------------
    # Métriques
    # ----------------------------------------------------------------------
    split_metrics       = ["rmse", "mae"]
    top_n_value         = 10
    relevance_threshold = 4.0
    test_size           = 0.20