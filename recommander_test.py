"""
recommender.py
==============
Composant de recommandation hybride du projet MLSMM2156.

Stratégie : Weighted Ensemble (cf. Practical Recommender Systems, ch. 12.4)
---------------------------------------------------------------------------
Chaque modèle produit un score prédit pour chaque film non vu par l'utilisateur.
Le score final est une combinaison linéaire pondérée de ces scores :

    score_final(u, i) = w_svd        * score_svd(u, i)
                      + w_user       * score_user_based(u, i)
                      + w_item       * score_item_based(u, i)
                      + w_content    * score_content_based(u, i)

Les poids sont définis dans constants.py et peuvent être ajustés.
Ils ont été choisis en donnant plus de poids aux modèles ayant le meilleur
RMSE (SVD et item-based) et moins au content-based (différent des autres).

Modèles utilisés
----------------
- SVDModel          (latent factor)      — poids le plus élevé
- UserBasedKNN      (user-based CF)      — poids moyen
- ItemBasedKNN      (item-based CF)      — poids moyen
- ContentBased V3   (content-based)      — poids plus faible mais apporte
                                           diversité et cold-start

Fonctionnalités
---------------
- recommend(user_id, n)   : top-N recommandations hybrides
- explain(user_id, movie_id) : explication lisible pour l'utilisateur
- get_seen_movies(user_id)   : films déjà vus (exclus des reco)
- group_recommend(user_ids, n) : recommandations pour un groupe

Usage
-----
    from recommender import HybridRecommender

    rec = HybridRecommender(use_implicit=True)
    rec.fit()

    recs = rec.recommend(user_id=-1, n=10)
    print(recs)

    explanation = rec.explain(-1, movie_id=1)
    print(explanation)
"""

import pandas as pd
import numpy as np

from surprise import Dataset, Reader, KNNBaseline

from constants import Constant as C
from loaders import load_movies, load_ratings
from models_test import (
    UserBasedKNN,
    ItemBasedKNN,
    ContentBased,
    SVDModel,
)


# ===========================================================================
# Poids du hybrid (définis dans constants.py, modifiables)
# ===========================================================================
#
# Justification des poids :
#   - SVD          (0.40) : meilleure généralisation, capture les patterns latents
#   - User-based   (0.25) : apporte la dimension "voisinage utilisateur"
#   - Item-based   (0.25) : meilleur RMSE des KNN, similitudes entre films
#   - Content-based(0.10) : diversité et recommandations cold-start (user -1)
#
# Note : les poids sommant à 1, le score final est dans la même échelle
#        que les ratings originaux (0.5 - 5.0).

WEIGHTS = {
    "svd":          getattr(C, "HYBRID_SVD_WEIGHT",          0.40),
    "user_based":   getattr(C, "HYBRID_USER_BASED_WEIGHT",   0.25),
    "item_based":   getattr(C, "HYBRID_ITEM_BASED_WEIGHT",   0.25),
    "content_based":getattr(C, "HYBRID_CONTENT_BASED_WEIGHT",0.10),
}


# ===========================================================================
# HybridRecommender
# ===========================================================================

class HybridRecommender:
    """
    Recommandeur hybride par weighted ensemble.

    Paramètres
    ----------
    use_implicit : bool
        Si True, charge les ratings avec le profil implicite (userId=-1).
        Si False, charge uniquement les ratings MovieLens originaux.
    features_method : str
        Méthode de features pour ContentBased ("V3" recommandé).
    alpha : float
        Régularisation Ridge pour ContentBased (24 = optimal hackathon).
    """

    def __init__(
        self,
        use_implicit:    bool  = True,
        features_method: str   = "V3",
        alpha:           float = 24.0
    ):
        self.use_implicit    = use_implicit
        self.features_method = features_method
        self.alpha           = alpha

        self.trainset      = None
        self.svd           = None
        self.user_based    = None
        self.item_based    = None
        self.content_based = None
        self._is_fitted    = False

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self):
        """
        Entraîne tous les modèles sur le trainset complet.
        À appeler une fois au démarrage de l'app (offline).
        """
        print("Building trainset...")
        self.trainset = self._build_trainset()

        print("Training SVD...")
        self.svd = SVDModel()
        self.svd.fit(self.trainset)

        print("Training UserBasedKNN...")
        self.user_based = UserBasedKNN()
        self.user_based.fit(self.trainset)

        print("Training ItemBasedKNN...")
        self.item_based = ItemBasedKNN()
        self.item_based.fit(self.trainset)

        print(f"Training ContentBased ({self.features_method})...")
        self.content_based = ContentBased(
            features_method=self.features_method,
            alpha=self.alpha
        )
        self.content_based.fit(self.trainset)

        self._is_fitted = True
        print("All models trained successfully.")
        return self

    def _build_trainset(self):
        ratings = load_ratings(
            surprise_format=False,
            use_implicit=self.use_implicit
        )
        reader = Reader(rating_scale=C.RATINGS_SCALE)
        data   = Dataset.load_from_df(ratings[C.USER_ITEM_RATINGS], reader)
        return data.build_full_trainset()

    # ------------------------------------------------------------------
    # Prédiction safe
    # ------------------------------------------------------------------

    def _safe_predict(self, model, user_id, movie_id, default=3.0) -> float:
        """
        Prédit un rating de façon sécurisée.
        Retourne `default` si le modèle échoue (utilisateur/film inconnu).
        """
        uid = self._normalize_user_id(user_id)
        mid = int(movie_id)
        try:
            return float(model.predict(uid, mid).est)
        except Exception:
            try:
                return float(model.predict(str(uid), mid).est)
            except Exception:
                return default

    @staticmethod
    def _normalize_user_id(user_id):
        """Convertit les user_id de formulaire (str) en int si possible."""
        try:
            return int(user_id)
        except (TypeError, ValueError):
            return user_id

    # ------------------------------------------------------------------
    # Films déjà vus
    # ------------------------------------------------------------------

    def get_seen_movies(self, user_id) -> set:
        """Retourne l'ensemble des movieId déjà notés par l'utilisateur."""
        uid = self._normalize_user_id(user_id)
        try:
            inner_uid = self.trainset.to_inner_uid(uid)
        except ValueError:
            try:
                inner_uid = self.trainset.to_inner_uid(str(uid))
            except ValueError:
                return set()

        return {
            int(self.trainset.to_raw_iid(iid))
            for iid, _ in self.trainset.ur[inner_uid]
        }

    # ------------------------------------------------------------------
    # Recommandation individuelle
    # ------------------------------------------------------------------

    def recommend(self, user_id, n: int = 10) -> pd.DataFrame:
        """
        Génère le top-N hybride pour un utilisateur.

        Retourne un DataFrame avec les colonnes :
            movieId, title, genres,
            final_score, svd_score, user_based_score,
            item_based_score, content_based_score
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before recommend().")

        movies        = load_movies()
        seen_movie_ids = self.get_seen_movies(user_id)

        rows = []
        for _, movie in movies.iterrows():
            movie_id = int(movie[C.ITEM_ID_COL])
            if movie_id in seen_movie_ids:
                continue

            svd_score     = self._safe_predict(self.svd,           user_id, movie_id)
            user_score    = self._safe_predict(self.user_based,     user_id, movie_id)
            item_score    = self._safe_predict(self.item_based,     user_id, movie_id)
            content_score = self._safe_predict(self.content_based,  user_id, movie_id)

            final_score = (
                WEIGHTS["svd"]           * svd_score
                + WEIGHTS["user_based"]  * user_score
                + WEIGHTS["item_based"]  * item_score
                + WEIGHTS["content_based"] * content_score
            )

            rows.append({
                C.ITEM_ID_COL:        movie_id,
                C.LABEL_COL:          movie[C.LABEL_COL],
                C.GENRES_COL:         movie[C.GENRES_COL],
                "final_score":        round(final_score,   3),
                "svd_score":          round(svd_score,     3),
                "user_based_score":   round(user_score,    3),
                "item_based_score":   round(item_score,    3),
                "content_based_score":round(content_score, 3),
            })

        recs = (
            pd.DataFrame(rows)
            .sort_values("final_score", ascending=False)
            .head(n)
            .reset_index(drop=True)
        )
        return recs

    # ------------------------------------------------------------------
    # Recommandation de groupe
    # ------------------------------------------------------------------

    def group_recommend(
        self,
        user_ids: list,
        n:        int = 10,
        strategy: str = "average"
    ) -> pd.DataFrame:
        """
        Génère des recommandations pour un groupe d'utilisateurs.

        Stratégies disponibles
        ----------------------
        "average"   : moyenne des scores individuels (least misery évitée)
        "least_misery" : minimum des scores (protège les préférences minoritaires)

        Paramètres
        ----------
        user_ids : liste d'user_id
        n        : nombre de recommandations
        strategy : "average" | "least_misery"
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before group_recommend().")

        # Collecter les scores individuels (top-50 par user)
        group_scores: dict = {}

        for user_id in user_ids:
            recs = self.recommend(user_id, n=50)
            for _, row in recs.iterrows():
                mid = row[C.ITEM_ID_COL]
                if mid not in group_scores:
                    group_scores[mid] = {
                        "title":  row[C.LABEL_COL],
                        "genres": row[C.GENRES_COL],
                        "scores": []
                    }
                group_scores[mid]["scores"].append(row["final_score"])

        # Agréger selon la stratégie
        group_recs = []
        for mid, data in group_scores.items():
            scores = data["scores"]
            if strategy == "least_misery":
                agg_score = min(scores)
            else:  # average
                agg_score = sum(scores) / len(scores)

            group_recs.append({
                C.ITEM_ID_COL:  mid,
                C.LABEL_COL:    data["title"],
                C.GENRES_COL:   data["genres"],
                "group_score":  round(agg_score, 3),
                "n_users":      len(scores)
            })

        return (
            pd.DataFrame(group_recs)
            .sort_values("group_score", ascending=False)
            .head(n)
            .reset_index(drop=True)
        )

    # ------------------------------------------------------------------
    # Explication
    # ------------------------------------------------------------------

    def explain(self, user_id, movie_id: int) -> str:
        """
        Génère une explication lisible pour une recommandation.

        Délègue au ContentBased qui expose les coefficients Ridge
        (feature importances par utilisateur).
        """
        if not self._is_fitted:
            return "Model not fitted yet."
        return self.content_based.explain(user_id, movie_id)

    # ------------------------------------------------------------------
    # Infos modèle (utile pour la page évaluation de l'app)
    # ------------------------------------------------------------------

    def get_weights(self) -> dict:
        """Retourne les poids du hybrid (pour affichage dans l'app)."""
        return WEIGHTS.copy()

    def get_model_info(self) -> list:
        """
        Retourne une liste de dicts décrivant chaque modèle du hybrid.
        Utile pour la page évaluation de l'app Flask.
        """
        return [
            {
                "name":   "SVD (Latent Factor)",
                "weight": WEIGHTS["svd"],
                "desc":   "Matrix factorization — captures hidden user/item patterns."
            },
            {
                "name":   "User-Based KNN (Pearson Baseline)",
                "weight": WEIGHTS["user_based"],
                "desc":   "Collaborative filtering — recommends what similar users liked."
            },
            {
                "name":   "Item-Based KNN (Pearson Baseline)",
                "weight": WEIGHTS["item_based"],
                "desc":   "Collaborative filtering — recommends films similar to those you liked."
            },
            {
                "name":   f"Content-Based Ridge ({self.features_method})",
                "weight": WEIGHTS["content_based"],
                "desc":   "Content filtering — based on genres, tags, visual style, and ratings."
            },
        ]