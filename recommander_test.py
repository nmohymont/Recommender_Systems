"""
recommander_test.py
===================
Composant de recommandation hybride — version app.

Modèles utilisés (rapides, performants) :
    - SVDModel       RMSE 0.887  poids 0.50
    - ItemBasedKNN   RMSE 0.876  poids 0.50

Le ContentBased V3 (RMSE 0.894) est évalué offline (evaluator.py)
mais n'est pas chargé dans l'app car trop lent au démarrage (~15 min).

Démarrage app : ~60 secondes.
"""

import pickle
import pandas as pd
import numpy as np

from surprise import Dataset, Reader

from constants import Constant as C
from loaders import load_movies, load_ratings
from models_test import SVDModel, ItemBasedKNN, ContentBased


# Poids du hybrid
W_SVD  = 0.50
W_ITEM = 0.50


class HybridRecommender:
    """
    Recommandeur hybride SVD + ItemBased.

    Paramètres
    ----------
    use_implicit : bool
        Si True, charge ratings_with_implicit_ilies.csv (inclut userId=-1).
    """

    def __init__(self, use_implicit: bool = True):
        self.use_implicit = use_implicit
        self.trainset     = None
        self.svd          = None
        self.item_based   = None
        self._is_fitted   = False

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self):
        """Entraîne les modèles sur le trainset complet."""
        print("Building trainset...")
        self.trainset = self._build_trainset()
        print(f"  {self.trainset.n_users} users | "
              f"{self.trainset.n_items} items | "
              f"{self.trainset.n_ratings} ratings")

        print("Training SVD...")
        self.svd = SVDModel()
        self.svd.fit(self.trainset)
        print("  ✓ SVD ready")

        print("Training ItemBasedKNN...")
        self.item_based = ItemBasedKNN()
        self.item_based.fit(self.trainset)
        print("  ✓ ItemBasedKNN ready")

        self._is_fitted = True
        print("Hybrid recommender ready.")
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
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_uid(user_id):
        try:
            return int(user_id)
        except (TypeError, ValueError):
            return user_id

    def _safe_predict(self, model, user_id, movie_id, default=3.0) -> float:
        uid = self._normalize_uid(user_id)
        mid = int(movie_id)
        for u in [uid, str(uid)]:
            try:
                return float(model.predict(u, mid).est)
            except Exception:
                pass
        return default

    def get_seen_movies(self, user_id) -> set:
        uid = self._normalize_uid(user_id)
        for u in [uid, str(uid)]:
            try:
                inner = self.trainset.to_inner_uid(u)
                return {
                    int(self.trainset.to_raw_iid(iid))
                    for iid, _ in self.trainset.ur[inner]
                }
            except ValueError:
                pass
        return set()

    # ------------------------------------------------------------------
    # Recommandation individuelle
    # ------------------------------------------------------------------

    def recommend(self, user_id, n: int = 10) -> pd.DataFrame:
        """Top-N hybride pour un utilisateur existant."""
        if not self._is_fitted:
            raise RuntimeError("Call fit() first.")

        movies         = load_movies()
        seen           = self.get_seen_movies(user_id)
        rows           = []

        for _, movie in movies.iterrows():
            mid = int(movie[C.ITEM_ID_COL])
            if mid in seen:
                continue

            svd_score  = self._safe_predict(self.svd,        user_id, mid)
            item_score = self._safe_predict(self.item_based,  user_id, mid)
            final      = W_SVD * svd_score + W_ITEM * item_score

            rows.append({
                C.ITEM_ID_COL:   mid,
                C.LABEL_COL:     movie[C.LABEL_COL],
                C.GENRES_COL:    movie[C.GENRES_COL],
                "final_score":   round(final,      3),
                "svd_score":     round(svd_score,  3),
                "item_score":    round(item_score, 3),
            })

        return (
            pd.DataFrame(rows)
            .sort_values("final_score", ascending=False)
            .head(n)
            .reset_index(drop=True)
        )

    def recommend_new_user(
        self,
        movie_ratings: dict,
        n: int = 10
    ) -> pd.DataFrame:
        """
        Recommandations pour un nouvel utilisateur non présent dans le trainset.

        Stratégie : content-based léger (similarité cosine sur genres)
        à partir des films notés par l'utilisateur.

        Paramètres
        ----------
        movie_ratings : dict {movieId: rating}
        """
        movies    = load_movies()
        seen_ids  = set(movie_ratings.keys())

        # Construire le profil utilisateur = moyenne des vecteurs genres pondérée
        genres_dummies = movies[C.GENRES_COL].str.get_dummies(sep="|")
        genres_dummies.index = movies[C.ITEM_ID_COL]

        liked_vectors = []
        for mid, rating in movie_ratings.items():
            if mid in genres_dummies.index and rating >= 3.0:
                liked_vectors.append(
                    genres_dummies.loc[mid].values * rating
                )

        if not liked_vectors:
            # Fallback : films les plus populaires
            return self._popular_fallback(movies, seen_ids, n)

        user_profile = np.mean(liked_vectors, axis=0)

        rows = []
        for _, movie in movies.iterrows():
            mid = int(movie[C.ITEM_ID_COL])
            if mid in seen_ids or mid not in genres_dummies.index:
                continue

            item_vec = genres_dummies.loc[mid].values
            norm     = np.linalg.norm(user_profile) * np.linalg.norm(item_vec)
            sim      = float(np.dot(user_profile, item_vec) / norm) if norm else 0.0

            rows.append({
                C.ITEM_ID_COL:   mid,
                C.LABEL_COL:     movie[C.LABEL_COL],
                C.GENRES_COL:    movie[C.GENRES_COL],
                "final_score":   round(sim,  3),
                "svd_score":     0.0,
                "item_score":    round(sim,  3),
            })

        return (
            pd.DataFrame(rows)
            .sort_values("final_score", ascending=False)
            .head(n)
            .reset_index(drop=True)
        )

    def _popular_fallback(self, movies, seen_ids, n):
        """Retourne les films les plus populaires si pas de profil."""
        ratings  = load_ratings(surprise_format=False, use_implicit=False)
        popular  = (
            ratings.groupby(C.ITEM_ID_COL)[C.RATING_COL]
            .agg(score="mean")
            .reset_index()
            .sort_values("score", ascending=False)
        )
        popular = popular[~popular[C.ITEM_ID_COL].isin(seen_ids)].head(n)
        merged  = popular.merge(movies, on=C.ITEM_ID_COL, how="left")
        merged["final_score"] = merged["score"].round(3)
        merged["svd_score"]   = 0.0
        merged["item_score"]  = 0.0
        return merged[[C.ITEM_ID_COL, C.LABEL_COL, C.GENRES_COL,
                        "final_score", "svd_score", "item_score"]].head(n)

    # ------------------------------------------------------------------
    # Recommandation de groupe
    # ------------------------------------------------------------------

    def recommend_group_by_movies(
        self,
        participants: list,
        n: int = 8,
        strategy: str = "average"
    ) -> pd.DataFrame:
        """
        Recommandations pour un groupe basées sur les films choisis.

        Chaque participant a une liste de films aimés et/ou souhaités.
        On génère un score de groupe par agrégation.

        Stratégies
        ----------
        "average"      : moyenne des scores → satisfait le plus grand nombre
        "least_misery" : minimum des scores → personne n'est déçu
        """
        movies    = load_movies().set_index(C.ITEM_ID_COL)
        all_ids   = set(movies.index)

        # Collecter tous les movieIds des participants (liked + wishlist)
        participant_movie_ids = []
        for p in participants:
            ids = (
                [m["id"] for m in p.get("liked",    [])]
                + [m["id"] for m in p.get("wishlist", [])]
            )
            participant_movie_ids.append(set(int(i) for i in ids if i))

        # Films candidats = union de tous les films choisis + films similaires
        all_chosen = set().union(*participant_movie_ids) if participant_movie_ids else set()

        # Pour chaque film du catalogue, calculer le score de groupe
        group_scores = {}

        for mid in all_ids:
            if mid in all_chosen:
                continue  # On ne recommande pas ce qu'ils ont déjà choisi

            scores = []
            for p_ids in participant_movie_ids:
                if not p_ids:
                    continue
                # Score basé sur similarité item-item avec les films du participant
                p_scores = []
                for chosen_id in p_ids:
                    try:
                        inner_i = self.trainset.to_inner_iid(mid)
                        inner_j = self.trainset.to_inner_iid(chosen_id)
                        sim = self.item_based.sim[inner_i, inner_j]
                        p_scores.append(float(sim))
                    except Exception:
                        pass
                if p_scores:
                    scores.append(np.mean(p_scores))

            if not scores:
                continue

            if strategy == "least_misery":
                group_score = min(scores)
            else:
                group_score = np.mean(scores)

            if group_score > 0:
                title  = movies.loc[mid, C.LABEL_COL]  if mid in movies.index else str(mid)
                genres = movies.loc[mid, C.GENRES_COL] if mid in movies.index else ""
                group_scores[mid] = {
                    C.LABEL_COL:   title,
                    C.GENRES_COL:  genres,
                    "group_score": round(group_score, 4),
                    "n_users":     len(scores)
                }

        if not group_scores:
            return pd.DataFrame(columns=[
                C.ITEM_ID_COL, C.LABEL_COL, C.GENRES_COL, "group_score", "n_users"
            ])

        df = pd.DataFrame.from_dict(group_scores, orient="index")
        df.index.name = C.ITEM_ID_COL
        df = df.reset_index()
        return (
            df.sort_values("group_score", ascending=False)
            .head(n)
            .reset_index(drop=True)
        )

    # ------------------------------------------------------------------
    # Explication
    # ------------------------------------------------------------------

    def explain(self, user_id, movie_id: int) -> str:
        """Explication lisible basée sur les genres du film."""
        from loaders import load_items
        movie_id = int(movie_id)
        items    = load_items()

        if movie_id not in items.index:
            return "Recommended based on your viewing history."

        title  = items.loc[movie_id, C.LABEL_COL]
        genres = items.loc[movie_id, C.GENRES_COL]

        # Trouver les films similaires déjà vus
        seen = self.get_seen_movies(user_id)
        similar_seen = []

        try:
            inner_i = self.trainset.to_inner_iid(movie_id)
            sims    = self.item_based.sim[inner_i]
            top_similar = np.argsort(sims)[::-1][:20]

            for inner_j in top_similar:
                raw_j = int(self.trainset.to_raw_iid(inner_j))
                if raw_j in seen and raw_j in items.index:
                    similar_seen.append(items.loc[raw_j, C.LABEL_COL])
                if len(similar_seen) >= 2:
                    break
        except Exception:
            pass

        if similar_seen:
            return (
                f"Recommended because users who liked "
                f"{' and '.join(similar_seen)} also enjoyed '{title}'. "
                f"Genres: {genres}."
            )
        return f"'{title}' matches your taste profile. Genres: {genres}."

    # ------------------------------------------------------------------
    # Infos pour l'app
    # ------------------------------------------------------------------

    def get_model_info(self) -> list:
        return [
            {
                "name":   "SVD (Latent Factor)",
                "weight": W_SVD,
                "rmse":   0.887,
                "desc":   "Captures hidden patterns in user-item interactions."
            },
            {
                "name":   "Item-Based KNN (Pearson Baseline)",
                "weight": W_ITEM,
                "rmse":   0.876,
                "desc":   "Recommends movies similar to ones you already liked."
            },
        ]

    def get_weights(self) -> dict:
        return {"svd": W_SVD, "item_based": W_ITEM}