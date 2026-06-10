"""
recommander_test.py
===================
Composant de recommandation hybride — version app.

Modèles utilisés :
    - SVDModel         RMSE 0.883  poids 0.45   meilleur nDCG@10
    - ContentBased V3  RMSE 0.858  poids 0.30   meilleur RMSE + cold-start + explainability
    - UserBasedITRKNN  RMSE 0.906  poids 0.25   meilleure diversité

Fonctionnalité dédiée :
    - ContentBased V5  (BERT embeddings)         bouton "Surprise Me" (novelty = 3.67)

Démarrage app : ~15 minutes (ContentBased V3 inclus).
"""

import pandas as pd
import numpy as np

from surprise import Dataset, Reader

from constants import Constant as C
from loaders import load_movies, load_ratings
from models_test import SVDModel, ItemBasedKNN, UserBasedITRKNN, ContentBased


# Poids du hybrid COMPLET (SVD + Content + ITR)
W_SVD     = 0.45
W_CONTENT = 0.30
W_USER    = 0.25


class HybridRecommender:
    """
    Recommandeur hybride SVD + ContentBased V3 + UserBased ITR.

    Paramètres
    ----------
    use_implicit : bool
        Si True, charge ratings_with_implicit_ilies.csv (inclut userId=-1).
    use_content : bool
        Si True, charge ContentBased V3 (~15 min au démarrage).
        Si False, utilise SVD + ITR uniquement (~90 sec).
    """

    def __init__(self, use_implicit: bool = True, use_content: bool = False):
        self.use_implicit = use_implicit
        self.use_content  = use_content
        self.trainset     = None
        self.svd          = None
        self.itr          = None
        self.content      = None   # V3 — hybride + explainability + cold-start
        self.content_v5   = None   # V5 — "Surprise Me" uniquement
        self.item_based   = None   # gardé pour le group recommender
        self._is_fitted   = False

        if self.use_content:
            self.w_svd     = W_SVD
            self.w_content = W_CONTENT
            self.w_user    = W_USER
        else:
            self.w_svd     = 0.60
            self.w_content = 0.0
            self.w_user    = 0.40

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self):
        print("Building trainset...")
        self.trainset = self._build_trainset()
        print(f"  {self.trainset.n_users} users | "
              f"{self.trainset.n_items} items | "
              f"{self.trainset.n_ratings} ratings")

        print("Training SVD...")
        self.svd = SVDModel(
            n_factors=35, n_epochs=98,
            lr_all=0.005, reg_all=0.0934, random_state=42
        )
        self.svd.fit(self.trainset)
        print("  ✓ SVD ready")

        print("Training UserBased ITR...")
        self.itr = UserBasedITRKNN(k=60, min_k=5)
        self.itr.fit(self.trainset)
        print("  ✓ ITR ready")

        if self.use_content:
            print("Training ContentBased V3...")
            self.content = ContentBased(features_method="V3", alpha=1.0)
            self.content.fit(self.trainset)
            print("  ✓ ContentBased V3 ready")

            print("Training ContentBased V5 (Surprise Me)...")
            self.content_v5 = ContentBased(features_method="V5", alpha=25.0)
            self.content_v5.fit(self.trainset)
            print("  ✓ ContentBased V5 ready")

        # ItemBased gardé pour le group recommender
        print("Training ItemBasedKNN...")
        self.item_based = ItemBasedKNN()
        self.item_based.fit(self.trainset)
        print("  ✓ ItemBasedKNN ready")

        self._is_fitted = True
        mode = "SVD + Content V3 + ITR" if self.use_content else "SVD + ITR"
        print(f"Hybrid recommender ready ({mode}).")
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
        """Top-N hybride SVD + Content + ITR pour un utilisateur existant."""
        if not self._is_fitted:
            raise RuntimeError("Call fit() first.")

        movies = load_movies()
        seen   = self.get_seen_movies(user_id)
        rows   = []

        for _, movie in movies.iterrows():
            mid = int(movie[C.ITEM_ID_COL])
            if mid in seen:
                continue

            svd_score     = self._safe_predict(self.svd,     user_id, mid)
            itr_score     = self._safe_predict(self.itr,     user_id, mid)
            content_score = self._safe_predict(self.content, user_id, mid) if self.use_content else 0.0

            if self.use_content:
                final = (self.w_svd * svd_score
                       + self.w_content * content_score
                       + self.w_user * itr_score)
            else:
                final = self.w_svd * svd_score + self.w_user * itr_score

            rows.append({
                C.ITEM_ID_COL:    mid,
                C.LABEL_COL:      movie[C.LABEL_COL],
                C.GENRES_COL:     movie[C.GENRES_COL],
                "final_score":    round(final,         3),
                "svd_score":      round(svd_score,     3),
                "item_score":     round(itr_score,     3),
                "content_score":  round(content_score, 3),
            })

        return (
            pd.DataFrame(rows)
            .sort_values("final_score", ascending=False)
            .head(n)
            .reset_index(drop=True)
        )

    def recommend_new_user(self, movie_ratings: dict, n: int = 10) -> pd.DataFrame:
        """
        Recommandations pour un nouvel utilisateur (cold-start).
        Stratégie : ContentBased V3 via le profil implicite userId=-1,
        dont les ratings ont été injectés dans le trainset via use_implicit=True.
        Fallback sur la popularité si V3 n'est pas disponible.
        """
        movies   = load_movies()
        seen_ids = set(movie_ratings.keys())

        # Tenter d'utiliser ContentBased V3 avec userId=-1
        if self.use_content and self.content is not None:
            rows = []
            for _, movie in movies.iterrows():
                mid = int(movie[C.ITEM_ID_COL])
                if mid in seen_ids:
                    continue
                score = self._safe_predict(self.content, user_id=-1, movie_id=mid)
                rows.append({
                    C.ITEM_ID_COL:   mid,
                    C.LABEL_COL:     movie[C.LABEL_COL],
                    C.GENRES_COL:    movie[C.GENRES_COL],
                    "final_score":   round(score, 3),
                    "svd_score":     0.0,
                    "item_score":    0.0,
                    "content_score": round(score, 3),
                })
            if rows:
                return (
                    pd.DataFrame(rows)
                    .sort_values("final_score", ascending=False)
                    .head(n)
                    .reset_index(drop=True)
                )

        return self._popular_fallback(movies, seen_ids, n)

    def _popular_fallback(self, movies, seen_ids, n):
        ratings = load_ratings(surprise_format=False, use_implicit=False)
        popular = (
            ratings.groupby(C.ITEM_ID_COL)[C.RATING_COL]
            .agg(score="mean")
            .reset_index()
            .sort_values("score", ascending=False)
        )
        popular = popular[~popular[C.ITEM_ID_COL].isin(seen_ids)].head(n)
        merged  = popular.merge(movies, on=C.ITEM_ID_COL, how="left")
        merged["final_score"]   = merged["score"].round(3)
        merged["svd_score"]     = 0.0
        merged["item_score"]    = 0.0
        merged["content_score"] = 0.0
        return merged[[C.ITEM_ID_COL, C.LABEL_COL, C.GENRES_COL,
                       "final_score", "svd_score", "item_score", "content_score"]].head(n)

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
        Recommandations pour un groupe via le modele hybride complet
        (SVD + ContentBased V3 + ITR).

        Pour chaque film du catalogue, on predit le score hybride de chaque
        participant, puis on agregue selon la strategie choisie :
          - average      : moyenne des scores -> maximise la satisfaction globale
          - least_misery : minimum des scores -> evite qu un participant deteste le film

        Les films deja mentionnes par les participants (liked + wishlist) sont
        exclus des recommandations.
        """
        movies = load_movies().set_index(C.ITEM_ID_COL)

        group_user_ids = []
        instant_movies = set()

        for p in participants:
            uid = p.get("user_id")
            if uid is not None:
                group_user_ids.append(uid)
            for m in p.get("liked", []):
                instant_movies.add(int(m["id"]))
            for m in p.get("wishlist", []):
                instant_movies.add(int(m["id"]))

        if not group_user_ids:
            return pd.DataFrame(columns=[
                C.ITEM_ID_COL, C.LABEL_COL, C.GENRES_COL, "group_score", "n_users"
            ])

        group_scores = {}

        for inner_iid in self.trainset.all_items():
            raw_iid = int(self.trainset.to_raw_iid(inner_iid))

            if raw_iid in instant_movies:
                continue

            user_scores = []
            for raw_uid in group_user_ids:
                try:
                    svd_s     = self._safe_predict(self.svd,     raw_uid, raw_iid)
                    itr_s     = self._safe_predict(self.itr,     raw_uid, raw_iid)
                    content_s = self._safe_predict(self.content, raw_uid, raw_iid) if self.use_content else 0.0

                    if self.use_content:
                        score = (self.w_svd * svd_s
                               + self.w_content * content_s
                               + self.w_user * itr_s)
                    else:
                        score = self.w_svd * svd_s + self.w_user * itr_s

                    user_scores.append(score)
                except Exception:
                    user_scores.append(float(self.trainset.global_mean))

            if not user_scores:
                continue

            group_score = min(user_scores) if strategy == "least_misery" else np.mean(user_scores)  # "average" or default

            group_scores[raw_iid] = {
                C.LABEL_COL:   movies.loc[raw_iid, C.LABEL_COL]  if raw_iid in movies.index else str(raw_iid),
                C.GENRES_COL:  movies.loc[raw_iid, C.GENRES_COL] if raw_iid in movies.index else "",
                "group_score": round(float(group_score), 4),
                "n_users":     len(user_scores),
            }

        if not group_scores:
            return pd.DataFrame(columns=[
                C.ITEM_ID_COL, C.LABEL_COL, C.GENRES_COL, "group_score", "n_users"
            ])

        df = pd.DataFrame.from_dict(group_scores, orient="index")
        df.index.name = C.ITEM_ID_COL
        return df.reset_index().sort_values("group_score", ascending=False).head(n).reset_index(drop=True)

    def recommend_group_surprise(
        self,
        participants: list,
        n: int = 8,
    ) -> pd.DataFrame:
        """
        Mode Surprise pour un groupe via ContentBased V5 (BERT embeddings).

        Algorithme (d apres les images de methodologie recues) :
          A. Fusionner les vecteurs BERT des films wishlist du groupe
             en un vecteur moyen V_group.
          B. Pour chaque film du catalogue, calculer :
               Score_Surprise(i) = cosine(V_group, V_i) / log(1 + popularity(i))
             La penalite de popularite fait emerger la long-tail semantique.
          C. Garder le Top 50, puis echantillonner n films aleatoirement
             (stochastic selection) pour garantir une experience unique a chaque clic.
        """
        import random
        if not self._is_fitted or self.content_v5 is None:
            raise RuntimeError("Call fit() with use_content=True first.")

        movies   = load_movies().set_index(C.ITEM_ID_COL)
        v5_feats = self.content_v5.content_features

        # --- A. Vecteur BERT moyen du groupe (wishlist uniquement) ---
        instant_movies = set()
        valid_vectors  = []

        for p in participants:
            for m in p.get("wishlist", []):
                mid = int(m["id"])
                instant_movies.add(mid)
                if mid in v5_feats.index:
                    valid_vectors.append(v5_feats.loc[mid].values)
            for m in p.get("liked", []):
                instant_movies.add(int(m["id"]))

        if not valid_vectors:
            return pd.DataFrame(columns=[
                C.ITEM_ID_COL, C.LABEL_COL, C.GENRES_COL, "group_score", "n_users"
            ])

        v_group = np.mean(valid_vectors, axis=0)
        norm_group = np.linalg.norm(v_group)

        # --- Popularite : nb de ratings par film ---
        ratings    = load_ratings(surprise_format=False, use_implicit=False)
        popularity = ratings.groupby(C.ITEM_ID_COL).size().to_dict()

        # --- B. Score Surprise pour chaque film ---
        candidates = []
        for raw_iid in v5_feats.index:
            if raw_iid in instant_movies:
                continue
            film_vec   = v5_feats.loc[raw_iid].values
            norm_film  = np.linalg.norm(film_vec)
            if norm_group == 0 or norm_film == 0:
                continue
            cosine_sim    = float(np.dot(v_group, film_vec) / (norm_group * norm_film))
            pop_count     = popularity.get(raw_iid, 1)
            surprise_score = cosine_sim / np.log(1 + pop_count)
            candidates.append((raw_iid, surprise_score))

        if not candidates:
            return pd.DataFrame(columns=[
                C.ITEM_ID_COL, C.LABEL_COL, C.GENRES_COL, "group_score", "n_users"
            ])

        # --- C. Top 50 puis stochastic selection ---
        candidates.sort(key=lambda x: x[1], reverse=True)
        top50  = candidates[:50]
        chosen = random.sample(top50, min(n, len(top50)))
        chosen.sort(key=lambda x: x[1], reverse=True)

        rows = []
        for raw_iid, score in chosen:
            rows.append({
                C.ITEM_ID_COL: raw_iid,
                C.LABEL_COL:   movies.loc[raw_iid, C.LABEL_COL]  if raw_iid in movies.index else str(raw_iid),
                C.GENRES_COL:  movies.loc[raw_iid, C.GENRES_COL] if raw_iid in movies.index else "",
                "group_score": round(float(score), 6),
                "n_users":     len(participants),
            })

        return pd.DataFrame(rows).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Surprise Me (individuel)
    # ------------------------------------------------------------------

    def surprise_me(self, user_id, n: int = 10) -> pd.DataFrame:
        """
        Recommandations inattendues via ContentBased V5 (BERT embeddings).
        Maximise la novelty (3.67) en exploitant la sémantique des descriptions.
        Utilisé exclusivement par le bouton 'Surprise Me'.
        """
        if not self._is_fitted or self.content_v5 is None:
            raise RuntimeError("Call fit() with use_content=True first.")

        movies = load_movies()
        seen   = self.get_seen_movies(user_id)
        rows   = []

        for _, movie in movies.iterrows():
            mid = int(movie[C.ITEM_ID_COL])
            if mid in seen:
                continue
            score = self._safe_predict(self.content_v5, user_id, mid)
            rows.append({
                C.ITEM_ID_COL:   mid,
                C.LABEL_COL:     movie[C.LABEL_COL],
                C.GENRES_COL:    movie[C.GENRES_COL],
                "final_score":   round(score, 3),
                "svd_score":     0.0,
                "item_score":    0.0,
                "content_score": round(score, 3),
            })

        return (
            pd.DataFrame(rows)
            .sort_values("final_score", ascending=False)
            .head(n)
            .reset_index(drop=True)
        )

    # ------------------------------------------------------------------
    # Explication
    # ------------------------------------------------------------------

    def explain(self, user_id, movie_id: int) -> str:
        """
        Génère une explication via les coefficients Ridge du ContentBased V3.
        Identifie les features (genres, tags, profil sémantique) les plus
        influentes dans les préférences de l'utilisateur.
        """
        if self.use_content and self.content is not None:
            return self.content.explain(user_id, movie_id)

        # Fallback si V3 non chargé
        from loaders import load_items
        items = load_items()
        movie_id = int(movie_id)
        if movie_id not in items.index:
            return "Recommended based on your viewing history."
        title  = items.loc[movie_id, C.LABEL_COL]
        genres = items.loc[movie_id, C.GENRES_COL]
        return f"'{title}' matches your taste profile. Genres: {genres}."