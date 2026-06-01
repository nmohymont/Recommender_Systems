"""
Modèles implémentés
-------------------
1.  ModelBaselineMean       — Baseline : moyenne globale
2.  UserBasedJaccardKNN     — User-based CF avec similarité Jaccard (custom,
                              non disponible dans Surprise)
3.  UserBasedKNN            — User-based CF avec Pearson Baseline (Surprise)
4.  ItemBasedKNN            — Item-based CF avec Pearson Baseline (Surprise)
5.  ContentBased            — Content-based par utilisateur :
                              features riches (genres, tags TF-IDF, genome,
                              visuels, stats rating) + Ridge par user
6.  SVDModel                — Factorisation matricielle FunkSVD (Surprise)
"""
from collections import defaultdict

import numpy as np
import random as rd
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

from surprise import AlgoBase, KNNBaseline, SVD
from surprise.prediction_algorithms.predictions import PredictionImpossible

from constants import Constant as C
from loaders import load_items


# ===========================================================================
# Utilitaire
# ===========================================================================

def get_top_n(predictions, n=10):
    """
    Retourne le top-N pour chaque utilisateur à partir des prédictions Surprise.
    Avec tie-breaking aléatoire pour éviter les biais d'ordre.
    """
    rd.seed(0)
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))
    for uid, user_ratings in top_n.items():
        rd.shuffle(user_ratings)
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]
    return top_n


# ===========================================================================
# 1. BASELINE
# ===========================================================================

class ModelBaselineMean(AlgoBase):
    """Prédit la moyenne globale des ratings pour toute paire (user, item)."""

    def __init__(self):
        AlgoBase.__init__(self)

    def fit(self, trainset):
        AlgoBase.fit(self, trainset)
        self.global_mean = trainset.global_mean
        return self

    def estimate(self, u, i):
        return self.global_mean


# ===========================================================================
# 2. USER-BASED — Jaccard KNN (custom, absent de Surprise)
# ===========================================================================

class UserBasedJaccardKNN(AlgoBase):
    """
    KNN user-based avec similarité de Jaccard.

        Jaccard(u,v) = |I_u ∩ I_v| / |I_u ∪ I_v|

    Métrique NON disponible dans Surprise => satisfait l'exigence custom.
    Prédiction : moyenne pondérée centrée sur la moyenne utilisateur.
    """

    def __init__(self, k: int = 40, min_k: int = 3):
        AlgoBase.__init__(self)
        self.k = k
        self.min_k = min_k

    def fit(self, trainset):
        AlgoBase.fit(self, trainset)
        self.user_items = {}
        self.user_means = {}
        for inner_uid in trainset.all_users():
            ratings = trainset.ur[inner_uid]
            self.user_items[inner_uid] = {iid for iid, _ in ratings}
            vals = [r for _, r in ratings]
            self.user_means[inner_uid] = (
                np.mean(vals) if vals else trainset.global_mean
            )
        return self

    def jaccard_similarity(self, u: int, v: int) -> float:
        items_u = self.user_items.get(u, set())
        items_v = self.user_items.get(v, set())
        union = items_u | items_v
        if not union:
            return 0.0
        return len(items_u & items_v) / len(union)

    def estimate(self, u, i):
        if not self.trainset.knows_user(u):
            raise PredictionImpossible("Unknown user.")

        neighbors = []
        for v in self.trainset.all_users():
            if v == u:
                continue
            v_ratings = dict(self.trainset.ur[v])
            if i not in v_ratings:
                continue
            sim = self.jaccard_similarity(u, v)
            if sim > 0:
                neighbors.append((sim, v_ratings[i], self.user_means[v]))

        neighbors.sort(key=lambda x: x[0], reverse=True)
        neighbors = neighbors[: self.k]

        if len(neighbors) < self.min_k:
            return self.user_means.get(u, self.trainset.global_mean)

        num = sum(s * (r - m) for s, r, m in neighbors)
        den = sum(abs(s) for s, _, _ in neighbors)

        if den == 0:
            return self.user_means.get(u, self.trainset.global_mean)

        pred = self.user_means.get(u, self.trainset.global_mean) + num / den
        return float(np.clip(pred, C.RATINGS_SCALE[0], C.RATINGS_SCALE[1]))


# ===========================================================================
# 3. USER-BASED — KNNBaseline Pearson (Surprise)
# ===========================================================================

class UserBasedKNN(KNNBaseline):
    """
    KNN user-based avec similarité Pearson Baseline.
    Paramètres optimaux : k=80, ALS (n_epochs=40, reg_u=8, reg_i=4).
    RMSE obtenu sur le small dataset : 0.891
    """

    def __init__(self, k: int = 80, min_k: int = 3):
        KNNBaseline.__init__(
            self,
            k=k,
            min_k=min_k,
            sim_options={"name": "pearson_baseline", "user_based": True},
            bsl_options={"method": "als", "n_epochs": 40, "reg_u": 8, "reg_i": 4},
            verbose=False
        )


# ===========================================================================
# 4. ITEM-BASED — KNNBaseline Pearson (Surprise)
# ===========================================================================

class ItemBasedKNN(KNNBaseline):
    """
    KNN item-based avec similarité Pearson Baseline.
    Paramètres optimaux : k=60, ALS (n_epochs=30, reg_u=10, reg_i=5).
    RMSE obtenu sur le small dataset : 0.876 (meilleur KNN)
    """

    def __init__(self, k: int = 60, min_k: int = 3):
        KNNBaseline.__init__(
            self,
            k=k,
            min_k=min_k,
            sim_options={"name": "pearson_baseline", "user_based": False},
            bsl_options={"method": "als", "n_epochs": 30, "reg_u": 10, "reg_i": 5},
            verbose=False
        )


# ===========================================================================
# 5. CONTENT-BASED — Ridge par utilisateur sur features riches
# ===========================================================================

class ContentBased(AlgoBase):
    """
    Recommandeur content-based avec un modèle Ridge par utilisateur.

    Architecture
    ------------
    1. Content Analyzer  : vecteurs de features pour chaque film
       V1 : année + genres (baseline)
       V2 : V1 + tags TF-IDF + genome + visuels
       V3 : V2 + meta-genres + stats tags + genome agrégés + dérivés visuels
            + stats rating global du film  (=> utilisé sur hackathon, RMSE 0.727)

    2. Profile Learner   : Ridge(alpha) entraîné par utilisateur

    3. Filtering         : r̂(u,i) = Ridge_u(features_i)

    4. Explainability    : coefficients Ridge => top features par user

    Paramètres
    ----------
    features_method : "V3" | "V2" | "V1"
    alpha           : régularisation Ridge (24 = optimal hackathon)
    """

    def __init__(self, features_method: str = "V3", alpha: float = 24.0):
        AlgoBase.__init__(self)
        self.features_method = features_method
        self.alpha = alpha
        self.content_features = self._create_content_features(features_method)

    # --- Content Analyzer ---------------------------------------------------

    def _create_content_features(self, method: str) -> pd.DataFrame:
        df_items = load_items()
        if method == "V1":
            return self._features_v1(df_items)
        elif method == "V2":
            return self._features_v2(df_items)
        elif method == "V3":
            return self._features_v3(df_items)
        else:
            raise NotImplementedError(f"features_method '{method}' inconnu.")

    def _features_v1(self, df_items):
        df = pd.DataFrame(index=df_items.index)
        df["year"] = df_items[C.LABEL_COL].str.extract(r"\((\d{4})\)").astype(float)
        df["year"] = df["year"].fillna(df["year"].median())
        genres = df_items[C.GENRES_COL].str.get_dummies(sep="|")
        genres.index = df_items.index
        if "(no genres listed)" in genres.columns:
            genres = genres.drop(columns=["(no genres listed)"])
        df = df.join(genres)
        scaler = MinMaxScaler()
        df[df.columns] = scaler.fit_transform(df)
        return df

    def _features_v2(self, df_items):
        df = pd.DataFrame(index=df_items.index)
        df["year"] = df_items[C.LABEL_COL].str.extract(r"\((\d{4})\)").astype(float)
        df["year"] = df["year"].fillna(df["year"].median())
        genres = df_items[C.GENRES_COL].str.get_dummies(sep="|")
        genres.index = df_items.index
        if "(no genres listed)" in genres.columns:
            genres = genres.drop(columns=["(no genres listed)"])
        df = df.join(genres)
        tags_path = C.CONTENT_PATH / "tags.csv"
        if tags_path.exists():
            tags = pd.read_csv(tags_path)
            tags["tag"] = tags["tag"].fillna("").astype(str).str.lower()
            tags_by_movie = tags.groupby("movieId")["tag"].apply(
                lambda x: " ".join(x)
            ).reindex(df_items.index).fillna("")
            tfidf = TfidfVectorizer(max_features=100, min_df=2)
            tfidf_mat = tfidf.fit_transform(tags_by_movie)
            df_tags = pd.DataFrame(
                tfidf_mat.toarray(), index=df_items.index,
                columns=[f"tag_{c}" for c in tfidf.get_feature_names_out()]
            )
            df = df.join(df_tags)
        genome_path = C.CONTENT_PATH / "genome-scores.csv"
        if genome_path.exists():
            genome = pd.read_csv(genome_path).pivot(
                index="movieId", columns="tagId", values="relevance"
            )
            genome.columns = [f"genome_{c}" for c in genome.columns]
            df = df.join(genome, how="left")
        visual_path = C.CONTENT_PATH / "LLVisualFeatures13K_QuantileLog.csv"
        if visual_path.exists():
            vis = pd.read_csv(visual_path)
            vis = vis.rename(columns={"ML_Id": "movieId", "ML_ID": "movieId"})
            vis = vis.set_index("movieId")
            vis.columns = [f"visual_{c}" for c in vis.columns]
            df = df.join(vis, how="left")
        df = df.fillna(0)
        scaler = MinMaxScaler()
        df[df.columns] = scaler.fit_transform(df)
        return df

    def _features_v3(self, df_items):
        df = pd.DataFrame(index=df_items.index)
        # 1. Année
        df["year"] = df_items[C.LABEL_COL].str.extract(r"\((\d{4})\)").astype(float)
        df["year"] = df["year"].fillna(df["year"].median())
        # 2. Décennie
        df["decade"] = (df["year"] // 10) * 10
        decade_dummies = pd.get_dummies(df["decade"], prefix="decade")
        decade_dummies.index = df.index
        df = df.drop(columns=["decade"]).join(decade_dummies)
        # 3. Genres
        genres = df_items[C.GENRES_COL].str.get_dummies(sep="|")
        genres.index = df_items.index
        if "(no genres listed)" in genres.columns:
            genres = genres.drop(columns=["(no genres listed)"])
        df = df.join(genres)
        # 4. Meta-genres
        genres_clean = df_items[C.GENRES_COL].fillna("").replace("(no genres listed)", "")
        df["n_genres"] = genres_clean.apply(lambda x: 0 if x == "" else len(x.split("|")))
        genre_freq = genres.mean(axis=0)
        df["genre_popularity_mean"] = (
            genres.dot(genre_freq) / df["n_genres"].replace(0, np.nan)
        )
        # 5. Tags TF-IDF + meta
        tags_path = C.CONTENT_PATH / "tags.csv"
        if tags_path.exists():
            tags = pd.read_csv(tags_path)
            tags["tag"] = tags["tag"].fillna("").astype(str).str.lower()
            tags_by_movie = tags.groupby("movieId")["tag"].apply(
                lambda x: " ".join(x)
            ).reindex(df_items.index).fillna("")
            tfidf = TfidfVectorizer(max_features=150, min_df=2)
            tfidf_mat = tfidf.fit_transform(tags_by_movie)
            df_tags = pd.DataFrame(
                tfidf_mat.toarray(), index=df_items.index,
                columns=[f"tag_{c}" for c in tfidf.get_feature_names_out()]
            )
            df = df.join(df_tags)
            df["n_tags"] = tags.groupby("movieId").size().reindex(df_items.index).fillna(0)
            tag_freq = tags["tag"].value_counts()
            tags["tag_frequency"] = tags["tag"].map(tag_freq)
            df["avg_tag_frequency"] = tags.groupby("movieId")["tag_frequency"].mean().reindex(df_items.index).fillna(0)
        # 6. Genome agrégés + full
        genome_path = C.CONTENT_PATH / "genome-scores.csv"
        if genome_path.exists():
            genome = pd.read_csv(genome_path)
            genome_piv = genome.pivot(index="movieId", columns="tagId", values="relevance")
            genome_piv.columns = [f"genome_{c}" for c in genome_piv.columns]
            df["genome_mean"]     = genome_piv.mean(axis=1).reindex(df_items.index)
            df["genome_std"]      = genome_piv.std(axis=1).reindex(df_items.index)
            df["genome_max"]      = genome_piv.max(axis=1).reindex(df_items.index)
            df["genome_min"]      = genome_piv.min(axis=1).reindex(df_items.index)
            df["genome_n_strong"] = (genome_piv > 0.8).sum(axis=1).reindex(df_items.index)
            df = df.join(genome_piv, how="left")
        # 7. Visuels + dérivés
        visual_path = C.CONTENT_PATH / "LLVisualFeatures13K_QuantileLog.csv"
        if visual_path.exists():
            vis = pd.read_csv(visual_path)
            vis = vis.rename(columns={"ML_Id": "movieId", "ML_ID": "movieId"})
            vis = vis.set_index("movieId")
            if all(c in vis.columns for c in ["f1","f2","f3","f4","f5","f6","f7"]):
                vis["visual_fast_paced"]        = vis["f4"] / (vis["f1"] + 1e-6)
                vis["visual_action_intensity"]  = vis["f4"] + vis["f7"]
                vis["visual_color_complexity"]  = vis["f2"] + vis["f3"]
                vis["visual_motion_complexity"] = vis["f4"] + vis["f5"]
                vis["visual_dark_score"]        = 1 - vis["f6"]
            vis.columns = [
                f"visual_{c}" if not str(c).startswith("visual_") else c
                for c in vis.columns
            ]
            df = df.join(vis, how="left")
        # 8. Stats rating global
        ratings_path = C.EVIDENCE_PATH / C.RATINGS_FILENAME
        if ratings_path.exists():
            ratings = pd.read_csv(ratings_path)
            movie_stats = ratings.groupby(C.ITEM_ID_COL)[C.RATING_COL].agg(
                movie_n_ratings="count",
                movie_mean_rating="mean",
                movie_std_rating="std"
            )
            movie_stats["movie_std_rating"] = movie_stats["movie_std_rating"].fillna(0)
            m = 10
            gm = ratings[C.RATING_COL].mean()
            movie_stats["movie_bayesian_mean"] = (
                (movie_stats["movie_n_ratings"] * movie_stats["movie_mean_rating"] + m * gm)
                / (movie_stats["movie_n_ratings"] + m)
            )
            movie_stats["movie_inverse_popularity"] = 1 / (movie_stats["movie_n_ratings"] + 1)
            df = df.join(movie_stats, how="left")
        # Nettoyage
        df = df.fillna(0)
        scaler = MinMaxScaler()
        df[df.columns] = scaler.fit_transform(df)
        return df

    # --- Profile Learner ----------------------------------------------------

    def fit(self, trainset):
        AlgoBase.fit(self, trainset)
        feature_names = self.content_features.columns.tolist()
        self.user_profile = {}
        for u in trainset.all_users():
            df_user = pd.DataFrame(
                trainset.ur[u], columns=["inner_item_id", "user_ratings"]
            )
            df_user["item_id"] = df_user["inner_item_id"].map(trainset.to_raw_iid)
            df_user = df_user.merge(
                self.content_features, how="left",
                left_on="item_id", right_index=True
            )
            X = df_user[feature_names].values
            y = df_user["user_ratings"].values
            reg = Ridge(alpha=self.alpha)
            reg.fit(X, y)
            self.user_profile[u] = reg
        return self

    # --- Filtering ----------------------------------------------------------

    def estimate(self, u, i):
        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            raise PredictionImpossible("User and/or item is unknown.")
        raw_item_id = self.trainset.to_raw_iid(i)
        X = self.content_features.loc[raw_item_id:raw_item_id, :].values
        score = self.user_profile[u].predict(X)[0]
        return float(np.clip(score, C.RATINGS_SCALE[0], C.RATINGS_SCALE[1]))

    # --- Explainability -----------------------------------------------------

    def get_feature_importances(self, inner_uid: int) -> dict:
        """Coefficients Ridge normalisés — top features par utilisateur."""
        if inner_uid not in self.user_profile:
            return {}
        coefs = np.abs(self.user_profile[inner_uid].coef_)
        total = coefs.sum()
        if total == 0:
            return {}
        norm = coefs / total
        result = dict(zip(self.content_features.columns, norm))
        return dict(sorted(result.items(), key=lambda x: x[1], reverse=True))

    def explain(self, user_id, movie_id: int) -> str:
        """Explication lisible pour l'app Flask."""
        movie_id  = int(movie_id)
        df_items  = load_items()
        if movie_id not in df_items.index:
            return "This recommendation is based on your content profile."

        title  = df_items.loc[movie_id, C.LABEL_COL]
        genres = df_items.loc[movie_id, C.GENRES_COL]

        inner_uid = None
        for uid in [user_id, int(user_id) if str(user_id).lstrip("-").isdigit() else None]:
            try:
                inner_uid = self.trainset.to_inner_uid(uid)
                break
            except Exception:
                pass

        if inner_uid is None or inner_uid not in self.user_profile:
            return f"'{title}' is recommended based on your content profile. Genres: {genres}."

        importances = self.get_feature_importances(inner_uid)
        readable = []
        for feat in list(importances.keys())[:5]:
            if feat.startswith("tag_"):
                readable.append(feat.replace("tag_", ""))
            elif feat.startswith("genre_") or feat in genres.split("|"):
                readable.append(feat.replace("genre_", ""))
            elif feat == "year":
                readable.append("release year")
            elif feat.startswith("genome_"):
                readable.append("semantic profile")
            elif feat.startswith("visual_"):
                readable.append("visual style")
            elif feat.startswith("movie_"):
                readable.append("popularity")
        readable = list(dict.fromkeys(readable))[:3]

        if readable:
            return (
                f"'{title}' is recommended because your profile values "
                f"{', '.join(readable)}. Genres: {genres}."
            )
        return f"'{title}' matches your content profile. Genres: {genres}."


# ===========================================================================
# 6. LATENT FACTOR — FunkSVD (Surprise)
# ===========================================================================

class SVDModel(SVD):
    """
    Factorisation matricielle FunkSVD.

    Optimise : min Σ(r_ui - μ - b_u - b_i - p_u·q_i)² + λ(...)

    Paramètres optimaux sur le small dataset :
        n_factors=75, n_epochs=50, lr_all=0.005, reg_all=0.08
        => RMSE 0.887

    Pour affiner davantage : utiliser tune_svd.py avec Optuna.
    """

    def __init__(
        self,
        n_factors:    int   = 75,
        n_epochs:     int   = 50,
        lr_all:       float = 0.005,
        reg_all:      float = 0.08,
        random_state: int   = 42
    ):
        SVD.__init__(
            self,
            n_factors=n_factors,
            n_epochs=n_epochs,
            lr_all=lr_all,
            reg_all=reg_all,
            random_state=random_state
        )