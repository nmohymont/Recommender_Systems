"""
models_test.py
==============
Tous les modèles de recommandation du projet MLSMM2156.

Modèles implémentés
-------------------
1.  ModelBaselineMean       — Baseline : moyenne globale
2.  UserBasedJaccardKNN     — User-based CF avec similarité Jaccard (custom) [commenté]
3.  UserBasedITRKNN         — User-based CF avec similarité ITR (custom) [commenté]
4.  UserBasedCosineKNN      — User-based CF avec Cosine (Surprise)
5.  UserBasedKNN            — User-based CF avec Pearson Baseline (Surprise)
6.  ItemBasedKNN            — Item-based CF avec Adjusted Cosine (Surprise)
7.  ItemBasedPearsonKNN     — Item-based CF avec Pearson Baseline (Surprise)
8.  ContentBased            — Content-based Ridge par utilisateur (V1/V2/V3)
9.  SVDModel                — Factorisation matricielle FunkSVD (Surprise)
"""

from collections import defaultdict

import numpy as np
import random as rd
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

from surprise import AlgoBase, KNNBaseline, KNNBasic, KNNWithMeans, SVD
from surprise.prediction_algorithms.predictions import PredictionImpossible

from constants import Constant as C
from loaders import load_items


# ===========================================================================
# Utilitaire
# ===========================================================================

def get_top_n(predictions, n=10):
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
# 3. USER-BASED — ITR KNN (custom, absent de Surprise) 
# ===========================================================================

class UserBasedITRKNN(AlgoBase):
    """"
    KNN user-based avec similarité ITR (Improved Triangle + User Rating Preferences).
 
    Référence : Iftikhar, A., Ghazanfar, M. A., Ayub, M., Mehmood, Z., & Maqsood, M.
    (2020). An Improved Product Recommendation Method for Collaborative Filtering.
    IEEE Access, 8, 123841-123857. https://doi.org/10.1109/ACCESS.2020.3007553
 
    Formule :
        sim_ITR(u,v) = sim_TRIANGLE(u,v) × sim_URP(u,v)
 
    sim_TRIANGLE(u,v) = 1 - sqrt(Σ_{i ∈ I_u ∪ I_v} (r_ui - r_vi)²)
                            / (sqrt(Σ r_ui²) + sqrt(Σ r_vi²))
        → mesure la distance entre les vecteurs de ratings sur l'union
          des items notés ; les ratings absents sont traités comme 0,
          ce qui pénalise les asymétries de couverture entre utilisateurs.
    
 
    sim_URP(u,v) = 1 - 1 / (1 + exp(-|μ_u - μ_v| × |σ_u - σ_v|))
    
        → sigmoïde décroissante bornée dans (0, 0.5] ; pénalise les
          paires d'utilisateurs dont les habitudes globales de notation
          divergent (niveau moyen et dispersion). Calculée sur l'ensemble
          complet des ratings de chaque utilisateur (I_u et I_v).
 
    Contrairement aux métriques built-in de Surprise, aucun min_support
    n'est appliqué : l'absence de co-ratings est précisément l'information
    qu'ITR exploite via sim_TRIANGLE sur l'union.
 
    Paramètres
    ----------
    k     : nombre de voisins maximum
    min_k : nombre minimum de voisins pour faire une prédiction
    """
 
    def __init__(self, k: int = 40, min_k: int = 3):
        AlgoBase.__init__(self)
        self.k     = k
        self.min_k = min_k
 
    def fit(self, trainset):
        AlgoBase.fit(self, trainset)
 
        self.user_means        = {}
        self.user_stds         = {}
        self.user_ratings_dict = {}
 
        for inner_uid in trainset.all_users():
            ratings = trainset.ur[inner_uid]
            vals    = [r for _, r in ratings]
 
            self.user_means[inner_uid]        = np.mean(vals) if vals else trainset.global_mean
            self.user_stds[inner_uid]         = np.std(vals)  if len(vals) > 1 else 0.0
            self.user_ratings_dict[inner_uid] = {iid: r for iid, r in ratings}
 
        return self
 
    def itr_similarity(self, u: int, v: int) -> float:
        """
        Calcule la similarité ITR entre deux utilisateurs u et v.
 
        Conformément à Iftikhar et al. (2020), sim_TRIANGLE est calculée
        sur I_uv = I_u ∪ I_v. Les ratings absents sont traités comme 0,
        permettant de capturer les asymétries de couverture entre
        utilisateurs — cas ignoré par les métriques co-rating seules.
 
        Retourne 0.0 uniquement si les deux utilisateurs n'ont aucun
        rating (denom_tri == 0), ce qui est impossible en pratique sur
        un dataset filtré.
        """
        ratings_u = self.user_ratings_dict.get(u, {})
        ratings_v = self.user_ratings_dict.get(v, {})
 
        union = set(ratings_u.keys()) | set(ratings_v.keys())
 
        if not union:
            return 0.0
 
        # --- sim_TRIANGLE sur I_u ∪ I_v ---
        diff_sq   = sum(
            (ratings_u.get(i, 0.0) - ratings_v.get(i, 0.0)) ** 2
            for i in union
        )
        norm_u    = np.sqrt(sum(ratings_u.get(i, 0.0) ** 2 for i in union))
        norm_v    = np.sqrt(sum(ratings_v.get(i, 0.0) ** 2 for i in union))
        denom_tri = norm_u + norm_v
 
        if denom_tri == 0:
            return 0.0
 
        sim_triangle = 1 - np.sqrt(diff_sq) / denom_tri
 
        # --- sim_URP sur I_u et I_v complets (μ et σ globaux) ---
        mu_u  = self.user_means.get(u, self.trainset.global_mean)
        mu_v  = self.user_means.get(v, self.trainset.global_mean)
        std_u = self.user_stds.get(u, 0.0)
        std_v = self.user_stds.get(v, 0.0)
 
        exponent = -abs(mu_u - mu_v) * abs(std_u - std_v)
        sim_urp  = 1 - 1 / (1 + np.exp(exponent))
 
        return float(sim_triangle * sim_urp)
 
    def estimate(self, u, i):
        if not self.trainset.knows_user(u):
            raise PredictionImpossible("Unknown user.")
 
        neighbors = []
        for v in self.trainset.all_users():
            if v == u:
                continue
            if i not in self.user_ratings_dict.get(v, {}):
                continue
            sim = self.itr_similarity(u, v)
            if sim > 0:
                neighbors.append((sim, self.user_ratings_dict[v][i], self.user_means[v]))
 
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
# 4. USER-BASED — Cosine KNN (Surprise)
# ===========================================================================

# ===========================================================================
# 4. USER-BASED — KNNWithMeans Cosine (Surprise)
# ===========================================================================

class UserBasedCosineKNN(KNNWithMeans):
    """
    KNN user-based avec similarité Cosine via KNNWithMeans (Surprise).
    Centre par la moyenne utilisateur μ_u avant le cosine —
    équivalent de l'Adjusted Cosine côté utilisateur.

    Paramètres
    ----------
    k           : nombre de voisins maximum
    min_k       : nombre minimum de voisins pour faire une prédiction
    min_support : nombre minimum de ratings communs entre deux utilisateurs
    """

    def __init__(self, k: int = 40, min_k: int = 1, min_support: int = 1):
        KNNWithMeans.__init__(
            self,
            k=k,
            min_k=min_k,
            sim_options={
                "name":        "cosine",
                "user_based":  True,
                "min_support": min_support
            },
            verbose=False
        )


# ===========================================================================
# 5. USER-BASED — KNNWithMeans Pearson (Surprise)
# ===========================================================================

class UserBasedKNN(KNNWithMeans):
    """
    KNN user-based avec similarité Pearson via KNNWithMeans (Surprise).
    Centre par la moyenne utilisateur μ_u avant Pearson.

    Paramètres
    ----------
    k           : nombre de voisins maximum
    min_k       : nombre minimum de voisins pour faire une prédiction
    min_support : nombre minimum de ratings communs entre deux utilisateurs
    """

    def __init__(self, k: int = 40, min_k: int = 1, min_support: int = 1):
        KNNWithMeans.__init__(
            self,
            k=k,
            min_k=min_k,
            sim_options={
                "name":        "pearson",
                "user_based":  True,
                "min_support": min_support
            },
            verbose=False
        )


# ===========================================================================
# 6. ITEM-BASED — Adjusted Cosine (Surprise)
# ===========================================================================

class ItemBasedKNN(KNNWithMeans):
    """
    KNN item-based avec similarité Adjusted Cosine.

    Centre par la moyenne UTILISATEUR avant le cosine :
        sim(i,j) = cosine( (r_ui - μ_u), (r_uj - μ_u) )
    pour tous les users u qui ont noté i ET j.

    Référence : Sarwar et al. (2001).
    """

    def __init__(self, k: int = 60, min_k: int = 3):
        KNNWithMeans.__init__(
            self,
            k=k,
            min_k=min_k,
            sim_options={
                "name":       "cosine",
                "user_based": False
            },
            verbose=False
        )


# ===========================================================================
# ITEM-BASED — Cosine classique (KNNBasic) — pour comparaison
# ===========================================================================

class ItemBasedCosineKNN(KNNBasic):
    """
    KNN item-based avec similarité Cosine classique (sans centrage).
    Contrairement à l'Adjusted Cosine, ne centre pas par la moyenne utilisateur.
    Utilisé pour comparaison avec ItemBasedKNN (Adjusted Cosine).
    """

    def __init__(self, k: int = 40, min_k: int = 3):
        KNNBasic.__init__(
            self,
            k=k,
            min_k=min_k,
            sim_options={
                "name":       "cosine",
                "user_based": False
            },
            verbose=False
        )


# ===========================================================================
# 8. CONTENT-BASED — Ridge par utilisateur
# ===========================================================================

class ContentBased(AlgoBase):
    """
    Recommandeur content-based avec un modèle Ridge par utilisateur.

    Features (V1/V2/V3), alpha=1 optimal empiriquement.
    """

    def __init__(self, features_method="V3", alpha=1.0, n_topics=20, max_features=150, sbert_model="all-MiniLM-L6-v2"):

        AlgoBase.__init__(self)
        self.features_method = features_method
        self.alpha = alpha
        self.n_topics = n_topics
        self.max_features = max_features
        self.sbert_model = sbert_model
        self.content_features = self._create_content_features(features_method)

    def _create_content_features(self, method: str) -> pd.DataFrame:
        df_items = load_items()
        if method == "V1":
            return self._features_v1(df_items)
        elif method == "V2":
            return self._features_v2(df_items)
        elif method == "V3":
            return self._features_v3(df_items)
        elif method == "V4":
            return self._features_v4(df_items)
        elif method == "V5":
            return self._features_v5(df_items)
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
            tfidf = TfidfVectorizer(max_features=self.max_features, min_df=2)
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
        df["year"] = df_items[C.LABEL_COL].str.extract(r"\((\d{4})\)").astype(float)
        df["year"] = df["year"].fillna(df["year"].median())
        df["decade"] = (df["year"] // 10) * 10
        decade_dummies = pd.get_dummies(df["decade"], prefix="decade")
        decade_dummies.index = df.index
        df = df.drop(columns=["decade"]).join(decade_dummies)
        genres = df_items[C.GENRES_COL].str.get_dummies(sep="|")
        genres.index = df_items.index
        if "(no genres listed)" in genres.columns:
            genres = genres.drop(columns=["(no genres listed)"])
        df = df.join(genres)
        genres_clean = df_items[C.GENRES_COL].fillna("").replace("(no genres listed)", "")
        df["n_genres"] = genres_clean.apply(lambda x: 0 if x == "" else len(x.split("|")))
        genre_freq = genres.mean(axis=0)
        df["genre_popularity_mean"] = (
            genres.dot(genre_freq) / df["n_genres"].replace(0, np.nan)
        )
        tags_path = C.CONTENT_PATH / "tags.csv"
        if tags_path.exists():
            tags = pd.read_csv(tags_path)
            tags["tag"] = tags["tag"].fillna("").astype(str).str.lower()
            tags_by_movie = tags.groupby("movieId")["tag"].apply(
                lambda x: " ".join(x)
            ).reindex(df_items.index).fillna("")
            tfidf = TfidfVectorizer(max_features=self.max_features, min_df=2)
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
        df = df.fillna(0)
        scaler = MinMaxScaler()
        df[df.columns] = scaler.fit_transform(df)
        return df

    
    def _features_v4(self, df_items):
        """V4 = V3 + LDA sur les overviews TMDB. Ajoute des features sémantiques basées sur les topics LDA extraits des descriptions des films."""
        import warnings
        warnings.filterwarnings("ignore")

    # Base : features V3
        df = self._features_v3(df_items)

    # Charger TMDB
        tmdb_path = C.CONTENT_PATH / "tmdb_features.csv"
        if not tmdb_path.exists():
            print("  tmdb_features.csv not found, skipping LDA features")
            return df
        
        tmdb = pd.read_csv(tmdb_path)
        tmdb[C.ITEM_ID_COL] = tmdb["movieId"].astype(float).astype(int)
        tmdb = tmdb.set_index(C.ITEM_ID_COL)

    # --- LDA sur les overviews ---
        try:
            from gensim import corpora
            from gensim.models import LdaModel
            from gensim.parsing.preprocessing import (
                preprocess_string,
                strip_punctuation,
                strip_numeric,
                remove_stopwords,
                strip_short
                )

            N_TOPICS = self.n_topics  # nombre de topics LDA

        # Prétraitement des overviews
            overviews = tmdb["overview"].fillna("").reindex(
                df_items.index, fill_value=""
                )

            FILTERS = [
                strip_punctuation,
                strip_numeric,
                remove_stopwords,
                strip_short
            ]

            processed = [
                preprocess_string(str(text), FILTERS)
                for text in overviews
            ]

        # Construire le dictionnaire et le corpus
            dictionary = corpora.Dictionary(processed)
            dictionary.filter_extremes(no_below=5, no_above=0.5)
            corpus = [dictionary.doc2bow(doc) for doc in processed]

        # Entraîner LDA
            lda = LdaModel(
                corpus=corpus,
                id2word=dictionary,
                num_topics=N_TOPICS,
                random_state=42,
                passes=10,
                alpha="auto"
            )

        # Extraire les distributions de topics pour chaque film
            topic_matrix = np.zeros((len(df_items), N_TOPICS))
            for idx, bow in enumerate(corpus):
                topics = lda.get_document_topics(bow, minimum_probability=0)
                for topic_id, prob in topics:
                    topic_matrix[idx, topic_id] = prob

            df_lda = pd.DataFrame(
                topic_matrix,
                index=df_items.index,
                columns=[f"lda_topic_{i}" for i in range(N_TOPICS)]
            )
           
            df = df.join(df_lda)

        # --- Features numériques TMDB supplémentaires ---
            df["tmdb_vote_average"]  = tmdb["vote_average"].reindex(df_items.index).fillna(0)
            df["tmdb_vote_count"]    = tmdb["vote_count"].reindex(df_items.index).fillna(0)
            df["tmdb_popularity"]    = tmdb["popularity"].reindex(df_items.index).fillna(0)
            df["tmdb_runtime"]       = tmdb["runtime"].reindex(df_items.index).fillna(0)

            print(f"  LDA: {N_TOPICS} topics extracted from TMDB overviews")

        except ImportError:
            print("  gensim not installed. pip install gensim")
        except Exception as e:
            print(f"  LDA error: {e}")

    # Re-normaliser tout
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        df[df.columns] = scaler.fit_transform(df.fillna(0))
        return df
    
    def _features_v5(self, df_items):
        """ V5 = V3 + Sentence-BERT embeddings sur les overviews TMDB. Remplace LDA par des embeddings denses de 384 dimensions."""
        # Base : features V3
        df = self._features_v3(df_items)

        # Charger TMDB
        tmdb_path = C.CONTENT_PATH / "tmdb_features.csv"
        if not tmdb_path.exists():
            print("  tmdb_features.csv not found, skipping SBERT features")
            return df

        tmdb = pd.read_csv(tmdb_path)
        tmdb[C.ITEM_ID_COL] = tmdb["movieId"].astype(float).astype(int)
        tmdb = tmdb.set_index(C.ITEM_ID_COL)

        try:
            from sentence_transformers import SentenceTransformer
            print(f"  Loading SBERT model: {self.sbert_model}...")
            model = SentenceTransformer(self.sbert_model)

            # Overviews alignées avec df_items
            overviews = tmdb["overview"].reindex(
                df_items.index, fill_value=""
                ).fillna("").tolist()

            print(f"  Encoding {len(overviews)} overviews...")
            embeddings = model.encode(
                overviews,
                batch_size=64,
                show_progress_bar=False
                )

            # Créer DataFrame des embeddings
            n_dims = embeddings.shape[1]
            df_sbert = pd.DataFrame(
                embeddings,
                index=df_items.index,
                columns=[f"sbert_{i}" for i in range(n_dims)]
                )
            df = df.join(df_sbert)

            # Features numériques TMDB
            df["tmdb_vote_average"] = tmdb["vote_average"].reindex(df_items.index).fillna(0)
            df["tmdb_vote_count"]   = tmdb["vote_count"].reindex(df_items.index).fillna(0)
            df["tmdb_popularity"]   = tmdb["popularity"].reindex(df_items.index).fillna(0)
            df["tmdb_runtime"]      = tmdb["runtime"].reindex(df_items.index).fillna(0)

            print(f"  SBERT: {n_dims} dimensions extracted")

        except ImportError:
            print("  sentence-transformers not installed.")
        except Exception as e:
            print(f"  SBERT error: {e}")

        # Re-normaliser
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        df[df.columns] = scaler.fit_transform(df.fillna(0))
        return df

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

    def estimate(self, u, i):
        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            raise PredictionImpossible("User and/or item is unknown.")
        raw_item_id = self.trainset.to_raw_iid(i)
        X = self.content_features.loc[raw_item_id:raw_item_id, :].values
        score = self.user_profile[u].predict(X)[0]
        return float(np.clip(score, C.RATINGS_SCALE[0], C.RATINGS_SCALE[1]))

    def get_feature_importances(self, inner_uid: int) -> dict:
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
# 9. LATENT FACTOR — FunkSVD (Surprise)
# ===========================================================================

class SVDModel(SVD):
    """
    Factorisation matricielle FunkSVD.
    Paramètres optimaux Optuna : n_factors=100, n_epochs=80,
    lr_all=0.00757, reg_all=0.1076 => RMSE 0.881
    """

    def __init__(
        self,
        n_factors:    int   = 100,
        n_epochs:     int   = 80,
        lr_all:       float = 0.007565017481617904,
        reg_all:      float = 0.10761653019253145,
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


# ===========================================================================
# 10. HYBRID MODEL — SVD + ContentBased V3 + UserBased ITR
# ===========================================================================

class HybridModel(AlgoBase):
    """
    Modèle hybride pondéré : SVD + ContentBased V3 + UserBased ITR.

    Stratégie : combinaison linéaire pondérée (Falk, 2019 — Chapter 12).

    Poids par défaut :
        w_svd     = 0.45  — meilleur nDCG@10
        w_content = 0.30  — meilleur RMSE + cold-start
        w_user    = 0.25  — meilleure diversité (ITR)
    """

    def __init__(self,
                 w_svd:     float = 0.45,
                 w_content: float = 0.30,
                 w_user:    float = 0.25):
        AlgoBase.__init__(self)
        self.w_svd     = w_svd
        self.w_content = w_content
        self.w_user    = w_user

        self.svd     = SVDModel(
            n_factors=35, n_epochs=98,
            lr_all=0.005, reg_all=0.0934, random_state=42
        )
        self.content = ContentBased(features_method="V3", alpha=1.0)
        self.user    = UserBasedITRKNN(k=60, min_k=5)

    def fit(self, trainset):
        AlgoBase.fit(self, trainset)
        print("  Fitting SVD...")
        self.svd.fit(trainset)
        print("  Fitting ContentBased V3...")
        self.content.fit(trainset)
        print("  Fitting ITR...")
        self.user.fit(trainset)
        return self

    def estimate(self, u, i):
        scores  = []
        weights = []

        try:
            scores.append(self.svd.predict(u, i).est)
            weights.append(self.w_svd)
        except Exception:
            pass

        try:
            scores.append(self.content.predict(u, i).est)
            weights.append(self.w_content)
        except Exception:
            pass

        try:
            scores.append(self.user.predict(u, i).est)
            weights.append(self.w_user)
        except Exception:
            pass

        if not scores:
            raise PredictionImpossible("All models failed.")

        total_w = sum(weights)
        pred    = sum(s * w for s, w in zip(scores, weights)) / total_w
        return float(np.clip(pred, C.RATINGS_SCALE[0], C.RATINGS_SCALE[1]))