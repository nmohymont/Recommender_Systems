# standard library imports
from collections import defaultdict
import os
from pathlib import Path


# third parties imports
import numpy as np
import pandas as pd
import random as rd
from surprise import AlgoBase
from surprise import KNNWithMeans
from surprise import SVD
from surprise import PredictionImpossible
from surprise import dump

from sklearn.linear_model import LinearRegression,Lasso, ElasticNet, Ridge,ElasticNetCV, Ridge, RidgeCV

from sklearn.svm import LinearSVR
from sklearn.model_selection import GridSearchCV

from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,MinMaxScaler

from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.ensemble import RandomForestRegressor

from loaders import load_ratings, load_items, get_tfidf_tags_features

from constants import Constant as C

def get_top_n(predictions, n):
    """Return the top-N recommendation for each user from a set of predictions.
    Source: inspired by https://github.com/NicolasHug/Surprise/blob/master/examples/top_n_recommendations.py
    and modified by cvandekerckh for random tie breaking

    Args:
        predictions(list of Prediction objects): The list of predictions, as
            returned by the test method of an algorithm.
        n(int): The number of recommendation to output for each user. Default
            is 10.
    Returns:
    A dict where keys are user (raw) ids and values are lists of tuples:
        [(raw item id, rating estimation), ...] of size n.
    """

    rd.seed(0)

    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        rd.shuffle(user_ratings)
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n


# First algorithm
class ModelBaseline1(AlgoBase):
    def __init__(self):
        AlgoBase.__init__(self)

    def estimate(self, u, i):
        return 2 


# Second algorithm
class ModelBaseline2(AlgoBase):
    def __init__(self):
        AlgoBase.__init__(self)

    def fit(self, trainset):
        AlgoBase.fit(self, trainset)
        rd.seed(0)

    def estimate(self, u, i):
        return rd.uniform(self.trainset.rating_scale[0], self.trainset.rating_scale[1])


# Third algorithm
class ModelBaseline3(AlgoBase):
    def __init__(self):
        AlgoBase.__init__(self)

    def fit(self, trainset):
        AlgoBase.fit(self, trainset)
        self.the_mean = np.mean([r for (_, _, r) in self.trainset.all_ratings()])

        return self

    def estimate(self, u, i):
        return self.the_mean


# Fourth Model
class ModelBaseline4(SVD):
    def __init__(self):
        SVD.__init__(self, n_factors=100)

class ContentBased(AlgoBase):
    def __init__(self, features_method, regressor_method):
        AlgoBase.__init__(self)
        self.features_method = features_method #ligne ajoutée pour sauvegarder la features_method dans l'objet avec self.
        self.regressor_method = regressor_method
        self.content_features = self.create_content_features(features_method)

    def create_content_features(self, features_method):
        """Content Analyzer"""
        df_items = load_items()
        if features_method is None:
            df_features = None

        elif features_method == "title_length": # a naive method that creates only 1 feature based on title length
            df_features = df_items[C.LABEL_COL].apply(lambda x: len(x)).to_frame('n_character_title')

        
        elif features_method == 'date':
            # 1. Extraction de l'année exacte
            df_year = df_items[C.LABEL_COL].str.extract(r'\((\d{4})\)').astype(float)
            df_year.columns = ['release_year']
            df_year['release_year'] = df_year['release_year'].fillna(df_year['release_year'].median())
            
            # 2. Conversion en Décennies (ex: 1987 divisé par 10 (sans reste) * 10 = 1980)
            df_year['decade'] = (df_year['release_year'] // 10) * 10
            
            # 3. Création des colonnes 0 et 1 (One-Hot Encoding)
            # Ça va créer des colonnes : decade_1970.0, decade_1980.0, decade_1990.0, etc.
            df_features = pd.get_dummies(df_year['decade'], prefix='decade', dtype=float)

            df_features = df_features.drop(
                columns=[c for c in df_features.columns 
                        if any(x in c for x in ['1900', '1910'])],
                errors='ignore'  # pas d'erreur si la colonne n'existe pas
            )
    
    
        elif features_method =="date_and_genres":
            df_year = self.create_content_features('date')
            
            
            # 2. Les genres 
            df_genres = df_items[C.GENRES_COL].str.get_dummies(sep='|')

            # Feature Engineering (Top 5 Combinaisons) ---
            # On crée de nouvelles colonnes basées sur la multiplication des genres simples
            df_genres['Comedy_Drama'] = df_genres['Comedy'] * df_genres['Drama']
            df_genres['Drama_Romance'] = df_genres['Drama'] * df_genres['Romance']
            df_genres['Comedy_Romance'] = df_genres['Comedy'] * df_genres['Romance']
            df_genres['Comedy_Drama_Romance'] = df_genres['Comedy'] * df_genres['Drama'] * df_genres['Romance']
            df_genres['Drama_Thriller'] = df_genres['Drama'] * df_genres['Thriller']

            # 3. Assemblage du tableau final
            df_features = pd.concat([df_year, df_genres], axis=1)

        elif features_method == 'tags':
            
            df_items = load_items()
            tags_path = C.CONTENT_PATH/ C.TAGS_FILENAME

            if tags_path.exists():
                tags = pd.read_csv(tags_path)
                tags["tag"] = tags["tag"].fillna("").astype(str).str.lower()

                # 1. Regroupement des tags par film
                tags_by_movie = tags.groupby("movieId")["tag"].apply(lambda x: " ".join(x))
                tags_by_movie = tags_by_movie.reindex(df_items.index).fillna("")

                # 2. Application du TF-IDF
                tfidf = TfidfVectorizer(max_features=150, min_df=2)
                tags_matrix = tfidf.fit_transform(tags_by_movie)

                # 3. Création du DataFrame de features
                # On utilise 'user_tag_' pour ne pas écraser les tags du Genome !
                df_features = pd.DataFrame(
                    tags_matrix.toarray(),
                    index=df_items.index,
                    columns=[f"user_tag_{c}" for c in tfidf.get_feature_names_out()]
                )

                # 4. Ajout des Meta-features
                n_tags = tags.groupby("movieId").size().reindex(df_items.index).fillna(0) 
                df_features["user_n_tags"] = n_tags

                tag_freq = tags["tag"].value_counts()
                tags["tag_frequency"] = tags["tag"].map(tag_freq)
                avg_tag_freq = tags.groupby("movieId")["tag_frequency"].mean()
                df_features["user_avg_tag_frequency"] = avg_tag_freq.reindex(df_items.index).fillna(0)

                # =========================================================
                # 🛡️ SÉCURITÉ MIN-MAX POUR L'ELASTICNET
                # =========================================================
                # On normalise uniquement les 2 grosses colonnes (TF-IDF est déjà entre 0 et 1)
                colonnes_a_normaliser = ["user_n_tags", "user_avg_tag_frequency"]
                
                for col in colonnes_a_normaliser:
                    min_val = df_features[col].min()
                    max_val = df_features[col].max()
                    if max_val > min_val:
                        df_features[col] = (df_features[col] - min_val) / (max_val - min_val)
                    else:
                        df_features[col] = 0.0


        elif features_method == 'genres_tfidf':
            
            # 1. Remplacer les '|' par des espaces pour faire des "phrases" de genres
            # "Action|Sci-Fi" devient "Action Sci-Fi"
            genres_as_sentences = df_items[C.GENRES_COL].str.replace('|', ' ')
            
            # 2. Initialiser le TF-IDF (pas besoin de min_df ou max_df car on a seulement ~20 genres de base)
            tfidf_genres = TfidfVectorizer()
            
            # 3. Entraîner et transformer
            tfidf_matrix = tfidf_genres.fit_transform(genres_as_sentences)
            
            # 4. Créer le DataFrame propre (avec un préfixe pour s'y retrouver plus tard)
            df_features = pd.DataFrame(
                tfidf_matrix.toarray(), 
                columns=[f"genre_{g}" for g in tfidf_genres.get_feature_names_out()], 
                index=df_items.index
            )
            
        elif features_method == 'genres_tfidf_feature_engineering':
            
            # 1. Préparation du texte
            genres_as_sentences = df_items[C.GENRES_COL].str.replace('|', ' ')
            
            # 2. Application du TF-IDF
            tfidf_genres = TfidfVectorizer()
            tfidf_matrix = tfidf_genres.fit_transform(genres_as_sentences).toarray() # On passe en array
            base_genre_names = tfidf_genres.get_feature_names_out()
            
            # 3. --- LA MAGIE DES INTERACTIONS ---
            # degree=2 : Croiser les genres 2 par 2 maximum
            # interaction_only=True : Ne pas faire (Action * Action), juste (Action * Comédie)
            # include_bias=False : Ne pas rajouter de colonne remplie de "1"
            poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
            
            # Création de la méga-matrice avec toutes les multiplications !
            poly_matrix = poly.fit_transform(tfidf_matrix)
            
            # Récupération des noms générés (ex: "action", "action comedy")
            raw_poly_names = poly.get_feature_names_out(base_genre_names)
            
            # 4. Nettoyage des noms pour que ce soit joli (et utile pour la tâche 9 'Explain')
            # "action comedy" -> "genre_action_comedy"
            clean_names = [f"genre_{name.replace(' ', '_')}" for name in raw_poly_names]
            
            # 5. Création du DataFrame final
            df_features = pd.DataFrame(
                poly_matrix, 
                columns=clean_names, 
                index=df_items.index
            )
        elif features_method == 'nb_ratings':
            ratings = load_ratings()  
            df_items = load_items() # ⚠️ INDISPENSABLE : On charge la liste de TOUS les films
        
            # 1. Calcul de base (ignore les films sans note)
            movie_stats = ratings.groupby(C.ITEM_ID_COL)[C.RATING_COL].agg(
                movie_n_ratings="count",
                movie_mean_rating="mean",
                movie_std_rating="std"
            )

            # 2. LA MAGIE ANTI-NAN : On aligne sur l'index de TOUS les films
            # Les films sans note vont apparaître remplis de NaN. On va les nettoyer.
            movie_stats = movie_stats.reindex(df_items.index)

            # 3. La moyenne mondiale
            global_mean = ratings[C.RATING_COL].mean()

            # 4. Remplissage intelligent des NaN pour les films "fantômes"
            movie_stats["movie_n_ratings"] = movie_stats["movie_n_ratings"].fillna(0)
            movie_stats["movie_mean_rating"] = movie_stats["movie_mean_rating"].fillna(global_mean) # Un film inconnu vaut la moyenne
            movie_stats["movie_std_rating"] = movie_stats["movie_std_rating"].fillna(0) # Pas de variance s'il n'y a pas de notes

            # 5. Calcul des Super-Features (Maintenant c'est 100% sûr)
            m = 10
            movie_stats["movie_bayesian_mean"] = (
                (movie_stats["movie_n_ratings"] * movie_stats["movie_mean_rating"] + m * global_mean)
                / (movie_stats["movie_n_ratings"] + m)
            )

            movie_stats["movie_inverse_popularity"] = 1 / (movie_stats["movie_n_ratings"] + 1)

            df_features = movie_stats
        
        elif features_method == 'genome':
            import os # Au cas où il ne soit pas importé en haut
            
            df_items = load_items()
            df_genome = pd.read_csv(C.CONTENT_PATH / 'genome-scores.csv')

            # 1. Pivot classique
            df_features = df_genome.pivot(
                index='movieId', columns='tagId', values='relevance')
            
            # 2. Renommer les colonnes avec 'tag_'
            df_features.columns = [f'tag_{col}' for col in df_features.columns]

            # 3. Calcul des statistiques (SANS le .reindex ici, pour éviter les NaNs !)
            df_features["genome_mean"] = df_features.mean(axis=1)
            df_features["genome_std"] = df_features.std(axis=1)
            df_features["genome_max"] = df_features.max(axis=1)
            df_features["genome_min"] = df_features.min(axis=1)
            df_features["genome_n_strong"] = (df_features > 0.8).sum(axis=1)

            # 4. On réaligne tout sur l'index de df_items et on remplit les trous par 0
            df_features = df_features.reindex(df_items.index).fillna(0)

        
        elif features_method == 'genome_bert':
            
            # 1. On charge la liste de tes films pour avoir la référence
            df_items = load_items()
            
            # 2. On lit le fichier pré-calculé (Pense bien au index_col='movieId' !)
            # Si ton fichier est dans le même dossier, tu mets juste le nom.
            # S'il est dans ton dossier de données, utilise C.CONTENT_PATH / 'genome_bert_features.csv'
            df_bert = pd.read_csv(C.CONTENT_PATH /'genome_bert_features_500.csv', index_col='movieId')

            # --- PARTIE B : LES TAGS BRUTS (MICRO) ---
            # On charge les scores originaux
            df_scores = pd.read_csv(C.CONTENT_PATH / 'genome-scores.csv')
            
            # On fait le pivot classique pour avoir les 1128 colonnes
            df_raw = df_scores.pivot(index='movieId', columns='tagId', values='relevance')
            
            # On calcule la variance de chaque tag sur l'ensemble des films
            tag_variances = df_raw.var()
            
            # MAGIE ICI : On ne garde que le Top 200 des tags avec la plus forte variance
            # Tu peux ajuster ce chiffre (ex: 150, 300)
            top_200_tags = tag_variances.nlargest(200).index
            df_raw_top = df_raw[top_200_tags]
            
            # On renomme proprement pour ne pas confondre avec BERT
            df_raw_top.columns = [f'raw_tag_{col}' for col in df_raw_top.columns]
            
            # --- PARTIE C : LA FUSION (HYBRIDATION) ---
            # On colle les 500 colonnes BERT avec les 200 colonnes brutes (Total : 700 colonnes)
            df_hybrid = pd.concat([df_bert, df_raw_top], axis=1)
            
            # On réaligne sur les index de tes films et on remplit les NaN par 0
            df_features = df_hybrid.reindex(df_items.index).fillna(0)
            
            # 3. On réaligne parfaitement sur les films du dataset (au cas où il en manquerait)
            df_features = df_features.reindex(df_items.index).fillna(0)

        # On regroupe les 3 méthodes en utilisant la valeur de features_method
        elif features_method in ['visuals_log', 'visuals_quantile', 'visuals_quantilelog']:
            
            # On détermine le bon fichier en fonction de la méthode demandée
            if features_method == 'visuals_log':
                filename = 'LLVisualFeatures13K_Log.csv'
            elif features_method == 'visuals_quantile':
                filename = 'LLVisualFeatures13K_Quantile.csv'
            else:
                filename = 'LLVisualFeatures13K_QuantileLog.csv'
                
            visuals_path = C.CONTENT_PATH / 'visuals' / filename
            df_visuals = pd.read_csv(visuals_path)
            df_visuals = df_visuals.rename(columns={"ML_Id": "movieId", "ML_ID": "movieId"})
            df_visuals = df_visuals.set_index("movieId")

            # Expected columns: f1 to f7
            if all(col in df_visuals.columns for col in ["f1", "f2", "f3", "f4", "f5", "f6", "f7"]):
                df_visuals["visual_fast_paced"] = df_visuals["f4"] / (df_visuals["f1"] + 1e-6)  # motion mean / average shot_length
                df_visuals["visual_action_intensity"] = df_visuals["f4"] + df_visuals["f7"]  # motion mean + number of shots
                df_visuals["visual_color_complexity"] = df_visuals["f2"] + df_visuals["f3"]  # color_mean + color_std
                df_visuals["visual_motion_complexity"] = df_visuals["f4"] + df_visuals["f5"]  # motion_mean + motion_std
                df_visuals["visual_dark_score"] = 1 - df_visuals["f6"]  # 1 - lighting

            # On ajoute le préfixe "visual_" partout
            df_visuals.columns = [f"visual_{c}" if not str(c).startswith("visual_") else c for c in df_visuals.columns]

            
            df_features = df_visuals.reindex(df_items.index).fillna(0)
            
            
        elif features_method == 'all_features':

            df_date = self.create_content_features('date')
            df_genome = self.create_content_features('genome')
            df_visuals = self.create_content_features('visuals_log')
            df_ratings = self.create_content_features('nb_ratings')
            df_tags = self.create_content_features('tags')
            df_genres = self.create_content_features('genres_tfidf')
            
            
            df_features = pd.concat([df_date, df_genome, df_ratings, df_visuals,df_tags, df_genres], axis=1) #df_date, , df_ratings, df_bayesian

            # Normalisation uniquement sur les features hors [0,1]
            cols_a_normaliser = [c for c in df_features.columns
                                if any(c.startswith(p) for p in
                                ['log_nb', 'mean_rating', 'std_rating',
                                'bayesian', 'vis_'])]

            if cols_a_normaliser:
                scaler = MinMaxScaler()
                df_features[cols_a_normaliser] = scaler.fit_transform(
                    df_features[cols_a_normaliser])

            # =========================================================
            # 🧹 FEATURE SELECTION — importance + stabilité
            # =========================================================
            chemin_coefs = 'MES_feature_coefficients.csv'

            if Path(chemin_coefs).exists():
                print("\n[INFO] Application du filtrage des features...")
                df_coefs = pd.read_csv(chemin_coefs)

                seuil_importance  = 0.02  # ← à tuner 0.02 (médiane)
                seuil_stabilite   = 1.35   # ← à tuner (cv_coef < 2.0)

                # Double filtre : importante ET stable
                if 'cv_coef' in df_coefs.columns:
                    features_a_garder = df_coefs[
                        (df_coefs['mean_abs_coef'] > seuil_importance) &
                        (df_coefs['cv_coef']       < seuil_stabilite)
                    ]['feature'].tolist()
                else:
                    # Fallback si ancien CSV sans cv_coef
                    features_a_garder = df_coefs[
                        df_coefs['mean_abs_coef'] > seuil_importance
                    ]['feature'].tolist()

                features_valides = [f for f in features_a_garder if f in df_features.columns]
                df_features = df_features[features_valides]

                print(f"[INFO] Features conservées : {len(features_valides)}")
                print(f"       dont retirées (instables) : "
                    f"{len(features_a_garder) - len(features_valides)}\n")
            else:
                print("\n[INFO] Pas de fichier de coefficients — toutes les features utilisées.\n")
        
        elif features_method == "V3":
            df_features = pd.DataFrame(index=df_items.index)

            # 1. Year
            df_features["year"] = df_items[C.LABEL_COL].str.extract(r"\((\d{4})\)").astype(float)
            df_features["year"] = df_features["year"].fillna(df_features["year"].median())

            # 2. Decade (one-hot encoding)
            df_features["decade"] = (df_features["year"] // 10) * 10
            decade_dummies = pd.get_dummies(df_features["decade"], prefix="decade")
            decade_dummies.index = df_features.index
            df_features = df_features.drop(columns=["decade"]).join(decade_dummies)

            # 3. Genres (one-hot encoding)
            genres_dummies = df_items[C.GENRES_COL].str.get_dummies(sep="|")
            genres_dummies.index = df_items.index

            if "(no genres listed)" in genres_dummies.columns:
                genres_dummies = genres_dummies.drop(columns=["(no genres listed)"])

            df_features = df_features.join(genres_dummies)

            # 4. Genre meta-features
            genres_clean = df_items[C.GENRES_COL].fillna("").replace("(no genres listed)", "")

            df_features["n_genres"] = genres_clean.apply(
                lambda x: 0 if x == "" else len(x.split("|")))  # number of genres (movie is specialized or hybrid)

            genre_freq = genres_dummies.mean(axis=0)
            genre_rarity = 1 - genre_freq

            df_features["genre_popularity_mean"] = genres_dummies.dot(genre_freq) / df_features["n_genres"].replace(0, np.nan)  # popularity of the genre

            # 5. Tags TF-IDF + tag meta-features
            tags_path = C.CONTENT_PATH / "tags.csv"

            if tags_path.exists():
                tags = pd.read_csv(tags_path)
                tags["tag"] = tags["tag"].fillna("").astype(str).str.lower()

                tags_by_movie = tags.groupby("movieId")["tag"].apply(lambda x: " ".join(x))
                tags_by_movie = tags_by_movie.reindex(df_items.index).fillna("")

                tfidf = TfidfVectorizer(max_features=150, min_df=2)
                tags_matrix = tfidf.fit_transform(tags_by_movie)

                df_tags = pd.DataFrame(
                    tags_matrix.toarray(),
                    index=df_items.index,
                    columns=[f"tag_{c}" for c in tfidf.get_feature_names_out()])

                df_features = df_features.join(df_tags)

                n_tags = tags.groupby("movieId").size().reindex(df_items.index).fillna(0)  # number of tags
                df_features["n_tags"] = n_tags

                tag_freq = tags["tag"].value_counts()
                tags["tag_frequency"] = tags["tag"].map(tag_freq)

                avg_tag_freq = tags.groupby("movieId")["tag_frequency"].mean()
                df_features["avg_tag_frequency"] = avg_tag_freq.reindex(df_items.index).fillna(0)

            # 6. Genome scores + aggregate genome features
            genome_path = C.CONTENT_PATH / "genome-scores.csv"

            if genome_path.exists():
                genome_scores = pd.read_csv(genome_path)

                genome_features = genome_scores.pivot(
                    index="movieId",
                    columns="tagId",
                    values="relevance")

                genome_features.columns = [f"genome_{c}" for c in genome_features.columns]

                df_features["genome_mean"] = genome_features.mean(axis=1).reindex(df_items.index)
                df_features["genome_std"] = genome_features.std(axis=1).reindex(df_items.index)
                df_features["genome_max"] = genome_features.max(axis=1).reindex(df_items.index)
                df_features["genome_min"] = genome_features.min(axis=1).reindex(df_items.index)
                df_features["genome_n_strong"] = (genome_features > 0.8).sum(axis=1).reindex(df_items.index)

                df_features = df_features.join(genome_features, how="left")

            # 7. Visual features + derived visual features
            visual_path = C.CONTENT_PATH / "LLVisualFeatures13K_QuantileLog.csv"

            if visual_path.exists():
                visuals = pd.read_csv(visual_path)
                visuals = visuals.rename(columns={"ML_Id": "movieId", "ML_ID": "movieId"})
                visuals = visuals.set_index("movieId")

                # Expected columns: f1 to f7
                if all(col in visuals.columns for col in ["f1", "f2", "f3", "f4", "f5", "f6", "f7"]):
                    visuals["visual_fast_paced"] = visuals["f4"] / (visuals["f1"] + 1e-6)  # motion mean / average shot_length
                    visuals["visual_action_intensity"] = visuals["f4"] + visuals["f7"]  # motion mean + number of shots
                    visuals["visual_color_complexity"] = visuals["f2"] + visuals["f3"]  # color_mean + color_std
                    visuals["visual_motion_complexity"] = visuals["f4"] + visuals["f5"]  # motion_mean + motion_std
                    visuals["visual_dark_score"] = 1 - visuals["f6"]  # 1 - lighting

                visuals.columns = [f"visual_{c}" if not str(c).startswith("visual_") else c for c in visuals.columns]

                df_features = df_features.join(visuals, how="left")

            # 8. Rating-based global movie features
            ratings_path = C.EVIDENCE_PATH / C.RATINGS_FILENAME

            if ratings_path.exists():
                ratings = pd.read_csv(ratings_path)

                movie_stats = ratings.groupby(C.ITEM_ID_COL)[C.RATING_COL].agg(
                    movie_n_ratings="count",
                    movie_mean_rating="mean",
                    movie_std_rating="std")

                movie_stats["movie_std_rating"] = movie_stats["movie_std_rating"].fillna(0)

                global_mean = ratings[C.RATING_COL].mean()
                m = 10

                movie_stats["movie_bayesian_mean"] = (
                    (movie_stats["movie_n_ratings"] * movie_stats["movie_mean_rating"] + m * global_mean)
                    / (movie_stats["movie_n_ratings"] + m))

                movie_stats["movie_inverse_popularity"] = 1 / (movie_stats["movie_n_ratings"] + 1)

                df_features = df_features.join(movie_stats, how="left")

            # 10. Missing values
            df_features = df_features.fillna(0)

            # 11. Normalize everything
            scaler = MinMaxScaler()
            df_features[df_features.columns] = scaler.fit_transform(df_features)
        else: 
            pass # (implement other feature creations here)
            
            raise NotImplementedError(f'Feature method {features_method} not yet implemented')
        return df_features
    
    def prepare_user_data(self, u):
        """Prépare et retourne les features (X) et la cible (y) pour un utilisateur donné."""
        feature_names = self.content_features.columns.tolist() 
        ratings_list = self.trainset.ur[u]
        
        df_user = pd.DataFrame(ratings_list, columns=['item_id', 'user_ratings'])
        df_user['item_id'] = df_user['item_id'].map(self.trainset.to_raw_iid)
        
        df_user = df_user.merge(
            self.content_features,
            how='left',
            left_on='item_id',
            right_index=True
        )
        
        X = df_user[feature_names]
        y = df_user['user_ratings'].values
        return X, y

    def fit(self, trainset):
        """Profile Learner"""
        AlgoBase.fit(self, trainset)
        
        # Preallocate user profiles
        self.user_profile = {u: None for u in trainset.all_users()}

        tous_les_coefficients =[]

        if self.regressor_method == 'random_score':
            pass
        
        elif self.regressor_method == 'random_sample':
            for u in self.user_profile:
                self.user_profile[u] = [rating for _, rating in self.trainset.ur[u]]
        
        elif self.regressor_method == 'linear_regression':
            for u in self.user_profile:
                X, y = self.prepare_user_data(u)
                reg = LinearRegression(fit_intercept=True)
                reg.fit(X, y)
                self.user_profile[u] = reg
                
        elif self.regressor_method == 'lasso_regression':
            for u in self.user_profile:
                X, y = self.prepare_user_data(u)
                reg = Lasso(alpha=0.1, fit_intercept=True)
                reg.fit(X, y)
                self.user_profile[u] = reg
                
        elif self.regressor_method == 'elasticnet':
            for u in self.user_profile:
                X, y = self.prepare_user_data(u)
                reg = ElasticNet(alpha=10.0, l1_ratio=0.5, fit_intercept=True, max_iter=5000)
                reg.fit(X, y)
                self.user_profile[u] = reg

        elif self.regressor_method == 'elasticnet_auto':
            from sklearn.linear_model import ElasticNet, ElasticNetCV

            tous_les_coefficients = []
            
            for u in self.user_profile:
                X, y = self.prepare_user_data(u)
                n_samples = len(y)
                    
                # --- NIVEAU 1 : PRUDENCE (5 à 200 films) ---
                if n_samples < 200:
                    # Ratio N/P trop faible pour une validation croisée stable.
                    # On fige le modèle sur TES valeurs championnes !
                    reg = ElasticNet(alpha=0.01, l1_ratio=0.8, fit_intercept=True, random_state=42, max_iter=2000)
                   
                # --- NIVEAU 2 : STANDARD (200 à 700 films) ---
                elif 200 <= n_samples < 700:
                    # On lance la recherche (cv=3) autour de tes valeurs.
                    # alphas : on teste 0.01, mais on garde des pénalités de sécurité (0.1, 1.0) au cas où.
                    reg = ElasticNetCV(
                        l1_ratio=[ 0.1, 0.3], 
                        alphas=[0.01, 0.05, 0.1],
                        cv=3, 
                        random_state=42,
                        max_iter=2000,
                        n_jobs=-2
                    )
                    
                # --- NIVEAU 3 : EXPERT (> 700 films) ---
                elif 700 <= n_samples < 1300 :
                    # Les Power Users ! On augmente la précision (cv=5).
                    # CORRECTION ALPHA : On autorise des alpha très faibles (0.001) 
                    # pour relâcher la pénalité car ils ont beaucoup de données.
                    reg = RidgeCV( 
                        alphas=[24],                        
                    )
                
                # --- Niveau 4 : Super Expert
                else :
                    # Ici, on sait que la variance est stable (std ~0.8).
                    # On utilise un alpha très protecteur pour éviter de "chasser le bruit".
                    # On teste des valeurs beaucoup plus hautes que prévu.
                    reg = RidgeCV(
                        alphas=[ 50], 
                    )


                reg.fit(X, y)
                self.user_profile[u] = reg

                # =========================================================
                # 🕵️ EXTRACTION DES COEFFICIENTS POUR CET UTILISATEUR
                # =========================================================
                
                # Pas de Pipeline, donc reg est directement le modèle (Ridge ou ElasticNet)
                coefs = reg.coef_
                
                dict_coefs_user = {'userId': u}
                for nom_feature, poids in zip(X.columns, coefs):  # ← X.columns fonctionne directement
                    dict_coefs_user[nom_feature] = poids

                tous_les_coefficients.append(dict_coefs_user)
                                    
                # Ajout à la liste globale
                tous_les_coefficients.append(dict_coefs_user)

            # =========================================================
            # 📊 AGRÉGATION GLOBALE (En dehors de la boucle 'for u')
            # =========================================================
            if tous_les_coefficients:
                print("\nGénération du rapport d'importance des features...")
                
                df_all_coefs = pd.DataFrame(tous_les_coefficients).drop(columns=['userId'])
                mean_coefs = df_all_coefs.mean()
                mean_abs_coefs = df_all_coefs.abs().mean()
                
                df_global_importance = pd.DataFrame({
                    'feature': mean_coefs.index,
                    'mean_abs_coef': mean_abs_coefs.values,
                    'mean_coef': mean_coefs.values
                })
                
                df_global_importance = df_global_importance.sort_values(by='mean_abs_coef', ascending=False)
                chemin_coefs = 'MES_feature_coefficients.csv'
                if not os.path.exists(chemin_coefs):
                    df_global_importance.to_csv(chemin_coefs, index=False)
                    print("✅ Fichier de référence créé.")
                else:
                    print("⚠️ Fichier de référence déjà existant — non écrasé.")

        elif self.regressor_method == 'linear_svr':
            for u in self.user_profile:
                X, y = self.prepare_user_data(u)
                reg = LinearSVR(epsilon=0.5, C=0.5, fit_intercept=True, max_iter=5000)
                reg.fit(X, y)
                self.user_profile[u] = reg

        elif self.regressor_method == 'linear_svr_auto':
            
            
            for u in self.user_profile:
                X, y = self.prepare_user_data(u)
                n_samples = len(y)
                
                base_svr = LinearSVR(random_state=42, max_iter=5000)
                
                # --- NIVEAU 1 : PRUDENCE (Profils peu denses) ---
                if n_samples < 200: #(q20) 20-25% des users
                    # On évite le GridSearch qui peut s'égarer sur peu de données
                    # On utilise tes valeurs "Championnes"
                    reg = LinearSVR(C=0.1, epsilon=0.25, random_state=42, max_iter=2000)
                
                # --- NIVEAU 2 : CONFIRMÉ (Profils standards) ---
                elif 200 <= n_samples < 700: # 65-70% des users q20 à q90
                    param_grid = {
                        'C': [0.1, 1.0, 5.0],
                        'epsilon': [0.1, 0.25]
                    }
                    reg = GridSearchCV(base_svr, param_grid, cv=3, scoring='neg_root_mean_squared_error',n_jobs=-2)
                
                # --- NIVEAU 3 : EXPERT (Power Users) ---
                else: # au dela de 700 10% restants
                    # Ici, on a assez de films pour être très précis
                    param_grid = {
                        'C': [1.0, 10.0, 50.0], # On teste des pénalités plus faibles
                        'epsilon': [0.05, 0.1, 0.25] # On cherche plus de précision (tube plus fin)
                    }
                    # cv=5 car on a assez de données pour diviser proprement le dataset
                    reg = GridSearchCV(base_svr, param_grid, cv=5, scoring='neg_root_mean_squared_error',n_jobs=-2)
                
                reg.fit(X, y)
                self.user_profile[u] = reg

        elif self.regressor_method == 'random_forest':
            for u in self.user_profile:
                X, y = self.prepare_user_data(u)
                reg = RandomForestRegressor(n_estimators=100, max_depth=5, min_samples_leaf=2, max_features='sqrt', random_state=42)
                reg.fit(X, y)
                self.user_profile[u] = reg
        
        elif self.regressor_method == "ridge_regression":

            tous_les_coefficients = []

            for u in self.user_profile:
                X, y = self.prepare_user_data(u)
                reg = Ridge(alpha=24)
                reg.fit(X, y)
                self.user_profile[u] = reg

                # =========================================================
                # 🕵️ EXTRACTION DES COEFFICIENTS POUR CET UTILISATEUR
                # =========================================================
                
                # Pas de Pipeline, donc reg est directement le modèle (Ridge ou ElasticNet)
                coefs = reg.coef_
                
                dict_coefs_user = {'userId': u}
                for nom_feature, poids in zip(X.columns, coefs):  # ← X.columns fonctionne directement
                    dict_coefs_user[nom_feature] = poids

                                    
                # Ajout à la liste globale
                tous_les_coefficients.append(dict_coefs_user)

            # =========================================================
            # 📊 AGRÉGATION GLOBALE (En dehors de la boucle 'for u')
            # =========================================================
            if tous_les_coefficients:
                print("\nGénération du rapport d'importance des features...")
                
                df_all_coefs = pd.DataFrame(tous_les_coefficients).drop(columns=['userId'])
                mean_coefs = df_all_coefs.mean()
                mean_abs_coefs = df_all_coefs.abs().mean()
                
                df_global_importance = pd.DataFrame({
                    'feature': mean_coefs.index,
                    'mean_abs_coef': mean_abs_coefs.values,
                    'mean_coef': mean_coefs.values,
                    'std_coef':      df_all_coefs.std(),  
                })
                
                # Coefficient de variation = instabilité relative
                df_global_importance['cv_coef'] = (
                     df_global_importance['std_coef'] / 
                    (df_global_importance['mean_abs_coef'] + 1e-8)
                )
    
                df_global_importance = df_global_importance.sort_values(by='mean_abs_coef', ascending=False)
                chemin_coefs = 'MES_feature_coefficients.csv'
                if not os.path.exists(chemin_coefs):
                    df_global_importance.to_csv(chemin_coefs, index=False)
                    print("✅ Fichier de référence créé.")
                else:
                    print("⚠️ Fichier de référence déjà existant — non écrasé.")
        elif self.regressor_method == "ridge_24":

            feature_names = self.content_features.columns.tolist()

            for u in self.user_profile:  # loop over every user in the self.user_profile dictionary

                df_user = pd.DataFrame(
                    self.trainset.ur[u],
                    columns=["inner_item_id", "user_ratings"])

                df_user["item_id"] = df_user["inner_item_id"].map(
                    self.trainset.to_raw_iid)

                df_user = df_user.merge(
                    self.content_features,
                    how="left",
                    left_on="item_id",
                    right_index=True)

                X = df_user[feature_names].values
                y = df_user["user_ratings"].values

                reg = Ridge(alpha=24)
                reg.fit(X, y)

                self.user_profile[u] = reg

        elif self.regressor_method == 'random_forest_with_selection':
            for u in self.user_profile:
                X, y = self.prepare_user_data(u)
                
                # S'il y a plus de colonnes que de données, on va filtrer !
                n_samples, n_features = X.shape
                
                # On décide de garder maximum 20 features, ou moins s'il y a peu de données
                k_best = min(20, n_features, max(1, n_samples // 2)) 
                
                # On crée un "Pipeline" : d'abord on filtre, ensuite on entraîne
                model = Pipeline([
                    ('feature_selection', SelectKBest(score_func=f_regression, k=k_best)),
                    ('rf', RandomForestRegressor(n_estimators=100, max_depth=5, min_samples_leaf=2, max_features='sqrt', random_state=42))
                ])
                
                # Fit le pipeline entier (il va sélectionner les colonnes, puis entraîner le RF)
                model.fit(X, y)
                self.user_profile[u] = model

        else:
            pass
            # (implement here the regressor fitting)  
            

        '''
        # --- AUTO-SAUVEGARDE DU MODÈLE À LA FIN DE L'ENTRAÎNEMENT ---
        dossier_destination = C.RECS_PATH
        os.makedirs(dossier_destination, exist_ok=True)
        chemin_fichier = os.path.join(dossier_destination, f"modele_{self.regressor_method}.p")
        dump.dump(chemin_fichier, algo=self)
        print(f"Modèle '{self.regressor_method}' sauvegardé avec succès sous {chemin_fichier}")
        '''
    def estimate(self, u, i):
        """Scoring component used for item filtering"""
        # First, handle cases for unknown users and items
        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            raise PredictionImpossible('User and/or item is unkown.')


        if self.regressor_method == 'random_score':
            rd.seed()
            score = rd.uniform(0.5,5)

        elif self.regressor_method == 'random_sample':
            rd.seed()
            score = rd.choice(self.user_profile[u])
        
        elif self.regressor_method in ['linear_regression', 'lasso_regression','random_forest', 'ridge_regression','elasticnet','random_forest_with_selection','elasticnet_auto','linear_svr','linear_svr_auto','ridge_24']:
            
            # 1. Convertir l'ID interne (i) en ID brut MovieLens
            raw_item_id = self.trainset.to_raw_iid(i)
            
            # 2. Récupérer les caractéristiques (features) de ce film spécifique
            # .values permet de l'extraire sous forme de tableau (array) au lieu d'un DataFrame
            X_i = self.content_features.loc[raw_item_id:raw_item_id, :].values
            
            # 3. Récupérer le modèle de régression qu'on a entraîné pour l'utilisateur 'u'
            reg = self.user_profile[u]
            
            # (Sécurité) Si pour une raison quelconque l'utilisateur n'a pas de modèle
            if reg is None:
                raise PredictionImpossible('Pas de modèle entraîné pour cet utilisateur.')
                
            # 4. Faire la prédiction
            # reg.predict() renvoie un tableau, ex: [3.45]. On veut juste le chiffre, donc on prend l'index [0]
            prediction_array = reg.predict(X_i)[0]
            score = np.clip(prediction_array, 0.5, 5)
    

        else:
            score=None
            # (implement here the regressor prediction)

        return score
