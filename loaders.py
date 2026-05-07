# third parties imports
import pandas as pd

# local imports
from constants import Constant as C
from surprise import Reader, Dataset # [cite: 33]
import datetime # Nécessaire pour l'exportation

from sklearn.feature_extraction.text import TfidfVectorizer

"""def load_ratings(surprise_format=False):
    df_ratings = pd.read_csv(C.EVIDENCE_PATH / C.RATINGS_FILENAME)
    if surprise_format:
        pass
    else:
        return df_ratings
"""

def load_ratings(surprise_format=False): # [cite: 37]
    df_ratings = pd.read_csv(C.EVIDENCE_PATH / C.RATINGS_FILENAME)
    
    if surprise_format: # 
        # Création du Reader avec l'échelle des notes définie dans constants.py 
        reader = Reader(rating_scale=C.RATINGS_SCALE)
        
        # Conversion du DataFrame pandas vers le format Dataset de surprise
        # Note : On sélectionne les colonnes dans l'ordre [User, Item, Rating] 
        data = Dataset.load_from_df(
            df_ratings[[C.USER_ID_COL, C.ITEM_ID_COL, C.RATING_COL]], 
            reader
        )
        return data 
    
    return df_ratings

def load_items():
    df_items = pd.read_csv(C.CONTENT_PATH / C.ITEMS_FILENAME)
    df_items = df_items.set_index(C.ITEM_ID_COL)
    return df_items


def export_evaluation_report(df):
    """ Export the report to the evaluation folder.

    The name of the report is versioned using today's date
    """
    # On s'assure que le dossier d'évaluation existe (défini dans constants.py) [cite: 97]
    C.EVALUATION_PATH.mkdir(parents=True, exist_ok=True)
    
    # On génère le nom du fichier avec la date du jour (ex: 2026_04_15.csv) [cite: 99, 100]
    date_str = datetime.datetime.now().strftime("%Y_%m_%d")
    filename = f"report_{date_str}.csv"
    
    # Exportation du DataFrame vers le dossier d'évaluation [cite: 98]
    df.to_csv(C.EVALUATION_PATH / filename, index=False)
    print(f"Rapport exporté avec succès : {filename}")
    pass

def get_tfidf_tags_features(filepath='tags.csv', min_df=5, max_df=0.5):
    """
    Extrait, nettoie et applique un TF-IDF sur les tags des utilisateurs.
    Retourne un DataFrame avec les movieId en index et les tags en colonnes.
    """
    # 1. Charger les données
    df_tags = pd.read_csv(filepath)
    
    # 2. Nettoyage : minuscules, strip, et on lie les mots composés
    df_tags['tag'] = df_tags['tag'].astype(str).str.lower().str.strip()
    df_tags['tag'] = df_tags['tag'].str.replace(' ', '_')
    
    # 3. Regrouper par film sous forme d'une phrase de tags uniques
    # Exemple de résultat : "action dark_comedy superhero"
    tags_per_movie = df_tags.groupby('movieId')['tag'].apply(lambda x: ' '.join(set(x)))
    
    # 4. Appliquer le TF-IDF
    tfidf = TfidfVectorizer(min_df=min_df, max_df=max_df)
    tfidf_matrix = tfidf.fit_transform(tags_per_movie)
    
    # 5. Créer le DataFrame final propre
    df_tags_tfidf = pd.DataFrame(
        tfidf_matrix.toarray(), 
        columns=tfidf.get_feature_names_out(), 
        index=tags_per_movie.index
    )
    
    return df_tags_tfidf