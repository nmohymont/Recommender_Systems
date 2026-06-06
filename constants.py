from pathlib import Path


class Constant:

    BASE_DIR = Path(__file__).resolve().parent

    DATA_PATH = BASE_DIR / 'data' / 'small'

    # Content
    CONTENT_PATH = DATA_PATH / 'content'

    # - item
    ITEMS_FILENAME = 'movies.csv'
    ITEM_ID_COL    = 'movieId'
    LABEL_COL      = 'title'
    GENRES_COL     = 'genres'

    # Evidence
    EVIDENCE_PATH = DATA_PATH / 'evidence'

    # - ratings
    RATINGS_FILENAME                = 'ratings.csv'
    RATINGS_TEST_FILENAME           = 'ratings_test.csv'
    RATINGS_WITH_IMPLICIT_FILENAME  = 'ratings_with_implicit_ilies.csv'

    USER_ID_COL      = 'userId'
    RATING_COL       = 'rating'
    TIMESTAMP_COL    = 'timestamp'
    USER_ITEM_RATINGS = [USER_ID_COL, ITEM_ID_COL, RATING_COL]

    # Rating scale
    RATINGS_SCALE = (0.5, 5.0)

    # Paths
    EVALUATION_PATH = CONTENT_PATH.parent / 'evaluation'
    TAGS_FILENAME   = 'tags.csv'
    RECS_PATH       = DATA_PATH / 'recs'

    # ------------------------------------------------------------------
    # Hybrid weights  (somme = 1.0)
    #
    # Justification :
    #   SVD          (0.40) : meilleur RMSE général, bon à la généralisation
    #   User-based   (0.25) : capture le voisinage utilisateur
    #   Item-based   (0.25) : meilleur RMSE des KNN (0.876 sur small)
    #   Content-based(0.10) : diversité + cold-start (userId=-1)
    # ------------------------------------------------------------------
    HYBRID_SVD_WEIGHT          = 0.45
    HYBRID_USER_BASED_WEIGHT   = 0.30
    HYBRID_CONTENT_BASED_WEIGHT= 0.25

    # Default top-N
    TOP_N_VALUE = 10