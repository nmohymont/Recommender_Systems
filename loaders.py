import datetime
import pandas as pd

from surprise import Reader, Dataset

from constants import Constant as C


def load_ratings(surprise_format=False, use_implicit=False):
    """
    Load MovieLens ratings.

    Parameters
    ----------
    surprise_format : bool
        If True, returns a Surprise Dataset.
    use_implicit : bool
        If True, loads ratings_with_implicit_ilies.csv.
        If False, loads the original ratings.csv.

    This keeps the original project logic while adding the implicit library option.
    """

    if use_implicit:
        ratings_path = C.EVIDENCE_PATH / C.RATINGS_WITH_IMPLICIT_FILENAME
    else:
        ratings_path = C.EVIDENCE_PATH / C.RATINGS_FILENAME

    df_ratings = pd.read_csv(ratings_path)

    if surprise_format:
        reader = Reader(rating_scale=C.RATINGS_SCALE)

        data = Dataset.load_from_df(
            df_ratings[C.USER_ITEM_RATINGS],
            reader
        )

        return data

    return df_ratings


def load_items():
    """
    Load movies with movieId as index.
    This keeps the original structure of the project.
    """

    df_items = pd.read_csv(C.CONTENT_PATH / C.ITEMS_FILENAME)
    df_items = df_items.set_index(C.ITEM_ID_COL)

    return df_items


def load_movies():
    """
    Load movies without setting movieId as index.
    Useful for the Flask app and recommendation display.
    """

    return pd.read_csv(C.CONTENT_PATH / C.ITEMS_FILENAME)


def load_library(library_filename="library_ilies.xlsx"):
    """
    Load the implicit library Excel file.
    """

    library_path = C.BASE_DIR / library_filename
    return pd.read_excel(library_path)


def export_evaluation_report(df):
    """
    Export the evaluation report to data/small/evaluation.
    """

    C.EVALUATION_PATH.mkdir(parents=True, exist_ok=True)

    date_str = datetime.datetime.now().strftime("%Y_%m_%d")
    filename = f"report_{date_str}.csv"

    output_path = C.EVALUATION_PATH / filename
    df.to_csv(output_path, index=False)

    print(f"Evaluation report exported: {output_path}")