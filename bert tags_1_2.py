from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from constants import Constant as C

df_tags = pd.read_csv(C.CONTENT_PATH / 'genome-tags.csv')
bert_model = SentenceTransformer('all-MiniLM-L6-v2')

print("Encodage BERT...", flush=True)
tag_embeddings = bert_model.encode(df_tags['tag'].tolist(), show_progress_bar=True)

np.save(C.CONTENT_PATH / 'tag_embeddings.npy', tag_embeddings)
print("✅ Embeddings sauvegardés.")