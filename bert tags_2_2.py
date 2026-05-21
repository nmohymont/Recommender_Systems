import os
os.environ["OMP_NUM_THREADS"] = "1"
# ... autres env vars

import numpy as np
import pandas as pd
import faiss

from sklearn.cluster import MiniBatchKMeans
from loaders import load_items
from constants import Constant as C

print("1/5 - Chargement...", flush=True)
tag_embeddings = np.load(C.CONTENT_PATH / 'tag_embeddings.npy')
df_tags   = pd.read_csv(C.CONTENT_PATH / 'genome-tags.csv')
df_scores = pd.read_csv(C.CONTENT_PATH / 'genome-scores.csv')
df_items  = load_items()

print("2/5 - KMeans avec faiss...", flush=True)

# faiss attend du float32
embeddings_f32 = tag_embeddings.astype(np.float32)
n_tags, dim = embeddings_f32.shape

kmeans_faiss = faiss.Kmeans(
    d=dim,          # dimension des vecteurs (384)
    k=1000,          # nombre de clusters
    niter=20,       # itérations (suffisant pour 1128 tags)
    verbose=True,   # affiche la progression
    seed=42
)

kmeans_faiss.train(embeddings_f32)
_, cluster_ids = kmeans_faiss.index.search(embeddings_f32, 1)
df_tags['theme_id'] = cluster_ids.flatten()

print("✅ KMeans OK", flush=True)

print("3/5 - Fusion...", flush=True)
df_merged     = df_scores.merge(df_tags[['tagId', 'theme_id']], on='tagId')
df_compressed = df_merged.groupby(['movieId', 'theme_id'])['relevance'].mean().reset_index()

print("4/5 - Pivot...", flush=True)
df_genome_features = df_compressed.pivot(index='movieId', columns='theme_id', values='relevance')
df_genome_features.columns = [f'bert_theme_{col}' for col in df_genome_features.columns]
df_features = df_genome_features.reindex(df_items.index).fillna(0)

print("5/5 - Sauvegarde...", flush=True)
df_features.to_csv(C.CONTENT_PATH / 'genome_bert_features_1000.csv')
print("✅ Fichier généré !")