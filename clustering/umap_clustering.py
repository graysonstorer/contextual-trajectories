from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import pandas as pd
import torch
import umap
import hdbscan
from sklearn.decomposition import PCA
import model_runs
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

data = pd.read_csv('Corpus/run_sentences_different_classes.csv')
sentences = data['sentence']
types = data['type']

# get_run_embedding returns a single token's embedding (shape (768,)) when a
# "run" form is found in the sentence, or None when it isn't. Skipping the
# Nones -- rather than appending them alongside real vectors -- is what keeps
# the embeddings list a uniform 2D shape for PCA/UMAP; mixing None in with
# real vectors is what produced the "inhomogeneous shape" error.
final_embeddings = []
used_sentences = []
used_types = []

for i in range(len(sentences)):
    emb = model_runs.get_run_embedding(sentences[i])
    if emb is None:
        continue
    final_embeddings.append(emb.cpu().numpy() if isinstance(emb, torch.Tensor) else np.asarray(emb))
    used_sentences.append(sentences[i])
    used_types.append(types[i])

final_embeddings = np.vstack(final_embeddings)
print("Embeddings shape:", final_embeddings.shape)
print(f"Kept {len(used_sentences)} / {len(sentences)} sentences (rest had no 'run' form found)")

pca_decomp = PCA(n_components=50).fit_transform(final_embeddings)
clusterable_embedding = umap.UMAP(n_components=2, min_dist=0.005, n_neighbors=50).fit_transform(pca_decomp)

labels = hdbscan.HDBSCAN(
    min_samples=2,
    min_cluster_size=20,
).fit_predict(clusterable_embedding)

clustered_sentences = pd.DataFrame()
clustered_sentences['sentence'] = used_sentences
clustered_sentences['label'] = labels
clustered_sentences['type'] = used_types

clustered_sentences.to_csv('clustering/UMAP_clustered_sentences.csv', index=False)

clustered = (labels >= 0)

plt.figure(figsize=(8, 8))
plt.scatter(clusterable_embedding[~clustered, 0],
            clusterable_embedding[~clustered, 1],
            color=(0.5, 0.5, 0.5),
            s=5,
            alpha=0.5,
            label='noise')

unique_clusters = sorted(c for c in set(labels) if c != -1)
cmap = plt.get_cmap('Dark2')

# Dark2 only has 8 distinct colors, so cycle through if there are more clusters
colors = {cid: cmap(i % cmap.N) for i, cid in enumerate(unique_clusters)}

for cluster_id in unique_clusters:
    member_idx = np.where(labels == cluster_id)[0]
    plt.scatter(clusterable_embedding[member_idx, 0],
                clusterable_embedding[member_idx, 1],
                color=colors[cluster_id],
                s=5,
                label=f'Cluster {cluster_id}')

label_max_chars = 40
for cluster_id in unique_clusters:
    member_idx = np.where(labels == cluster_id)[0]
    cluster_pts = clusterable_embedding[member_idx]
    centroid = cluster_pts.mean(axis=0)

    dists = np.linalg.norm(cluster_pts - centroid, axis=1)
    rep_sentence = str(used_sentences[member_idx[np.argmin(dists)]])
    if len(rep_sentence) > label_max_chars:
        rep_sentence = rep_sentence[:label_max_chars].rstrip() + "..."
    label_text = f"{rep_sentence} (n={len(member_idx)})"

    plt.scatter(centroid[0], centroid[1], color='black', marker='X', s=160,
                edgecolors='white', linewidths=0.8, zorder=4)
    plt.annotate(
        label_text,
        (centroid[0], centroid[1]),
        textcoords="offset points",
        xytext=(6, 6),
        fontsize=8,
        fontweight='bold',
        color='black',
        bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='black', alpha=0.8),
        zorder=5,
    )

plt.title('UMAP Clusters of target token embeddings')
plt.legend(markerscale=2, fontsize=8, loc='best')
plt.tight_layout()
plt.savefig('clustering/UMAP_clustered_sentences.png')
plt.show()
BB
