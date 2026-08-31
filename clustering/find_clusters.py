import sklearn
import numpy as np
import pandas as pd
import torch
import model_runs
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
data = pd.read_csv('../Corpus/manual_run_sentences_for_classification.csv')
sentences = data['sentence']
final_embeddings = []
for i in range(len(sentences)):
    final_embeddings.append(model_runs.get_embedding(sentences[i], target_word='Run'))

n_clusters = 3
kmeans = sklearn.cluster.KMeans(n_clusters=n_clusters, random_state=42, n_init=1000)
labels = kmeans.fit_predict(final_embeddings)
centroids = kmeans.cluster_centers_
centroid_labels = [centroids[i] for i in labels]

print()

import matplotlib.pyplot as plt

data['labels'] = labels

print(data.head())


pca = sklearn.decomposition.PCA(n_components=2)
df = pca.fit_transform(final_embeddings)
# Get unique labels
u_labels = np.unique(labels)

# Plot each cluster
for i in u_labels:
    plt.scatter(df[labels == i, 0], df[labels == i, 1], label=f'Cluster {i}')

data.to_csv('../Corpus/machine_labeled.csv', index=False)

plt.legend()
plt.title('K-Means Clusters')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.savefig('../saved_plots/machine_labeled.png')
plt.show()

for i in range(1, 10):
    n_clusters = i + 1
    kmeans = sklearn.cluster.KMeans(n_clusters=n_clusters, random_state=42, n_init=1000)
    labels = kmeans.fit_predict(final_embeddings)
    centroids = kmeans.cluster_centers_
    centroid_labels = [centroids[j] for j in labels]
    silhouette_avg = sklearn.metrics.silhouette_score(final_embeddings, labels)
    print("For n_clusters = ", n_clusters, " The average silhouette score is ", silhouette_avg)
# silhouette_avg = sklearn.metrics.silhouette_score(final_embeddings, labels)
# print(
#     "For n_clusters =",
#     n_clusters,
#     "The average silhouette_score is :",
#     silhouette_avg,
# )
