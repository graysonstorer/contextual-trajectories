from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import seaborn as sns
import sklearn
import numpy as np
import pandas as pd
import torch
import umap
import model_runs
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


data = pd.read_csv('Corpus/run_sentences_different_classes.csv')
sentences = data['sentence']
types = data['type']
final_embeddings = []
for i in range(len(sentences)):
    final_embeddings.append(model_runs.get_embedding(sentences[i], target_word='run'))
print()

import matplotlib.pyplot as plt

print(data.head())

pca = sklearn.decomposition.PCA(n_components=50)
# df = pca.fit_transform(final_embeddings)

# print(pca.explained_variance_ratio_)
#
color_map = {
    'imperative': 'green',
    'interrogative': 'red',
    'exclamatory': 'blue',
    'declarative': 'purple'
}
#
# weird_sentences = ['The water runs from the well.', 'Athletes run many successful businesses.',
#                    'I run best in cooler weather conditions.']
# weird_sentence_embeddings = []
# for i in range(len(weird_sentences)):
#     weird_sentence_embeddings.append(model_runs.get_embedding(weird_sentences[i], target_word='run'))
# weird_sentences_decomposition = pca.transform(weird_sentence_embeddings)
# weird_types = np.array(['declarative', 'declarative', 'declarative'])
#
# for weird_sentence_type, color in color_map.items():
#     mask = (weird_types == weird_sentence_type)
#     plt.scatter(
#         weird_sentences_decomposition[mask, 0],
#         weird_sentences_decomposition[mask, 1],
#         color = color,
#         marker = '^',
#         s = 100,
#         linewidths = 5,
#     )
#
# for sentence_type, color in color_map.items():
#     mask = (types == sentence_type)
#
#     plt.scatter(
#         df[mask, 0],
#         df[mask, 1],
#         color=color,
#         label=sentence_type
#     )
#
#
# plt.legend()
# plt.title('Latent Space of Run sentences')
# plt.xlabel('PCA Component 1')
# plt.ylabel('PCA Component 2')
# plt.savefig('perturbation_testing/saved_plots/latent_space.png')
# plt.show()

# reducer = umap.UMAP(n_epochs=5000)
# dfu = StandardScaler().fit_transform(final_embeddings)
# proj = reducer.fit_transform(dfu)
# print(proj.shape)
# for sentence_type, color in color_map.items():
#     mask = (types == sentence_type)
#     plt.scatter(
#         proj[mask, 0],
#         proj[mask, 1],
#         color=color,
#         label=sentence_type
#     )
#
# plt.legend()
# plt.title('UMAP Projection of Latent Space of Run sentences')
# plt.savefig('perturbation_testing/saved_plots/umap.png')
# plt.show()



reducer = umap.UMAP(min_dist=0.01, n_neighbors=20)
dfu = StandardScaler().fit_transform(pca.fit_transform(final_embeddings))
proj = reducer.fit_transform(dfu)
print(proj.shape)
for sentence_type, color in color_map.items():
    mask = (types == sentence_type)
    plt.scatter(
        proj[mask, 0],
        proj[mask, 1],
        color=color,
        label=sentence_type
    )

plt.legend()
plt.title('PCA -> UMAP Projection of Latent Space of Run sentences')
plt.savefig('perturbation_testing/saved_plots/pca_then_umap.png')
plt.show()


# PCA (10 - 5 - UMAP)


# reducer = umap.UMAP(n_epochs=5000)
# pca10 = sklearn.decomposition.PCA(n_components=10)
# pca5 = sklearn.decomposition.PCA(n_components=5)
# dfu = StandardScaler().fit_transform(pca5.fit_transform(pca10.fit_transform(final_embeddings)))
# proj = reducer.fit_transform(dfu)
# for sentence_type, color in color_map.items():
#     mask = (types == sentence_type)
#     plt.scatter(
#         proj[mask, 0],
#         proj[mask, 1],
#         color=color,
#         label=sentence_type
#     )
# plt.legend()
# plt.title('PCA10 -> PCA5 -> UMAP Projection of Latent Space of Run sentences')
# plt.savefig('perturbation_testing/saved_plots/pcapcaumap.png')
# plt.show()