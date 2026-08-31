from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sentence_transformers import SentenceTransformer, SimilarityFunction
import argparse
import numpy as np
from text_utils import *
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.special import softmax
from model_runs import *


embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2',
                               similarity_fn_name=SimilarityFunction.EUCLIDEAN)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--trajectory_path', type=str, default='saved_trajectories/garden_trajectories.npy')
    parser.add_argument('--g_path', type=str, default='saved_trajectories/a_trajectories.npy')
    parser.add_argument('--ng_path', type=str, default='saved_trajectories/u_trajectories.npy')
    parser.add_argument('--j', type=int, default=0)
    return parser.parse_args()


def euclidean_distance(a, b):
    return np.linalg.norm(a - b)


def cosine_sim(a, b):
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return np.nan  # or return 0.0, depending on your application

    return np.dot(a, b) / (norm_a * norm_b)


def calc_distances(trajectories):
    distances = []
    for i in range(len(trajectories)):
        subdistances = []
        for j in range(len(trajectories[i])):
            if j > 0:
                subdistances.append(np.linalg.norm(trajectories[i][j][1] - trajectories[i][j - 1][1]))
        distances.append(subdistances)
    return distances


def vectors_between(trajectories):
    all_sequences = []
    for i in range(len(trajectories)):
        sequence = []
        for j in range(len(trajectories[i])):
            if j > 0:
                diff = trajectories[i][j] - trajectories[i][j - 1]  # (768,) difference vector
                sequence.append(diff)
        all_sequences.append(sequence)
    return all_sequences


def calc_cosine_sim(g, ng):
    return [
        cosine_sim(a, b)
        for a, b in zip(g, ng)
    ]


def calc_euclidean_distance(g, ng):
    return [
        euclidean_distance(a, b)
        for a, b in zip(g, ng)
    ]


def main():
    grodner_ambiguous = pd.read_csv('Corpus/grodner_ambiguous_same_index.csv')
    grodner_unambiguous = pd.read_csv('Corpus/grodner_unambiguous_same_index.csv')
    all_similarities = []

    for j in range(len(grodner_ambiguous)):
        g = [
            emb for token in incrementize(grodner_ambiguous["Stimulus"][j])
            if (emb := get_hinge_embedding(token, grodner_ambiguous["hinge"][j])) is not None
        ]

        ng = [
            emb for token in incrementize(grodner_unambiguous["Stimulus"][j])
            if (emb := get_hinge_embedding(token, grodner_ambiguous["hinge"][j])) is not None
        ]

        sims = calc_euclidean_distance(g, ng)

        # Skip sentences that produced no similarities
        if len(sims) > 0:
            all_similarities.append(sims)

    # Make sure we actually have data
    if len(all_similarities) == 0:
        raise ValueError("No cosine similarities were computed.")

    # Find the longest trajectory
    max_len = max(len(sims) for sims in all_similarities)

    # Create an array filled with NaNs
    similarities = np.full((len(all_similarities), max_len), np.nan)

    # Fill each row with one sentence's similarities
    for i, sims in enumerate(all_similarities):
        similarities[i, :len(sims)] = sims

    # Mean similarity at each increment
    mean_similarity = np.nanmean(similarities, axis=0)

    # Standard deviation at each increment
    std_similarity = np.nanstd(similarities, axis=0)

    # Number of valid observations at each increment
    n = np.sum(~np.isnan(similarities), axis=0)

    # Standard error of the mean
    sem_similarity = std_similarity / np.sqrt(n)

    # X-axis
    x = np.arange(len(mean_similarity))

    # Plot
    plt.figure(figsize=(8, 4))

    plt.plot(x, mean_similarity, linewidth=2, color='blue')
    plt.fill_between(
        x,
        mean_similarity - sem_similarity,
        mean_similarity + sem_similarity,
        alpha=0.25, color='blue'
    )

    plt.xlabel("Increment")
    plt.ylabel("Mean Euclidean Distance")
    plt.title("Average Ambiguous vs. Unambiguous Semantic Distance")
    plt.grid(True)
    plt.tight_layout()
    plt.show()




if __name__ == '__main__':
    main()