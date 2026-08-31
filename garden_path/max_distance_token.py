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
from collections import Counter


embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2',
                               similarity_fn_name=SimilarityFunction.EUCLIDEAN)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--trajectory_path', type=str, default='saved_trajectories/garden_trajectories.npy')
    parser.add_argument('--g_path', type=str, default='saved_trajectories/a_trajectories.npy')
    parser.add_argument('--ng_path', type=str, default='saved_trajectories/u_trajectories.npy')
    parser.add_argument('--j', type=int, default=0)
    return parser.parse_args()


def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def euclidean_distance(a, b):
    return np.linalg.norm(a - b)


def calc_euclidean_distance(g, ng):
    return [
        euclidean_distance(a, b)
        for a, b in zip(g, ng)
    ]


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


def build_word_trajectory(sentence, word_idx):
    """
    Build the embedding trajectory of a single word (by position in the
    sentence) across every incremental prefix of the sentence.

    Mirrors the 'num_utterance' logic used elsewhere in the codebase: if the
    target word repeats earlier in the sentence, num_utterance disambiguates
    which occurrence to track (e.g. the 2nd time 'the' appears).
    """
    words = sentence.split()
    target_word = words[word_idx]

    # count how many times this exact word form has already appeared
    # before word_idx, so get_embedding tracks the correct occurrence
    prev = 1
    for h in range(word_idx):
        if words[h] == target_word:
            prev += 1

    trajectory = [
        emb for token in incrementize(sentence)
        if (emb := get_embedding(token, target_word=target_word, num_utterance=prev)) is not None
    ]
    return trajectory


def total_movement(trajectory):
    """Sum of step-to-step Euclidean distances -- how far a word's embedding
    travels over the course of the sentence being incrementally revealed."""
    if len(trajectory) < 2:
        return 0.0
    return sum(
        np.linalg.norm(trajectory[k] - trajectory[k - 1])
        for k in range(1, len(trajectory))
    )


def find_most_moving_word_idx(sentence):
    """
    Scan every word position in the sentence, build its trajectory, and
    return the index of the word whose embedding moves the most across
    increments, along with its trajectory.
    """
    words = sentence.split()
    best_idx = None
    best_trajectory = None
    best_movement = -np.inf

    for i in range(len(words)):
        trajectory = build_word_trajectory(sentence, i)
        movement = total_movement(trajectory)
        if movement > best_movement:
            best_movement = movement
            best_idx = i
            best_trajectory = trajectory

    return best_idx, best_trajectory, best_movement


def find_least_moving_word_idx(sentence):
    """
    Scan every word position in the sentence, build its trajectory, and
    return the index of the word whose embedding moves the most across
    increments, along with its trajectory.
    """
    words = sentence.split()
    best_idx = None
    best_trajectory = None
    best_movement = np.inf

    for i in range(len(words)):
        trajectory = build_word_trajectory(sentence, i)
        movement = total_movement(trajectory)
        if movement < best_movement:
            best_movement = movement
            best_idx = i
            best_trajectory = trajectory

    return best_idx, best_trajectory, best_movement


def main():
    grodner_ambiguous = pd.read_csv('Corpus/grodner_ambiguous.csv')
    grodner_unambiguous = pd.read_csv('Corpus/grodner_unambiguous.csv')
    all_similarities = []
    tokens = []
    for j in range(len(grodner_ambiguous)):
        ambiguous_sentence = grodner_ambiguous["Stimulus"][j]
        unambiguous_sentence = grodner_unambiguous["Stimulus"][j]

        # Find which word moves the most in the ambiguous (garden-path)
        # trajectory -- this is the word we'll compare across G vs. NG.
        best_idx, g, movement = find_least_moving_word_idx(ambiguous_sentence)
        if best_idx is None or g is None or len(g) == 0:
            continue
        tokens.append(ambiguous_sentence.split()[best_idx])
        print(f"[j={j}] least-moving word: '{ambiguous_sentence.split()[best_idx]}' "
              f"(idx={best_idx}, movement={movement:.4f})")

        # Track that same word position in the unambiguous sentence so the
        # two trajectories are aligned/comparable.
        ng = build_word_trajectory(unambiguous_sentence, best_idx)

        if len(ng) == 0:
            continue

        sims = calc_euclidean_distance(g, ng)

        # Skip sentences that produced no similarities
        if len(sims) > 0:
            all_similarities.append(sims)
    print(Counter(tokens).keys())
    print(Counter(tokens).values())
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
    plt.title("Average Ambiguous vs. Unambiguous Similarity (most-moving token)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()



if __name__ == '__main__':
    main()