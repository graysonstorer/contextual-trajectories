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
from collections import Counter, defaultdict
from scipy.stats import ks_2samp


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


def collect_token_movements(sentences):
    """
    For every sentence and every word position within it, build that word's
    trajectory and compute its total movement. Movements are grouped by the
    word's surface form (as it literally appears, including case/punctuation
    attached, e.g. 'the' vs 'The' vs 'car.' are distinct tokens), so a word
    that recurs -- whether multiple times in one sentence or across many
    sentences -- accumulates multiple movement observations in the same
    bucket.

    Returns: dict mapping token -> list of movement values (one per
    occurrence encountered).
    """
    token_movements = defaultdict(list)

    for sentence in sentences:
        words = sentence.split()
        for i in range(len(words)):
            trajectory = build_word_trajectory(sentence, i)
            if len(trajectory) < 2:
                continue  # no movement possible with fewer than 2 points
            movement = total_movement(trajectory)
            token_movements[words[i]].append(movement)

    return token_movements


def average_token_movements(token_movements):
    """
    Collapse each token's list of movement observations down to a single
    average movement value.

    Returns: dict mapping token -> average movement (float).
    """
    return {
        token: float(np.mean(movements))
        for token, movements in token_movements.items()
        if len(movements) > 0
    }


def plot_token_movement_distribution(avg_movements, bins=20,
                                      title="Distribution of Average Token Movement"):
    """
    Given a dict mapping token -> average movement, plot a histogram showing
    the distribution of average movement values across all distinct tokens
    (i.e. how many tokens have low/high average movement, not a per-token
    bar chart).
    """
    values = list(avg_movements.values())

    plt.figure(figsize=(8, 4))
    plt.hist(values, bins=bins, color='teal', alpha=0.75, edgecolor='black')
    plt.xlabel("Average Movement")
    plt.ylabel("Number of Tokens")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_token_movement_bar(avg_movements, top_n=None,
                             title="Average Movement per Token"):
    """
    Bar chart of average movement for individual tokens, sorted descending.
    If top_n is given, only the top_n highest-movement tokens are shown
    (useful when there are many distinct tokens and a full bar chart would
    be unreadable).
    """
    sorted_items = sorted(avg_movements.items(), key=lambda kv: kv[1], reverse=True)
    if top_n is not None:
        sorted_items = sorted_items[:top_n]

    tokens = [t for t, _ in sorted_items]
    values = [v for _, v in sorted_items]

    plt.figure(figsize=(max(8, 0.4 * len(tokens)), 5))
    plt.bar(tokens, values, color='teal', alpha=0.8, edgecolor='black')
    plt.xticks(rotation=90)
    plt.ylabel("Average Movement")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def main():
    grodner_ambiguous = pd.read_csv('Corpus/ambiguous.csv')
    grodner_unambiguous = pd.read_csv('Corpus/unambiguous.csv')
    all_similarities = []
    tokens = []
    # --- distribution of average movement per token ---
    # all_sentences = list(grodner_ambiguous["Stimulus"]) + list(grodner_unambiguous["sentence"])
    token_movements_g = collect_token_movements(grodner_ambiguous['sentence'])
    avg_movements_g = average_token_movements(token_movements_g)
    token_movement_ng = collect_token_movements(grodner_unambiguous['sentence'])
    avg_movement_ng = average_token_movements(token_movement_ng)

    plot_token_movement_distribution(avg_movements_g, title = "Token Movement Distribution (garden path)")
    plot_token_movement_distribution(avg_movement_ng, title = "Token Movement Distribution (disambiguated)")
    g_values = list(avg_movements_g.values())
    ng_values = list(avg_movement_ng.values())

    ks_statistic, ks_pvalue = ks_2samp(g_values, ng_values)

    print("\nKolmogorov-Smirnov test (garden path vs. disambiguated token movement):")
    print(f"  KS statistic = {ks_statistic:.4f}")
    print(f"  p-value      = {ks_pvalue:.4g}")
    if ks_pvalue < 0.05:
        print("  -> Distributions are significantly different (p < 0.05)")
    else:
        print("  -> No significant difference detected (p >= 0.05)")

if __name__ == '__main__':
    main()