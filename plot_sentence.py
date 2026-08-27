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
from text_utils import *

embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2',
                               similarity_fn_name=SimilarityFunction.EUCLIDEAN)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--sentence', type=str, default='saved_trajectories/dad_joke_trajectories.npy')
    return parser.parse_args()


def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def do_PCA(trajectories_1, trajectories_2, trajectories_3, title, labels=None):
    def normalize_group(group):
        """
        get_trajectory_cls() returns a flat list of step embeddings for ONE
        sentence: [step0, step1, ...] where each step is a 1D vector.
        If that's what we got, wrap it as a single-trajectory group so the
        rest of the function can treat every group uniformly as
        list-of-trajectories, each trajectory a list-of-step-vectors.
        """
        first = np.array(group[0])
        if first.ndim == 1:
            return [group]
        return group

    trajectories_1 = normalize_group(trajectories_1)
    trajectories_2 = normalize_group(trajectories_2)
    trajectories_3 = normalize_group(trajectories_3)

    if labels is None:
        labels = ['Sentence 1', 'Sentence 2', 'Sentence 3']

    colors = ['tab:blue', 'tab:orange', 'tab:green']

    pca = PCA(n_components=2)

    all_points = []
    traj_indices = []
    group_indices = []
    for group_idx, trajectory_group in enumerate([trajectories_1, trajectories_2, trajectories_3]):
        for traj_idx, trajectory in enumerate(trajectory_group):
            for step in trajectory:
                pooled = np.array(step)
                if pooled.ndim != 1:
                    raise ValueError(
                        f"Expected a 1D embedding per step, got shape {pooled.shape} "
                        f"in group {group_idx}, trajectory {traj_idx}."
                    )
                all_points.append(pooled)
                traj_indices.append(traj_idx)
                group_indices.append(group_idx)

    all_points = np.array(all_points)  # (N, D)
    all_points_2d = pca.fit_transform(all_points)

    traj_indices = np.array(traj_indices)
    group_indices = np.array(group_indices)

    plt.figure(figsize=(8, 8))

    groups = [trajectories_1, trajectories_2, trajectories_3]
    for group_idx, trajectories in enumerate(groups):
        color = colors[group_idx % len(colors)]
        for traj_idx in range(len(trajectories)):
            mask = (group_indices == group_idx) & (traj_indices == traj_idx)
            pts = all_points_2d[mask]
            if len(pts) == 0:
                continue

            line_label = labels[group_idx] if traj_idx == 0 else None

            plt.plot(pts[:, 0], pts[:, 1], color=color, linewidth=1.5,
                      zorder=1, label=line_label)
            plt.scatter(pts[0, 0], pts[0, 1], color=color, marker='o',
                        s=200, edgecolors='black', zorder=2)
            plt.scatter(pts[-1, 0], pts[-1, 1], color=color, marker='X',
                        s=200, edgecolors='black', zorder=2)

    plt.title(title)
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.tight_layout()
    plt.legend()
    plt.show()


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


def main():
    args = parse_args()
    sentence_1 = "The detective said there had been a murder."
    sentence_2 = "The detective said there had been a murder of crows."
    sentence_3 = "The detective said there had been a murder of crows, which explained the feathers and blood."

    # print(incrementize(sentence))
    trajectory_cls = get_trajectory_cls(incrementize(sentence_1))
    trajectory_cls_2 = get_trajectory_cls(incrementize(sentence_2))
    trajectory_cls_3 = get_trajectory_cls(incrementize(sentence_3))
    # trajectory_run = get_trajectory(incrementize(sentence))
    # title = sentence_3
    # do_PCA(trajectory_cls, trajectory_cls_2, trajectory_cls_3, title)
    # calc_distances(trajectories)
    # vectors_between(trajectories)
    title = sentence_3
    do_PCA(trajectory_cls, trajectory_cls_2, trajectory_cls_3, title,
           labels=[sentence_1, sentence_2, sentence_3])


if __name__ == '__main__':
    main()