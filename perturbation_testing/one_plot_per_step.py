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


def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


import torch

def do_PCA(gcls, ngcls, title):
    pca = PCA(n_components=2)
    all_points = []
    for group in [gcls, ngcls]:
        for trajectory in group:
            trajectory = [
                step if isinstance(step, torch.Tensor)
                else torch.tensor(step)
                for step in trajectory
            ]
            pts = torch.stack(trajectory).cpu().numpy()
            all_points.append(pts)
    # (total_points, embedding_dim)
    all_points = np.vstack(all_points)
    print("PCA input shape:", all_points.shape)

    pca.fit(all_points)

    plt.figure(figsize=(8, 8))

    groups = [
        (gcls, "blue", "*", "G"),
        (ngcls, "red", "*", "NG"),
    ]
    first_1 = True
    for group, color, marker, label in groups:

        first = True
        for trajectory in group:

            trajectory = [
                step if isinstance(step, torch.Tensor)
                else torch.tensor(step)
                for step in trajectory
            ]

            pts = torch.stack(trajectory).cpu().numpy()
            pts_2d = pca.transform(pts)

            plt.plot(
                pts_2d[:, 0],
                pts_2d[:, 1],
                color=color,
                alpha=0.75,
                linewidth=1,
                marker=marker,
                label = label if first else None
            )

            plt.scatter(
                pts_2d[0, 0],
                pts_2d[0, 1],
                color='green',
                marker="o",
                s=100,
                label='start' if first_1 else None,
                zorder=2
            )

            plt.scatter(
                pts_2d[-1, 0],
                pts_2d[-1, 1],
                color='red',
                marker='s',
                s=100,
                label='end' if first_1 else None,
                zorder=2
            )

            first = False
        first_1 = False

    plt.title(title)
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.legend()
    plt.tight_layout()
    plt.savefig("gptraj.png")
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
    data = pd.read_csv('Corpus/run_sentences_different_classes.csv')
    sentences = data['sentence']
    types = data['type']
    trajectories = []
    for i in range(len(sentences)):
        trajectories.append(([get_cls_embedding(word) for word in incrementize(sentences[i])], types[i]))
    color_map = {
        'imperative': 'green',
        'interrogative': 'red',
        'exclamatory': 'blue',
        'declarative': 'purple'
    }
    max_len = max(len(traj) for traj, _ in trajectories)
    for step in range(max_len):
        print(len(trajectories[step]))
        do_PCA_at_step(trajectories, color_map, step, f"step {step}")


def do_PCA_at_step(trajectories, color_map, step, title):
    """
    Plot a single PCA scatter of every sentence's embedding at one specific
    incremental step, colored by sentence type. Skips sentences whose
    trajectory hasn't reached `step` yet (shorter sentences).
    """
    pca = PCA(n_components=2)

    step_points = []
    step_colors = []

    for trajectory, sentence_type in trajectories:
        if step >= len(trajectory):
            continue  # this sentence's trajectory doesn't reach this step

        emb = trajectory[step]
        emb = emb if isinstance(emb, torch.Tensor) else torch.tensor(emb)
        step_points.append(emb.cpu().numpy())
        step_colors.append(color_map[sentence_type])
    print(f'number of points in this step: {len(step_points)}')
    if len(step_points) <= 1:
        print(f"[step {step}] not enough trajectories reach this step, skipping")
        return

    all_points = np.vstack(step_points)
    print(f"[step {step}] PCA input shape:", all_points.shape)
    pca.fit(all_points)
    pts_2d = pca.transform(all_points)

    plt.figure(figsize=(8, 8))
    plt.scatter(pts_2d[:, 0], pts_2d[:, 1], c=step_colors, s=100, alpha=0.8)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=color,
               markersize=10, label=label)
        for label, color in color_map.items()
    ]
    plt.legend(handles=legend_elements)

    plt.title(title)
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.tight_layout()
    plt.savefig(f"gptraj_step_{step}.png")
    plt.show()


if __name__ == '__main__':
    main()