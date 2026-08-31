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
    parser.add_argument('--trajectory_path', type=str, default='saved_trajectories/dad_joke_trajectories.npy')
    parser.add_argument('--g_path', type=str, default='saved_trajectories/dad_joke_trajectories.npy')
    parser.add_argument('--ng_path', type=str, default='saved_trajectories/dad_joke_trajectories.npy')
    parser.add_argument('--j', type=int, default=0)
    return parser.parse_args()


def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def do_PCA(trajectories, g, ng, title):
    pca = PCA(n_components=2)
    all_points = []
    traj_indices = []

    # for traj_idx, trajectory in enumerate(g):
    #     for step in trajectory:
    #         all_points.append(np.array(step))
    #         traj_indices.append(traj_idx)

    for traj_idx, trajectory in enumerate(ng):
        for step in trajectory:
            all_points.append(np.array(step))
            traj_indices.append(traj_idx)

    all_points = np.array(all_points)
    pca.fit(all_points)  # fit on all trajectories for a shared space

    plt.figure(figsize=(8, 8))
    traj_indices = np.array(traj_indices)

    for group, color, label in [(g, 'blue', 'G'), (ng, 'red', 'NG')]:
        for traj_idx, trajectory in enumerate(group):
            pts = np.array(trajectory)
            pts_2d = pca.transform(pts)
            plt.plot(pts_2d[:, 0], pts_2d[:, 1], color=color, alpha=0.75, linewidth=1, marker='^')
            plt.scatter(pts_2d[0, 0], pts_2d[0, 1], color=color, marker='o', s=5)
            plt.scatter(pts_2d[-1, 0], pts_2d[-1, 1], color=color, marker='X', s=15,
                        label=label if traj_idx == 0 else None)

    plt.title(title)
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.tight_layout()
    # plt.legend()
    plt.savefig('joke_traj.png')
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
    j = args.j
    trajectories = np.load(args.trajectory_path, allow_pickle=True)
    jokes_list = pd.read_csv('Corpus/used_jokes.csv')['0']
    title = jokes_list[j]
    g = np.load(args.g_path, allow_pickle=True)
    ng = np.load(args.ng_path, allow_pickle=True)
    do_PCA(trajectories, trajectories[j:j+1], trajectories[j:j+1], title)
    calc_distances(trajectories)
    vectors_between(trajectories)


if __name__ == '__main__':
    main()