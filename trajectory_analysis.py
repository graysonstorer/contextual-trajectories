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
    parser.add_argument('--trajectory_path', type=str, default='saved_trajectories/trajectories.npy')
    return parser.parse_args()


def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def do_PCA(trajectories):
    x_vector = get_cls_embedding('').numpy()
    y_vector = get_cls_embedding('').numpy()
    z_vector = get_cls_embedding('').numpy()

    # basis_matrix = np.stack([x_vector, y_vector, z_vector])
    pca = PCA(n_components=2)


    x_sim, y_sim, z_sim = [], [], []
    all_points = []
    traj_indices = []

    for traj_idx, trajectory in enumerate(trajectories):
        for step in trajectory:
            pooled = np.array(step)
            all_points.append(pooled)
            traj_indices.append(traj_idx)
            x_sim.append(cosine_sim(x_vector, pooled))
            y_sim.append(cosine_sim(y_vector, pooled))
            z_sim.append(cosine_sim(z_vector, pooled))

    all_points = np.array(all_points)  # (N, 768)
    # x_vector_pca = pca.transform(x_vector.reshape(1, -1))
    # y_vector_pca = pca.transform(y_vector.reshape(1, -1))
    # z_vector_pca = pca.transform(z_vector.reshape(1, -1))
    all_points_2d = pca.fit_transform(all_points)
    plt.figure(figsize=(8, 8))
    traj_indices = np.array(traj_indices)
    similarities = np.stack([x_sim, y_sim, z_sim], axis=1)
    # rgb_colors = softmax(similarities * 15, axis=1)
    for traj_idx in range(len(trajectories)):
        mask = traj_indices ==traj_idx
        pts = all_points_2d[mask]
        plt.scatter(pts[0, 0], pts[0, 1], color='green', marker='o', s=20)
        plt.scatter(pts[-1, 0], pts[-1, 1], color='red', marker='X', s=20)
    for traj_idx in range(len(trajectories)):
        mask = traj_indices == traj_idx
        pts = all_points_2d[mask]
        # colors = rgb_colors[mask]  # per-point colors for this trajectory
        plt.plot(pts[:, 0], pts[:, 1], color='black', alpha=0.25, linewidth=0.25)
    # for traj_idx in range(len(trajectories)):
    #     mask = traj_indices ==traj_idx
    #     pts = all_points_2d[mask]
    #     plt.scatter(pts[0, 0], pts[0, 1], color='green', marker='o', s=20)
    #     plt.scatter(pts[-1, 0], pts[-1, 1], color='red', marker='X', s=20)
    plt.title('Trajectory paths in PCA space')
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
    for traj in trajectories:
        sequence = []
        for j in range(1, len(traj)):
            curr = np.asarray(traj[j], dtype=np.float32)
            prev = np.asarray(traj[j - 1], dtype=np.float32)
            diff = curr - prev
            sequence.append(diff)
        all_sequences.append(sequence)
    return all_sequences


def diff_vector_origin(trajectories):
    all_sequences = []
    for traj in trajectories:
        sequence = []
        for j in range(1, len(traj)):
            curr = np.asarray(traj[j], dtype=np.float32)
            orig = np.asarray(traj[0], dtype=np.float32)
            diff = curr - orig
            sequence.append(diff)
        all_sequences.append(sequence)
    return all_sequences


def mag_from_origin(trajectories):
    all_mags = []
    for traj in trajectories:
        sequence = []
        for j in range(1, len(traj)):
            curr = np.asarray(traj[j], dtype=np.float32)
            orig = np.asarray(traj[0], dtype=np.float32)
            norm = np.linalg.norm(curr - orig)
            sequence.append(norm)
        all_mags.append(sequence)
    return all_mags


def main():
    args = parse_args()
    trajectories = np.load(args.trajectory_path, allow_pickle=True)
    do_PCA(trajectories)
    calc_distances(trajectories)
    vectors_between(trajectories)


if __name__ == '__main__':
    main()