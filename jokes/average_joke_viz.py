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


from scipy.linalg import orthogonal_procrustes

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np


def resample(points, n_points):
    """
    Resample a trajectory to n_points.
    Works in arbitrary dimensions.
    """
    t_old = np.linspace(0, 1, len(points))
    t_new = np.linspace(0, 1, n_points)

    new = np.empty((n_points, points.shape[1]))

    for d in range(points.shape[1]):
        new[:, d] = np.interp(t_new, t_old, points[:, d])

    return new


def rotate_to_horizontal(points, start_idx=0, end_idx=-1):
    """Rotate 2D points so the vector from points[start_idx] to points[end_idx] is horizontal."""
    start = points[start_idx]
    end = points[end_idx]
    delta = end - start
    angle = np.arctan2(delta[1], delta[0])  # angle of start->end vector

    cos_a, sin_a = np.cos(-angle), np.sin(-angle)
    rot = np.array([[cos_a, -sin_a],
                     [sin_a,  cos_a]])

    # rotate around the start point so start stays fixed
    centered = points - start
    rotated = centered @ rot.T
    return rotated + start


def do_PCA(trajectories):

    ############################################################
    # Resample every trajectory
    ############################################################

    resampled = []
    for traj in trajectories:
        traj = np.vstack(traj)
        traj = resample(traj, 15)
        resampled.append(traj)
    resampled = np.array(resampled)
    steps = np.diff(resampled, axis=1)
    # Save magnitudes
    step_lengths = np.linalg.norm(steps, axis=2, keepdims=True)
    # Normalize each step
    steps /= step_lengths + 1e-8
    # Average directions
    mean_directions = steps.mean(axis=0)
    # Renormalize
    mean_directions /= (
            np.linalg.norm(mean_directions, axis=1, keepdims=True) + 1e-8
    )
    # Average magnitudes
    mean_lengths = step_lengths.mean(axis=0)
    # Final averaged steps
    mean_steps = mean_directions * mean_lengths
    embedding_dim = resampled.shape[2]

    mean_traj = np.zeros((mean_steps.shape[0] + 1, embedding_dim))

    # start at origin
    for i in range(mean_steps.shape[0]):
        mean_traj[i + 1] = mean_traj[i] + mean_steps[i]

    ############################################################
    # Reconstruct every trajectory from its own steps
    # (all start at origin for comparison)
    ############################################################

    reconstructed = []

    for traj_steps in steps:

        traj = np.zeros((traj_steps.shape[0] + 1, embedding_dim))

        for i in range(traj_steps.shape[0]):
            traj[i + 1] = traj[i] + traj_steps[i]

        reconstructed.append(traj)

    reconstructed = np.array(reconstructed)

    ############################################################
    # Fit PCA ONCE using all reconstructed trajectories
    ############################################################

    all_points = reconstructed.reshape(-1, embedding_dim)

    pca = PCA(n_components=2)
    pca.fit(all_points)

    ############################################################
    # Project trajectories
    ############################################################

    projected = np.array([
        pca.transform(traj)
        for traj in reconstructed
    ])

    mean_2d = pca.transform(mean_traj)

    ############################################################
    # Align: rotate so start->end is horizontal, then ensure
    # the curve is concave up, applying the same transform to
    # every individual trajectory for consistency.
    ############################################################

    mean_2d = rotate_to_horizontal(mean_2d)

    baseline_y = mean_2d[0, 1]
    flip = mean_2d[:, 1].mean() > baseline_y
    if flip:
        mean_2d[:, 1] = 2 * baseline_y - mean_2d[:, 1]

    projected = np.array([rotate_to_horizontal(traj, 0, -1) for traj in projected])
    if flip:
        for traj in projected:
            traj[:, 1] = 2 * baseline_y - traj[:, 1]

    ############################################################
    # Plot
    ############################################################

    plt.figure(figsize=(8,8))

    # for traj in projected:
    #     plt.plot(
    #         traj[:,0],
    #         traj[:,1],
    #         color='gray',
    #         alpha=0.05,
    #         linewidth=1
    #     )

    # Shaded error band (std across trajectories at each timestep)
    y_std = projected[:, :, 1].std(axis=0)

    plt.fill_between(
        mean_2d[:, 0],
        mean_2d[:, 1] - y_std,
        mean_2d[:, 1] + y_std,
        color='blue',
        alpha=0.15,
        zorder=0,
        label='_nolegend_'
    )

    plt.plot(
        mean_2d[:,0],
        mean_2d[:,1],
        color='blue',
        linewidth=1,
        label='Average',
        zorder=1
    )

    plt.scatter(
        mean_2d[0, 0],
        mean_2d[0, 1],
        color='green',
        s=100,
        label='Start',
        zorder=2
    )

    plt.scatter(
        mean_2d[-1, 0],
        mean_2d[-1, 1],
        color='red',
        s=100,
        label='End',
        zorder=2
    )

    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.title("Average Joke Trajectory")
    plt.axis("equal")
    plt.legend()
    plt.tight_layout()
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
    trajectories = np.load(args.trajectory_path, allow_pickle=True)
    do_PCA(trajectories)


if __name__ == '__main__':
    main()