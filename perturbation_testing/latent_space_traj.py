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
from sklearn.cluster import DBSCAN
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

def do_PCA(gcls, ngcls, title, dim=2):
    """
    dim: total number of PCA components to fit. The plot always shows
    components (dim-1) and dim (1-indexed), e.g. dim=2 -> PC1 vs PC2,
    dim=4 -> PC3 vs PC4. All points (line + start + end) are sliced
    consistently using the same two columns: [dim-2, dim-1] (0-indexed).
    """
    pca = PCA(n_components=dim)
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

    x_idx, y_idx = dim - 2, dim - 1

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
                pts_2d[:, x_idx],
                pts_2d[:, y_idx],
                color=color,
                alpha=0.75,
                linewidth=1,
                marker=marker,
                label = label if first else None
            )

            plt.scatter(
                pts_2d[0, x_idx],
                pts_2d[0, y_idx],
                color='green',
                marker="o",
                s=100,
                label='start' if first_1 else None,
                zorder=2
            )

            plt.scatter(
                pts_2d[-1, x_idx],
                pts_2d[-1, y_idx],
                color='red',
                marker='s',
                s=100,
                label='end' if first_1 else None,
                zorder=2
            )

            first = False
        first_1 = False

    plt.title(title)
    plt.xlabel(f"PC {x_idx + 1}")
    plt.ylabel(f"PC {y_idx + 1}")
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
                diff = trajectories[i][j] - trajectories[i][j - 1]
                sequence.append(diff)
        all_sequences.append(sequence)
    return all_sequences


def main():
    data = pd.read_csv('Corpus/run_sentences_different_classes.csv')
    sentences = data['sentence']
    types = data['type']
    trajectories = []
    for i in range(len(sentences)):
        trajectory = [
            emb for word in incrementize(sentences[i])
            if (emb := get_cls_embedding(word)) is not None
        ]
        if len(trajectory) == 0:
            continue  # skip sentences where 'run' was never found
        # carry the raw sentence text alongside the trajectory + type so we can
        # label cluster centroids with something meaningful later on
        trajectories.append((trajectory, types[i], sentences[i]))

    color_map = {
        'imperative': 'green',
        'interrogative': 'red',
        'exclamatory': 'blue',
        'declarative': 'purple'
    }

    do_PCA_with_color_map(
        trajectories,
        color_map,
        "Trajectories of the CLS token in different sentences labeled by type (3rd and 4th Principle Components)",
        cluster_eps=0.035,
        cluster_min_samples=2,
        dim=4
    )


def do_PCA_with_color_map(trajectories, color_map, title, cluster_eps=0.5, cluster_min_samples=2,
                           label_max_chars=40, dim=2):
    """
    trajectories: list of (trajectory, sentence_type, sentence_text)
    dim: total number of PCA components to fit. The plot shows components
        (dim-1) and dim (1-indexed) -- e.g. dim=2 -> PC1 vs PC2, dim=4 ->
        PC3 vs PC4. Every point drawn (line, start marker, end marker,
        cluster centroid) is sliced from the SAME two columns
        [dim-2, dim-1] so nothing ends up mixing coordinate spaces.
    cluster_eps / cluster_min_samples: DBSCAN params, tuned in the same
        2D space that's actually being plotted (PC[dim-1] / PC[dim]),
        since clustering is done on the projected, plotted endpoints.
        You will likely need to tune cluster_eps for your own data --
        smaller eps -> more, tighter clusters.
    label_max_chars: truncate the sentence text used to label a centroid so
        the plot doesn't get overrun with text.
    """
    pca = PCA(n_components=dim)

    all_points = []
    colors = []
    sentences = []
    per_trajectory_points = []  # keep each trajectory's own (unprojected) points around for reuse

    for trajectory, sentence_type, sentence_text in trajectories:
        colors.append(color_map[sentence_type])
        sentences.append(sentence_text)
        tensors = [
            step if isinstance(step, torch.Tensor)
            else torch.tensor(step)
            for step in trajectory
        ]
        pts = torch.stack(tensors).cpu().numpy()
        all_points.append(pts)
        per_trajectory_points.append(pts)

    stacked_points = np.vstack(all_points)
    print("PCA input shape:", stacked_points.shape)
    pca.fit(stacked_points)
    plt.figure(figsize=(8, 8))

    # the two PCA columns actually being plotted -- everything below must
    # consistently index into these same two columns
    x_idx, y_idx = dim - 2, dim - 1

    # pass 1: lines + start markers, and collect projected endpoints (in the
    # SAME x_idx/y_idx plane as the lines) for clustering
    end_points_2d = []
    for i, pts in enumerate(per_trajectory_points):
        pts_2d = pca.transform(pts)
        plt.plot(pts_2d[:, x_idx], pts_2d[:, y_idx], color=colors[i], linewidth=1, alpha=0.5, marker='^', zorder=1)
        plt.scatter(pts_2d[0, x_idx], pts_2d[0, y_idx], color='green', marker="o", s=100, zorder=2)
        end_points_2d.append([pts_2d[-1, x_idx], pts_2d[-1, y_idx]])
    end_points_2d = np.array(end_points_2d)

    # pass 2: end markers
    for i, pts in enumerate(per_trajectory_points):
        pts_2d = pca.transform(pts)
        plt.scatter(pts_2d[-1, x_idx], pts_2d[-1, y_idx], color=colors[i], marker='s', s=100, zorder=3)

    # --- cluster the final points of each trajectory (in the plotted plane)
    # and label centroids ---
    clustering = DBSCAN(eps=cluster_eps, min_samples=cluster_min_samples).fit(end_points_2d)
    cluster_labels = clustering.labels_  # -1 == noise / not part of any cluster

    for cluster_id in sorted(set(cluster_labels)):
        member_idx = np.where(cluster_labels == cluster_id)[0]
        cluster_pts = end_points_2d[member_idx]
        centroid = cluster_pts.mean(axis=0)

        # representative sentence = the one whose endpoint is closest to the centroid
        dists = np.linalg.norm(cluster_pts - centroid, axis=1)
        rep_sentence = sentences[member_idx[np.argmin(dists)]]
        rep_sentence = str(rep_sentence)
        if len(rep_sentence) > label_max_chars:
            rep_sentence = rep_sentence[:label_max_chars].rstrip() + "..."

        is_noise = cluster_id == -1
        marker_color = 'gray' if is_noise else 'black'
        label_text = rep_sentence if is_noise else f"{rep_sentence} (n={len(member_idx)})"

        plt.scatter(centroid[0], centroid[1], color=marker_color, marker='X', s=160,
                    edgecolors='white', linewidths=0.8, zorder=4)
        plt.annotate(
            label_text,
            (centroid[0], centroid[1]),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=8,
            fontweight='bold',
            color=marker_color,
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=marker_color, alpha=0.8),
            zorder=5,
        )

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=color, lw=2, label=label)
        for label, color in color_map.items()
    ]
    legend_elements.append(
        Line2D([0], [0], marker='X', color='w', markerfacecolor='black',
               markeredgecolor='white', markersize=10, label='cluster centroid')
    )
    plt.legend(handles=legend_elements)
    plt.title(title)
    plt.xlabel(f"PC {x_idx + 1}")
    plt.ylabel(f"PC {y_idx + 1}")
    plt.tight_layout()
    plt.savefig("gptraj.png")
    plt.show()



def do_PCA_target(g, ng, title, words=None):
    pca = PCA(n_components=2)

    all_points = []
    for group in [g, ng]:
        for subgroup in group:
            for trajectory in subgroup:
                trajectory = [
                    step if isinstance(step, torch.Tensor)
                    else torch.tensor(step)
                    for step in trajectory
                ]
                pts = torch.stack(trajectory).cpu().numpy()
                all_points.append(pts)

    all_points = np.vstack(all_points)
    print("PCA input shape:", all_points.shape)
    pca.fit(all_points)

    plt.figure(figsize=(8, 8))

    groups = [
        (g, "blue", "^", "G"),
        (ng, "red", "^", "NG")
    ]
    first_1 = True
    for group, color, marker, label in groups:
        for word_idx, subgroup in enumerate(group):
            first = True
            for trajectory in subgroup:
                trajectory = [
                    step if isinstance(step, torch.Tensor)
                    else torch.tensor(step)
                    for step in trajectory
                ]
                pts = torch.stack(trajectory).cpu().numpy()
                pts_2d = pca.transform(pts)
                plt.plot(
                    pts_2d[:, 0], pts_2d[:, 1],
                    color=color, alpha=0.75, linewidth=1, marker=marker,
                    label=label if first else None
                )
                plt.scatter(pts_2d[0, 0], pts_2d[0, 1], color='green', marker="o",
                            s=100, label='start' if first_1 else None, zorder=2)
                plt.scatter(pts_2d[-1, 0], pts_2d[-1, 1], color='red', marker='s',
                            s=100, label='end' if first_1 else None, zorder=2)

                # --- label the cluster with its word ---
                if words is not None and word_idx < len(words):
                    plt.annotate(
                        words[word_idx],
                        (pts_2d[-1, 0], pts_2d[-1, 1]),
                        textcoords="offset points",
                        xytext=(5, 5),
                        fontsize=9,
                        fontweight='bold'
                    )
            first = False
        first_1 = False

    plt.title(title)
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.tight_layout()
    plt.savefig("gptraj.png")
    plt.show()


if __name__ == '__main__':
    main()