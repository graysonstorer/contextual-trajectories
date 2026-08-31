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
    grodner_ambiguous = pd.read_csv('Corpus/grodner_ambiguous.csv')
    grodner_unambiguous = pd.read_csv('Corpus/grodner_unambiguous.csv')
    args = parse_args()
    j = args.j
    trajectories = [[get_cls_embedding(token) for token in incrementize(grodner_ambiguous['Stimulus'][j])]]
    every_single_garden_path = []
    for i in range(len(grodner_ambiguous['Stimulus'][j].split())):
        prev = 1
        for h in range(i):
            if grodner_ambiguous['Stimulus'][j].split()[h] == grodner_ambiguous['Stimulus'][j].split()[i]:
                prev += 1
        temp = [[
            emb for token in incrementize(grodner_ambiguous['Stimulus'][j])
            if (emb := get_embedding(token, target_word=grodner_ambiguous['Stimulus'][j].split()[i], num_utterance = prev)) is not None
        ]]
        every_single_garden_path.append(temp)

    every_single_unambiguous = []
    for i in range(len(grodner_ambiguous['Stimulus'][j].split())):
        prev = 1
        for h in range(i):
            if grodner_ambiguous['Stimulus'][j].split()[h] == grodner_ambiguous['Stimulus'][j].split()[i]:
                prev += 1
        temp = [[
            emb for token in incrementize(grodner_unambiguous['Stimulus'][j])
            if (emb := get_embedding(token, target_word=grodner_ambiguous['Stimulus'][j].split()[i], num_utterance = prev))  is not None
        ]]
        every_single_unambiguous.append(temp)


    hinge_token_increments = incrementize(grodner_ambiguous["Stimulus"][j])
    full_sentence = hinge_token_increments[len(hinge_token_increments) - 1]
    idx = full_sentence.index(grodner_ambiguous['hinge'][j])
    full_hinge = full_sentence[idx:]

    cls_increments = incrementize(grodner_ambiguous["Stimulus"][j])
    full_cls = cls_increments[len(cls_increments) - 1]
    do_PCA_target(every_single_garden_path, every_single_unambiguous, full_cls, words=grodner_ambiguous['Stimulus'][j].split())
    # for i in range(min(len(every_single_garden_path), len(every_single_unambiguous))):
    #     do_PCA(every_single_garden_path[i], every_single_unambiguous[i], grodner_ambiguous['Stimulus'][j].split()[i])
    # calc_distances(trajectories)
    # vectors_between(trajectories)



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



# def prespec_sentence_pca(sentence_1, sentence_2):
#     all_sentence_1 = []
#     all_sentence_2 = []
#     for i in range(len(sentence_1)):
#         prev = 1
#         for h in range(i):



if __name__ == '__main__':
    main()