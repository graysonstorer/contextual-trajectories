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
    parser.add_argument('--ambiguous_path', type=str, default='saved_trajectories/a_trajectories.npy')
    parser.add_argument('--unambiguous_path', type=str, default='saved_trajectories/u_trajectories.npy')
    return parser.parse_args()


def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


from scipy.linalg import orthogonal_procrustes

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np


import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def resample(points, n_points=20):
    t_old = np.linspace(0,1,len(points))
    t_new = np.linspace(0,1,n_points)
    out = np.empty((n_points, points.shape[1]))
    for d in range(points.shape[1]):
        out[:,d] = np.interp(t_new,t_old,points[:,d])
    return out

def normalize_endpoints(traj):
    traj = traj.copy()
    traj -= traj[0]
    end = traj[-1]
    L = np.linalg.norm(end)
    if L>1e-8:
        traj /= L
    return traj

def prepare(trajectories,n=20):
    arr=[]
    for t in trajectories:
        t=np.vstack(t)
        t=resample(t,n)
        t=normalize_endpoints(t)
        arr.append(t)
    return np.array(arr)

def do_PCA(ambiguous,unambiguous,title):
    amb=prepare(ambiguous)
    unamb=prepare(unambiguous)

    all_points=np.vstack((amb.reshape(-1,amb.shape[-1]),
                          unamb.reshape(-1,unamb.shape[-1])))
    pca=PCA(n_components=2)
    pca.fit(all_points)

    amb2=np.array([pca.transform(x) for x in amb])
    un2=np.array([pca.transform(x) for x in unamb])

    mean_a=amb2.mean(axis=0)
    mean_u=un2.mean(axis=0)

    # force identical endpoints
    mean_a = amb2.mean(axis=0)
    mean_u = un2.mean(axis=0)

    start = mean_u[0]
    end = mean_u[-1]

    theta = np.arctan2(end[1] - start[1], end[0] - start[0])

    R = np.array([
        [np.cos(-theta), -np.sin(-theta)],
        [np.sin(-theta), np.cos(-theta)]
    ])

    for i in range(len(amb2)):
        amb2[i] = (amb2[i] - start) @ R.T + start

    for i in range(len(un2)):
        un2[i] = (un2[i] - start) @ R.T + start

    mean_a = amb2.mean(axis=0)
    mean_u = un2.mean(axis=0)

    std_a = amb2.std(axis=0)
    std_u = un2.std(axis=0)

    start = mean_u[0]
    end = mean_a[-1]

    plt.fill_between(mean_a[:,0],mean_a[:,1]-std_a[:,1],mean_a[:,1]+std_a[:,1],alpha=.2,color='blue')
    plt.fill_between(mean_u[:,0],mean_u[:,1]-std_u[:,1],mean_u[:,1]+std_u[:,1],alpha=.2,color='red')
    plt.plot(mean_a[:,0],mean_a[:,1],label='Garden Path',color='blue')
    plt.plot(mean_u[:,0],mean_u[:,1],label='Unambiguous',color='red')
    plt.scatter(*start,c='green',s=300,label='Start',zorder=2)
    plt.scatter(*end,c='red',s=300,label='End',zorder=2)
    plt.axis('equal')
    plt.legend()
    plt.title(title)
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
    ambiguous_trajectories = np.load(args.ambiguous_path, allow_pickle=True)
    unambiguous_trajectories = np.load(args.unambiguous_path, allow_pickle=True)
    do_PCA(ambiguous_trajectories, unambiguous_trajectories, "Average Trajectories, Garden Path vs. Unambiguous")


if __name__ == '__main__':
    main()