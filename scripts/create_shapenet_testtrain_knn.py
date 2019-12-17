"""
create knn index for testing data based on training data
1. load from built pkl
2. load a list source file synsetnumber/shapename
3. find NN in the loaded pkl
"""
import sys
import os
import pickle
import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR + "/..")
from losses import LabeledChamferDistance
from datasets import ShapeNetSeg, _numpy_chamfer
labeled_chamfer_distance = LabeledChamferDistance(beta=1.0, gamma=1, delta=0)



def compute_nearest_neighbors_graph(train_datas, save_path):
    len_train = len(train_datas)
    all_datas = train_datas
    all_datas = [data[0] for data in all_datas]
    all_datas = ShapeNetSeg._uniformize_sizes(all_datas)
    dim = all_datas.shape[1]*3
    num_neighbors = min(5, len_train)
    nbrs = NearestNeighbors(n_neighbors=num_neighbors, algorithm='ball_tree', metric=_numpy_chamfer,
                            n_jobs=4).fit(all_datas.view(all_datas.size(0), -1).numpy())
    np.savez(save_path+"_knn_built.npy", database=nbrs, dim=dim)

if __name__ == "__main__":
    pkl_path = sys.argv[1]

    with open(pkl_path, "rb") as fp:
        train_datas = pickle.load(fp)

    dirname = os.path.dirname(pkl_path)
    pkl_name = os.path.splitext(os.path.basename(pkl_path))[0]
    compute_nearest_neighbors_graph(train_datas, pkl_path)