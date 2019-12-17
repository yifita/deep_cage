import pandas as pd
import sys
import os
import torch
import numpy as np
import openmesh as om
from pytorch_points.network.operations import faiss_knn
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR + "/..")
from common import read_trimesh

source_model = sys.argv[1]
target_model = sys.argv[2]
orig_label_path = os.path.splitext(source_model)[0]+".picked"
new_lable = os.path.splitext(target_model)[0]+".picked"

target_mesh = om.read_polymesh(target_model)
# target_mesh = om.read_trimesh(target_model)
target_shape_arr = target_mesh.points()
target_shape = target_shape_arr.copy()
target_shape = torch.from_numpy(
    target_shape[:, :3].astype(np.float32))
target_shape.unsqueeze_(0)
orig_label = pd.read_csv(orig_label_path, delimiter=" ",
                         skiprows=1, header=None)
orig_label_name = orig_label.iloc[:, 5]
source_points = torch.from_numpy(
    orig_label.iloc[:, 6:9].to_numpy().astype(np.float32))
source_points = source_points.unsqueeze(0)
# find the closest point on the original meshes
source_mesh = om.read_polymesh(source_model)
# source_mesh = om.read_trimesh(source_model)
source_shape_arr = source_mesh.points()
source_shape = source_shape_arr.copy()
source_shape = torch.from_numpy(source_shape[None, :, :3]).float()
_, idx, _ = faiss_knn(1, source_points, source_shape, NCHW=False)

target_points = torch.gather(target_shape.unsqueeze(1).expand(-1, source_points.shape[1], -1, -1), 2,
                             idx.unsqueeze(-1).expand(-1, -1, -1, 3))
# save to pd again
orig_label[9] = idx.squeeze(0).squeeze(-1)
ncol = orig_label.shape[1]
orig_label.to_csv(orig_label_path, sep=" ", header=[str(orig_label.shape[0])]+[""]*(ncol-1),index=False)
orig_label.iloc[:, 6:9] = target_points.squeeze().numpy()
orig_label.to_csv(new_lable, sep=" ", header=[str(
    orig_label.shape[0])]+[""]*(ncol-1), index=False)
