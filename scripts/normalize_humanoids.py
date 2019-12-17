import os
from glob import glob
import torch
import numpy as np
from pytorch_points.utils.geometry_utils import compute_face_normals_and_areas
from pytorch_points.utils.pc_utils import center_bounding_box
import sys
import pymesh

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR + "/..")
from common import find_files, read_trimesh

def run(in_files, out_dir, ref_file):
    ref_v, ref_f = read_trimesh(ref_file)
    ref_v = torch.from_numpy(ref_v[:, :3]).float()
    ref_f = torch.from_numpy(ref_f).long()
    _, ref_area = compute_face_normals_and_areas(ref_v, ref_f)
    ref_area = torch.sum(ref_area, dim=-1)
    for in_file in in_files:
        v, f = read_trimesh(in_file)
        v = torch.from_numpy(v[:, :3]).float()
        f = torch.from_numpy(f).long()
        v, _, _ = center_bounding_box(v)
        _, area = compute_face_normals_and_areas(v, f)
        area = torch.sum(area, dim=-1)
        ratio = torch.sqrt(ref_area/area)
        ratio = ratio.unsqueeze(-1).unsqueeze(-1)
        v = v * ratio
        out_path = os.path.join(out_dir, os.path.basename(in_file))
        pymesh.save_mesh_raw(out_path, v.numpy(), f.numpy())
        print("saved to {}".format(out_path))


if __name__ == "__main__":
    in_file = sys.argv[1]
    ref_file = sys.argv[2]
    out_dir = sys.argv[3]
    if os.path.isdir(in_file):
        in_file = find_files(in_file, ["ply", "obj"])

    os.makedirs(out_dir, exist_ok=True)
    run(in_file, out_dir, ref_file)
