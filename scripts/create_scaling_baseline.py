"""
1. Load Sa Sb from a directory
2. Get bounding box size
3. Anisotropic scaling
4. Save
"""
import numpy as np
import os
import sys
import pymesh
from glob import glob
from pytorch_points.misc import logger
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR + "/..")
from common import read_trimesh

work_dir = sys.argv[1]
output_dir = sys.argv[2]
sources = glob(os.path.join(work_dir, "*Sa.*"))
logger.info("Found {} source files".format(len(sources)))

os.makedirs(output_dir, exist_ok=True)

for source in sources:
    target = source.replace("Sa", "Sb")
    fn = os.path.basename(target)
    fn = fn.replace("Sb", "Sab")
    V_t, F_t = read_trimesh(target, clean=True)
    V_t = V_t[:,:3]
    V_s, F_s = read_trimesh(source, clean=True)
    V_s = V_s[:,:3]

    bb_max = np.max(V_t, axis=0)
    bb_min = np.min(V_t, axis=0)
    size_t = (bb_max - bb_min)

    bb_max = np.max(V_s, axis=0)
    bb_min = np.min(V_s, axis=0)
    size_s = (bb_max - bb_min)

    V_st = (V_s * size_t/size_s)
    pymesh.save_mesh(os.path.join(output_dir, fn.replace("Sab", "Sa")), pymesh.form_mesh(V_s, F_s))
    pymesh.save_mesh(os.path.join(output_dir, fn.replace("Sab", "Sb")), pymesh.form_mesh(V_t, F_t))
    pymesh.save_mesh_raw(os.path.join(output_dir, fn), V_st, F_s)