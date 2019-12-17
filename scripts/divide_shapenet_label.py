import multiprocessing
import subprocess
import shlex
from shutil import copy2
import os
from glob import glob
import numpy as np
from collections import OrderedDict
from pytorch_points.utils.pc_utils import load, save_ply, save_ply_property
from multiprocessing.pool import ThreadPool
from matplotlib import cm
import matplotlib as mpl
import re

cmap = cm.get_cmap(name="seismic")
cnorm = mpl.colors.Normalize(vmin=-1.5, vmax=1.5)

def worker(input_file, output_file):
    points = load(input_file)
    labels = points[:, -1]
    save_ply_property(points[:,:3], labels, output_file, normals=points[:,3:6])

if __name__ == "__main__":
    work_dir = "/home/mnt/points/data/ShapeNet/PartSeg_v0/shapenetcore_partanno_segmentation_benchmark_v0_normal/03001627"
    output_dir = os.path.join(work_dir, "seg")
    os.makedirs(output_dir, exist_ok=True)
    ###################################
    # Sample
    ###################################
    pool = ThreadPool()
    for input_file in glob(os.path.join(work_dir, "*.txt")):
        output_file = os.path.join(output_dir, os.path.basename(input_file)[:-4]+".ply")
        pool.apply_async(worker, (input_file, output_file))
        # worker(input_file, output_file)
        # break

    # Close the pool and wait for each running task to complete
    pool.close()
    pool.join()
