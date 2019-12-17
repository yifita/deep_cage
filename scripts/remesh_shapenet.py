import multiprocessing
import subprocess
import shlex
from shutil import copy2
import os
from glob import glob
import numpy as np
from collections import OrderedDict
from pytorch_points.utils.pc_utils import save_ply
from pytorch_points.misc import logger
from multiprocessing.pool import ThreadPool
import sys
import pymesh

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR + "/..")

from common import remesh, find_files

def remesh_and_save(input_path, output_path):
    try:
        obj = remesh(input_path)
        pymesh.save_mesh(output_path, obj, )
    except Exception as e:
        return ("", str(e))
    else:
        return ("","")

if __name__ == "__main__":
    N_CORE = 8
    N_POINT = 5000
    print("Using %d of %d cores" % (N_CORE, multiprocessing.cpu_count()))

    source_dir = sys.argv[1]  # input directoy
    output_dir = sys.argv[2]  # output directory

    ###################################
    # 1. gather source and target
    ###################################
    source_files = find_files(source_dir, 'obj')
    logger.info("Found {} source files".format(len(source_files)))

    os.makedirs(output_dir, exist_ok=True)

    ###################################
    # Sample
    ###################################
    pool = ThreadPool(processes=N_CORE)
    results = []
    for input_file in source_files:
        source_name = os.path.splitext(os.path.basename(input_file))[0]
        my_out_dir =  os.path.join(output_dir, os.path.relpath(os.path.dirname(input_file), source_dir))
        os.makedirs(my_out_dir, exist_ok=True)
        output_file = os.path.join(my_out_dir, source_name+".obj")
        results.append(pool.apply_async(remesh_and_save, (input_file, output_file)))

    # Close the pool and wait for each running task to complete
    pool.close()
    pool.join()
    for result in results:
        out, err = result.get()
        if len(err) > 0:
            print("err: {}".format(err))
    results.clear()
