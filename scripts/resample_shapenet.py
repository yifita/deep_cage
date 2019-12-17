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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR + "/..")

from common import find_files

def call_proc(cmd):
    """ This runs in a separate thread. """
    #subprocess.call(shlex.split(cmd))  # This will block until cmd finishes
    p = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    return (out, err)


if __name__ == "__main__":
    N_CORE = 8
    N_POINT = 5000
    print("Using %d of %d cores" % (N_CORE, multiprocessing.cpu_count()))

    source_dir = sys.argv[1]  # input directoy
    output_dir = sys.argv[2]  # output directory

    SAMPLE_BIN = os.path.join("/home/ywang/.local/bin/Thea/MeshSample")

    ###################################
    # 1. gather source and target
    ###################################
    source_files = find_files(source_dir, 'obj')
    logger.info("Found {} source files".format(len(source_files)))

    os.makedirs(output_dir, exist_ok=True)

    ###################################
    # Sample
    ###################################
    pool = ThreadPool()
    results = []
    for input_file in source_files:
        source_name = os.path.splitext(os.path.basename(input_file))[0]
        my_out_dir =  os.path.join(output_dir, os.path.relpath(os.path.dirname(input_file), source_dir))
        os.makedirs(my_out_dir, exist_ok=True)
        output_file = os.path.join(my_out_dir, source_name+".pts")
        # ./MeshSample -n2048 INPUT OUTPUT
        results.append(pool.apply_async(call_proc, (SAMPLE_BIN + " -n{} {} {}".format(N_POINT, input_file, output_file),)))

    # Close the pool and wait for each running task to complete
    pool.close()
    pool.join()
    for result in results:
        out, err = result.get()
        if len(err) > 0:
            print("err: {}".format(err))
    results.clear()
