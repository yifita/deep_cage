"""
Create non-rigid ICP benchmarks
=====
1. sample vertices and additional N=2000 points
MeshSample -v source.obj vertices.pts (V points)
MeshSample -n2000 source.obj extra.pts
cat vertices.pts extra.pts > source.pts

2. sample M=3000 points on the target
MeshSample -n3000 target.obj target.pts

3. Register source.pts target.pts offsets.txt    # (and other options as needed)
take the first V points from offset.

4. save obj
"""

import multiprocessing
import subprocess
import shlex
from shutil import copy2
import os
from glob import glob
import numpy as np
from collections import OrderedDict
from multiprocessing.pool import ThreadPool
from matplotlib import cm
import matplotlib as mpl
import re
import tempfile
from pytorch_points.utils.pc_utils import save_ply, load, save_ply_with_face, read_ply_with_face
from pytorch_points.misc import logger

SAMPLE_BIN = os.path.join("/home/ywang/.local/bin/Thea/MeshSample")
REGISTER_BIN = os.path.join("/home/ywang/.local/bin/Thea/Register")
N_CORE = 8

n_sample = 2000

def call_proc(cmd):
    """ This runs in a separate thread. """
    #subprocess.call(shlex.split(cmd))  # This will block until cmd finishes
    p = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    return (out, err)


if __name__ == "__main__":
    import sys
    source_pattern = sys.argv[1]  # "*Sa.ply"
    target_pattern = sys.argv[2]  # "*Sb.ply"
    output_dir = sys.argv[3]  # output directory

    ###################################
    # 1. gather source and target
    ###################################
    source_files = sorted(glob(source_pattern))
    target_files = sorted(glob(target_pattern))
    if len(source_files) != len(target_files):
        logger.error("Unequal source and target files")
    logger.info("Found {} source and target files".format(len(source_files)))

    print("Using %d of %d cores" % (N_CORE, multiprocessing.cpu_count()))

    #######################################################
    # 1. Sample original vertices and N additional vertices in source
    # 2. Sample M points on target
    #######################################################
    pool1 = ThreadPool(processes=N_CORE)
    results = []
    vsample_dir = tempfile.mkdtemp()
    for i, source in enumerate(source_files):
        # ./MeshSample -v source output
        output_file = os.path.join(vsample_dir, str(i)+"_s_v.pts")
        results.append(pool1.apply_async(call_proc, (SAMPLE_BIN + " -v {} {}".format(source, output_file),)))
        output_file = os.path.join(vsample_dir, str(i)+"_s_n{}.pts".format(n_sample))
        results.append(pool1.apply_async(call_proc, (SAMPLE_BIN + " -n{} {} {}".format(n_sample, source, output_file),)))

    # Close the pool
    pool1.close()

    pool2 = ThreadPool(processes=N_CORE)
    for i, target in enumerate(target_files):
        # ./MeshSample -v target output
        # ./MeshSample -v source output
        output_file = os.path.join(vsample_dir, str(i)+"_t_v.pts")
        results.append(pool2.apply_async(call_proc, (SAMPLE_BIN + " -v {} {}".format(target, output_file),)))
        output_file = os.path.join(vsample_dir, str(i)+"_t_n{}.pts".format(n_sample))
        results.append(pool2.apply_async(call_proc, (SAMPLE_BIN + " -n{} {} {}".format(n_sample, target, output_file),)))

    pool2.close()
    pool1.join()
    pool2.join()
    logger.success("Sampled {}+V points from source and target".format(n_sample))
    for result in results:
        out, err = result.get()
        if len(err) > 0:
            print("err: {}".format(repr(err)))
    results.clear()

    #######################################################
    # 3. Merge _[st]_v.pts and _[st]_n{X}.pts
    #######################################################
    os.makedirs(output_dir, exist_ok=True)
    pool3 = ThreadPool(processes=N_CORE)
    source_target_samples = []
    def append_file(*args):
        try:
            with open(args[-1], "w") as f:
                for fname in args[:-1]:
                    with open(fname) as infile:
                        for line in infile:
                            f.write(line)
        except Exception as e:
            return ("", "Error with inputs: "+", ".join(args)+str(e))
        else:
            return ("", "")

    for i in range(len(source_files)):
        target_name = os.path.splitext(os.path.basename(target_files[i]))[0]
        source_name = os.path.splitext(os.path.basename(source_files[i]))[0]
        sv_file = os.path.join(vsample_dir, str(i)+"_s_v.pts")
        sn_file = os.path.join(vsample_dir, str(i)+"_s_n{}.pts".format(n_sample))
        s_file = os.path.join(output_dir, source_name+".pts")
        tv_file = os.path.join(vsample_dir, str(i)+"_t_v.pts")
        tn_file = os.path.join(vsample_dir, str(i)+"_t_n{}.pts".format(n_sample))
        t_file = os.path.join(output_dir, target_name+".pts")
        source_target_samples.append((s_file, t_file))

        if os.path.isfile(sv_file) and os.path.isfile(sn_file):
            results.append(pool3.apply_async(append_file, (sv_file, sn_file, s_file)))
        if os.path.isfile(tv_file) and os.path.isfile(tn_file):
            results.append(pool3.apply_async(append_file, (tv_file, tn_file, t_file)))

    pool3.close()
    pool3.join()
    for result in results:
        out, err = result.get()
        if len(err) > 0:
            print("err: {}".format(err))
    results.clear()
    logger.success("Merged {}+V points to {}".format(n_sample, output_dir))

    ###################################
    # Register source.pts target.pts offsets.txt
    ###################################
    pool4 = ThreadPool(processes=N_CORE)
    ref_file = None
    for source, target in source_target_samples:
        output_file = os.path.splitext(source)[0]+"_offset.pts"
        # ./Register --labels FILE1 FILE2 OUTPUT
        results.append(pool4.apply_async(call_proc, (REGISTER_BIN + " {} {} {}".format(source, target, output_file),)))

    # Close the pool4 and wait for each running task to complete
    pool4.close()
    pool4.join()

    for result in results:
        out, err = result.get()
        if len(err) > 0:
            print("err: {}".format(err))
    results.clear()
    logger.success("Registered source to target")

    # remove unused file
    call_proc("find {} -name \"*corr*.pts\" -or -name \"*offset*.pts\" -or -name \"*colored*.ply\" -delete")

    ###################################
    # save deformed source as meshes
    ###################################
    deformed = glob(os.path.join(output_dir, "*_deformed.pts"))
    logger.info("Found {} deformed and {} source files".format(len(deformed), len(source_files)))
    # assert(len(deformed) == len(source_files))
    for i in range(len(source_files)):
        try:
            source_name = os.path.splitext(os.path.basename(source_files[i]))[0]
            # remove "_deformed"
            deformed = os.path.join(output_dir, source_name+"_deformed.pts")
            # read original mesh, get faces and number vertices
            V, faces = read_ply_with_face(source_files[i])
            faces = faces[:,:3].astype(np.int64)
            npoint = V.shape[0]
            # read deformed pts
            pts = load(deformed)[:npoint,:3]
            output_name = source_name.replace("Sa", "Sab")
            save_ply_with_face(pts[:,:3], faces, os.path.join(output_dir, output_name+".ply"))
        except Exception as e:
            import pdb; pdb.set_trace()
            print(e)

    logger.success("Wrote new mesh files to {}".format(output_dir))