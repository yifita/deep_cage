import torch
import sys
import os
import traceback
import numpy as np
from glob import glob
from collections import defaultdict
from multiprocessing.pool import ThreadPool
from pytorch_points.misc import logger
from pytorch_points.utils.pc_utils import read_ply_with_face, load
from pytorch_points.network.model_loss import nndistance
from pytorch_points.network.geo_operations import CotLaplacian, UniformLaplacian

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR + "/..")

from common import find_files, call_proc, read_trimesh

N_CORE=4
N_SAMPLE=5000
SAMPLE_BIN = os.path.join("/home/ywang/.local/bin/Thea/MeshSample")

def sample_pts(input_dir, output_dir, overwrite_pts=False):
    pool1 = ThreadPool(processes=N_CORE)
    results = []
    source_files = glob(os.path.join(input_dir, "*Sa*.ply"))+glob(os.path.join(input_dir, "*Sb.ply"))
    if len(source_files) == 0:
        source_files = glob(os.path.join(input_dir, "*Sa*.obj"))+glob(os.path.join(input_dir, "*Sb.obj"))
    logger.info("Sampling {} meshes into {}".format(len(source_files),output_dir))
    for source in source_files:
        source_filename = os.path.splitext(os.path.basename(source))[0]
        # ./MeshSample -v source output
        output_file = os.path.join(output_dir, "{}.pts".format(source_filename))
        if overwrite_pts or not os.path.isfile(output_file):
            # results.append(pool1.apply_async(call_proc, (SAMPLE_BIN + " -n{} -s1 {} {}".format(N_SAMPLE, source, output_file),)))
            results.append(pool1.apply_async(call_proc, (SAMPLE_BIN + " -n{} {} {}".format(N_SAMPLE, source, output_file),)))

    # Close the pool
    pool1.close()
    pool1.join()
    for result in results:
        out, err = result.get()
        if len(err) > 0:
            print("err: {}".format(repr(err)))
    results.clear()
    return True

def svr_sample_pts(input_dir, output_dir, overwrite_pts=True):
    pool1 = ThreadPool(processes=N_CORE)
    results = []
    source_files = glob(os.path.join(input_dir, "*Sb.ply"))
    if len(source_files) == 0:
        source_files = glob(os.path.join(input_dir, "*Sb.obj"))
    target_names = np.unique(np.array([os.path.basename(p).split("-")[1] for p in source_files])).tolist()
    logger.info("Sampling {} target meshes into {}".format(len(target_names),output_dir))
    for target in target_names:
        source_filename = glob(os.path.join(input_dir, "*{}-Sb.*".format(target)))[0]
        # ./MeshSample -v source output
        output_file = os.path.join(output_dir, "{}.pts".format(target))
        if overwrite_pts or not os.path.isfile(output_file):
            # results.append(pool1.apply_async(call_proc, (SAMPLE_BIN + " -n{} -s1 {} {}".format(N_SAMPLE, source_filename, output_file),)))
            results.append(pool1.apply_async(call_proc, (SAMPLE_BIN + " -n{} {} {}".format(N_SAMPLE, source_filename, output_file),)))
    source_files = glob(os.path.join(input_dir, "*Sab*.ply"))
    if len(source_files) == 0:
        source_files = glob(os.path.join(input_dir, "*Sab*.obj"))
    logger.info("Sampling {} output meshes into {}".format(len(source_files),output_dir))
    for source in source_files:
        source_filename = os.path.splitext(os.path.basename(source))[0]
        # ./MeshSample -v source output
        output_file = os.path.join(output_dir, "{}.pts".format(source_filename))
        if overwrite_pts or not os.path.isfile(output_file):
            # results.append(pool1.apply_async(call_proc, (SAMPLE_BIN + " -n{} -s1 {} {}".format(N_SAMPLE, source, output_file),)))
            results.append(pool1.apply_async(call_proc, (SAMPLE_BIN + " -n{} {} {}".format(N_SAMPLE, source, output_file),)))
    if len(target_names) > 0:
        return True

    source_files = glob(os.path.join(input_dir, "*.obj"))+glob(os.path.join(input_dir, "*.ply"))
    logger.info("Sampling {} output meshes into {}".format(len(source_files),output_dir))
    for source in source_files:
        source_filename = os.path.splitext(os.path.basename(source))[0]
        # ./MeshSample -v source output
        output_file = os.path.join(output_dir, "{}.pts".format(source_filename))
        if overwrite_pts or not os.path.isfile(output_file):
            # results.append(pool1.apply_async(call_proc, (SAMPLE_BIN + " -n{} -s1 {} {}".format(N_SAMPLE, source, output_file),)))
            results.append(pool1.apply_async(call_proc, (SAMPLE_BIN + " -n{} {} {}".format(N_SAMPLE, source, output_file),)))

    # Close the pool
    pool1.close()
    pool1.join()
    for result in results:
        out, err = result.get()
        if len(err) > 0:
            print("err: {}".format(repr(err)))
    results.clear()
    return True

def evaluate_deformation(result_dirs, resample, mse=False, overwrite_pts=False):
    CD_name = "MSE" if mse else "CD"
    if isinstance(result_dirs, str):
        result_dirs = [result_dirs]
    ########## initialize ############
    eval_result = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))  # eval_result[metric[folder[file]]]
    cotLap = CotLaplacian()
    uniLap = UniformLaplacian()
    if resample:
        for cur_dir in result_dirs:
            pts_dir = os.path.join(cur_dir, "eval_pts")
            os.makedirs(pts_dir, exist_ok=True)
            result = sample_pts(cur_dir, pts_dir, overwrite_pts)
            if not result:
                logger.warn("Failed to sample points in {}".format(cur_dir))

    ########## load results ###########
    # find Sa.ply, Sb.ply and a list of Sab.ply
    ###################################
    [print("dir{}: {}".format(i, name)) for i, name in enumerate(result_dirs)]
    files = glob(os.path.join(result_dirs[0], "*.ply"))+glob(os.path.join(result_dirs[0], "*.obj"))
    source_files = [p for p in files if "Sa." in p]
    target_files = [p.replace("Sa", "Sb") for p in source_files]
    assert(all([os.path.isfile(f) for f in target_files]))
    logger.info("Found {} source target pairs".format(len(source_files)))
    ########## evaluation ############
    print("{}: {}".format("filename".ljust(70), " | ".join(["dir{}".format(i).rjust(45) for i in range(len(result_dirs))])))
    print("{}: {}".format(" ".ljust(70), " | ".join([(CD_name+"/CotLap/CotLapNorm/UniLap/UniLapNorm").rjust(45) for i in range(len(result_dirs))])))
    cnt = 0
    for source, target in zip(source_files, target_files):
        source_filename = os.path.basename(source)
        target_filename = os.path.basename(target)
        try:
            if resample:
                source_pts_file = os.path.join(result_dirs[0], "eval_pts", source_filename[:-4]+".pts")
                target_pts_file = os.path.join(result_dirs[0], "eval_pts", target_filename[:-4]+".pts")
                if not os.path.isfile(source_pts_file):
                    logger.warn("Cound\'t find {}. Skip to process the next.".format(source_pts_file))
                    continue
                if not os.path.isfile(target_pts_file):
                    logger.warn("Cound\'t find {}. Skip to process the next.".format(target_pts_file))
                source_pts = load(source_pts_file)
                target_pts = load(target_pts_file)
                source_pts = torch.from_numpy(source_pts[:,:3].astype(np.float32)).unsqueeze(0).cuda()
                target_pts = torch.from_numpy(target_pts[:,:3].astype(np.float32)).unsqueeze(0).cuda()

            ext = os.path.splitext(source_filename)[1]
            sab_str = source_filename.replace("Sa"+ext, "Sab*")
            outputs = [glob( os.path.join(cur_dir, sab_str) ) for cur_dir in result_dirs]
            if not all([len(o) > 0 for o in outputs]):
                logger.warn("Couldn\'t find {} in all folders, skipping to process the next".format(sab_str))
                continue

            # read Sa, Sb
            source_shape, source_face = read_trimesh(source, clean=False)
            target_shape, _ = read_trimesh(target, clean=False)
            source_shape = torch.from_numpy(source_shape[:,:3].astype(np.float32)).unsqueeze(0).cuda()
            target_shape = torch.from_numpy(target_shape[:,:3].astype(np.float32)).unsqueeze(0).cuda()
            source_face = torch.from_numpy(source_face[:,:3].astype(np.int64)).unsqueeze(0).cuda()

            # laplacian for source (fixed)
            cotLap.L = None
            ref_lap = cotLap(source_shape, source_face)
            ref_lap_norm = torch.norm(ref_lap, dim=-1)

            uniLap.L = None
            ref_ulap = uniLap(source_shape, source_face)
            ref_ulap_norm = torch.norm(ref_ulap, dim=-1)

            filename = os.path.splitext(os.path.basename(source))[0]
            for output, cur_dir in zip(outputs, result_dirs):
                if len(output)>1:
                    logger.warn("Found multiple outputs {}. Using the last one".format(output))
                if len(output) == 0:
                    logger.warn("Found no outputs for {} in {}".format(sab_str, cur_dir))
                    continue
                output = output[-1]

                output_shape, output_face = read_trimesh(output, clean=False)
                output_shape = torch.from_numpy(output_shape[:,:3].astype(np.float32)).unsqueeze(0).cuda()
                output_face = torch.from_numpy(output_face[:,:3].astype(np.int64)).unsqueeze(0).cuda()

                # chamfer
                if not mse:
                    if resample:
                        output_filename = os.path.basename(output)
                        output_pts_file = os.path.join(cur_dir, "eval_pts", output_filename[:-4]+".pts")
                        output_pts = load(output_pts_file)
                        output_pts = torch.from_numpy(output_pts[:,:3].astype(np.float32)).unsqueeze(0).cuda()
                        dist12, dist21, _, _ = nndistance(target_pts, output_pts)
                    else:
                        dist12, dist21, _, _ = nndistance(target_shape, output_shape)
                    cd = torch.mean(torch.mean(dist12, dim=-1) + torch.mean(dist21, dim=-1))
                    eval_result[cur_dir][CD_name][filename] = cd
                    eval_result[cur_dir][CD_name]["avg"] += (cd - eval_result[cur_dir][CD_name]["avg"])/(eval_result[cur_dir][CD_name]["cnt"]+1)
                    eval_result[cur_dir][CD_name]["cnt"] += 1
                else:
                    mse = torch.sum((output_shape-target_shape)**2, dim=-1).mean().item()
                    eval_result[cur_dir][CD_name][filename] = mse
                    eval_result[cur_dir][CD_name]["avg"] += (mse - eval_result[cur_dir][CD_name]["avg"])/(eval_result[cur_dir][CD_name]["cnt"]+1)
                    eval_result[cur_dir][CD_name]["cnt"] += 1


                lap = cotLap(output_shape)
                lap_loss = torch.mean((lap-ref_lap)**2).item()
                eval_result[cur_dir]["CotLap"][filename] = lap_loss
                eval_result[cur_dir]["CotLap"]["avg"] += (lap_loss - eval_result[cur_dir]["CotLap"]["avg"])/(eval_result[cur_dir]["CotLap"]["cnt"]+1)
                eval_result[cur_dir]["CotLap"]["cnt"] += 1

                lap_norm = torch.norm(lap, dim=-1)
                lap_norm_loss = torch.mean((lap_norm-ref_lap_norm).abs()).item()
                eval_result[cur_dir]["CotLapNorm"][filename] = lap_norm_loss
                eval_result[cur_dir]["CotLapNorm"]["avg"] += (lap_norm_loss - eval_result[cur_dir]["CotLapNorm"]["avg"])/(eval_result[cur_dir]["CotLapNorm"]["cnt"]+1)
                eval_result[cur_dir]["CotLapNorm"]["cnt"] += 1

                lap = uniLap(output_shape)
                lap_loss = torch.mean((lap-ref_ulap)**2).item()
                eval_result[cur_dir]["UniLap"][filename] = lap_loss
                eval_result[cur_dir]["UniLap"]["avg"] += (lap_loss - eval_result[cur_dir]["UniLap"]["avg"])/(eval_result[cur_dir]["UniLap"]["cnt"]+1)
                eval_result[cur_dir]["UniLap"]["cnt"] += 1

                lap_norm = torch.norm(lap, dim=-1)
                lap_norm_loss = torch.mean((lap_norm-ref_ulap_norm).abs()).item()
                eval_result[cur_dir]["UniLapNorm"][filename] = lap_norm_loss
                eval_result[cur_dir]["UniLapNorm"]["avg"] += (lap_norm_loss - eval_result[cur_dir]["UniLapNorm"]["avg"])/(eval_result[cur_dir]["UniLapNorm"]["cnt"]+1)
                eval_result[cur_dir]["UniLapNorm"]["cnt"] += 1


            print("{}: {}".format(filename.ljust(70), " | ".join(
                ["{:8.4g}/{:8.4g}/{:8.4g}/{:8.4g}/{:8.4g}".format(
                    eval_result[cur_dir][CD_name][filename],
                    eval_result[cur_dir]["CotLap"][filename], eval_result[cur_dir]["CotLapNorm"][filename],
                    eval_result[cur_dir]["UniLap"][filename], eval_result[cur_dir]["UniLapNorm"][filename]
                    )
                for cur_dir in result_dirs]
                ).ljust(30)))
        except Exception as e:
            traceback.print_exc(file=sys.stdout)
            logger.warn("Failed to evaluation {}. Skip to process the next.".format(source_filename))


    print("{}: {}".format("AVG".ljust(70), " | ".join(
        ["{:8.4g}/{:8.4g}/{:8.4g}/{:8.4g}/{:8.4g}".format(eval_result[cur_dir][CD_name]["avg"],
            eval_result[cur_dir]["CotLap"]["avg"], eval_result[cur_dir]["CotLapNorm"]["avg"],
            eval_result[cur_dir]["UniLap"]["avg"], eval_result[cur_dir]["UniLapNorm"]["avg"],
            )
            for cur_dir in result_dirs]
        ).ljust(30)))

    ########## write evaluation ############
    for cur_dir in result_dirs:
        for metric in eval_result[cur_dir]:
            output_file = os.path.join(cur_dir, "eval_{}.txt".format(metric))
            with open(output_file, "w") as eval_file:
                for name, value in eval_result[cur_dir][metric].items():
                    if (name != "avg" and name != "cnt"):
                        eval_file.write("{} {:8.4g}\n".format(name, value))

                eval_file.write("avg {:8.4g}".format(eval_result[cur_dir][metric]["avg"]))


def evaluate_svr(result_dirs, resample, overwrite_pts=False):
    """ ours is the first in the result dirs """
    if isinstance(result_dirs, str):
        result_dirs = [result_dirs]
    ########## initialize ############
    eval_result = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 1e10)))  # eval_result[metric[folder[file]]]
    avg_result = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))  # eval_result[metric[folder[file]]]

    cotLap = CotLaplacian()
    uniLap = UniformLaplacian()
    if resample and not opt.mse:
        for cur_dir in result_dirs:
            pts_dir = os.path.join(cur_dir, "eval_pts")
            os.makedirs(pts_dir, exist_ok=True)
            result = svr_sample_pts(cur_dir, pts_dir, overwrite_pts)
            if not result:
                logger.warn("Failed to sample points in {}".format(cur_dir))

    ########## load results ###########
    # find Sa.ply, Sb.ply and a list of Sab.ply
    ###################################
    [print("dir{}: {}".format(i, name)) for i, name in enumerate(result_dirs)]
    files = find_files(result_dirs[0], ["ply", "obj"])
    target_files = [p for p in files if "Sb." in p]
    target_names = np.unique(np.array([os.path.basename(p).split("-")[1] for p in target_files])).tolist()
    logger.info("Found {} target files".format(len(target_names)))

    ########## evaluation ############
    print("{}: {}".format("filename".ljust(70), " | ".join(["dir{}".format(i).rjust(20) for i in range(len(result_dirs))])))
    print("{}: {}".format(" ".ljust(70), " | ".join(["CD/HD".rjust(20) for i in range(len(result_dirs))])))
    cnt = 0
    for target in target_names:
        # 1. load ground truth
        gt_path = glob(os.path.join(result_dirs[0], "*-{}-Sb.*".format(target)))[0]
        try:
            gt_shape, gt_face = read_trimesh(gt_path, clean=False)
            if resample:
                gt_pts_file = os.path.join(result_dirs[0], "eval_pts", "{}.pts".format(target))
                if not os.path.isfile(gt_pts_file):
                    logger.warn("Cound\'t find {}. Skip to process the next.".format(gt_pts_file))
                    continue
                gt_pts = load(gt_pts_file)
                gt_pts = torch.from_numpy(gt_pts[:,:3].astype(np.float32)).unsqueeze(0).cuda()

            ours_paths = glob(os.path.join(result_dirs[0], "*-{}-Sab.*".format(target)))
            others_path = [glob( os.path.join(cur_dir, "{}.*".format(target)) ) for cur_dir in result_dirs[1:]]

            # 2. evaluate ours, all *-{target}-Sab
            if len(ours_paths) == 0:
                logger.warn("Cound\'t find {}. Skip to process the next.".format(os.path.join(result_dirs[0], "*-{}-Sab.*".format(target))))
                continue

            for ours in ours_paths:
                # load shape and points
                output_shape, output_face = read_trimesh(ours, clean=False)
                ours = os.path.basename(ours)
                cur_dir = result_dirs[0]
                if resample:
                    output_pts_file = os.path.join(cur_dir, "eval_pts", ours[:-4]+".pts")
                    if not os.path.isfile(output_pts_file):
                        logger.warn("Cound\'t find {}. Skip to process the next source.".format(output_pts_file))
                        continue
                    output_pts = load(output_pts_file)
                    output_pts = torch.from_numpy(output_pts[:,:3].astype(np.float32)).unsqueeze(0).cuda()
                    # compute chamfer
                    dist12, dist21, _, _ = nndistance(gt_pts, output_pts)
                    cd = torch.mean(torch.mean(dist12, dim=-1) + torch.mean(dist21, dim=-1)).item()
                    hd = max(torch.max(dist12).item(), torch.max(dist21).item())
                else:
                    dist12, dist21, _, _ = nndistance(gt_shape, output_shape)
                    cd = torch.mean(torch.mean(dist12, dim=-1) + torch.mean(dist21, dim=-1)).item()
                    hd = max(torch.max(dist12).item(), torch.max(dist21).item())

                eval_result[cur_dir]["CD"][target] = min(eval_result[cur_dir]["CD"][target], cd)
                avg_result[cur_dir]["CD"]["avg"] += (cd - avg_result[cur_dir]["CD"]["avg"])/(avg_result[cur_dir]["CD"]["cnt"]+1)
                avg_result[cur_dir]["CD"]["cnt"]+=1
                eval_result[cur_dir]["HD"][target] = min(eval_result[cur_dir]["HD"][target], hd)
                avg_result[cur_dir]["HD"]["avg"] += (hd - avg_result[cur_dir]["HD"]["avg"])/(avg_result[cur_dir]["HD"]["cnt"]+1)
                avg_result[cur_dir]["HD"]["cnt"]+=1

            # 3. evaluation others
            for cur_dir in result_dirs[1:]:
                result_path = glob(os.path.join(cur_dir, "{}.*".format(target)))
                if len(result_path) == 0:
                    logger.warn("Cound\'t find {}. Skip to process the next.".format(result_path))
                    continue
                result_path = result_path[0]
                output_shape, output_face = read_trimesh(result_path, clean=False)
                result_name = os.path.splitext(os.path.basename(result_path))[0]
                if resample:
                    output_pts_file = os.path.join(cur_dir, "eval_pts", result_name+".pts")
                    if not os.path.isfile(output_pts_file):
                        logger.warn("Cound\'t find {}. Skip to process the next source.".format(output_pts_file))
                        continue
                    output_pts = load(output_pts_file)
                    output_pts = torch.from_numpy(output_pts[:,:3].astype(np.float32)).unsqueeze(0).cuda()
                    # compute chamfer
                    dist12, dist21, _, _ = nndistance(gt_pts, output_pts)
                    cd = torch.mean(torch.mean(dist12, dim=-1) + torch.mean(dist21, dim=-1)).item()
                    hd = max(torch.max(dist12).item(), torch.max(dist21).item())
                else:
                    dist12, dist21, _, _ = nndistance(gt_shape, output_shape)
                    cd = torch.mean(torch.mean(dist12, dim=-1) + torch.mean(dist21, dim=-1)).item()
                    hd = max(torch.max(dist12).item(), torch.max(dist21).item())

                eval_result[cur_dir]["CD"][target] = min(eval_result[cur_dir]["CD"][target], cd)
                avg_result[cur_dir]["CD"]["avg"] += (cd - avg_result[cur_dir]["CD"]["avg"])/(avg_result[cur_dir]["CD"]["cnt"]+1)
                avg_result[cur_dir]["CD"]["cnt"]+=1
                eval_result[cur_dir]["HD"][target] = min(eval_result[cur_dir]["HD"][target], hd)
                avg_result[cur_dir]["HD"]["avg"] += (hd - avg_result[cur_dir]["HD"]["avg"])/(avg_result[cur_dir]["HD"]["cnt"]+1)
                avg_result[cur_dir]["HD"]["cnt"]+=1

            print("{}: {}".format(target.ljust(70), " | ".join(
                ["{:8.4g}/{:8.4g}".format(
                    eval_result[cur_dir]["CD"][target],
                    eval_result[cur_dir]["HD"][target],
                    )
                for cur_dir in result_dirs]
                ).ljust(30)))
        except Exception as e:
            traceback.print_exc(file=sys.stdout)
            logger.warn("Failed to evaluation {}. Skip to process the next.".format(target))


    print("{}: {}".format("AVG".ljust(70), " | ".join(
        ["{:8.4g}/{:8.4g}".format(
            avg_result[cur_dir]["CD"]["avg"],
            avg_result[cur_dir]["HD"]["avg"],
            )
            for cur_dir in result_dirs]
        ).ljust(30)))

    ########## write evaluation ############
    for cur_dir in result_dirs:
        for metric in eval_result[cur_dir]:
            output_file = os.path.join(cur_dir, "eval_{}.txt".format(metric))
            with open(output_file, "w") as eval_file:
                for name, value in eval_result[cur_dir][metric].items():
                    if (name != "avg" and name != "cnt"):
                        eval_file.write("{} {:8.4g}\n".format(name, value))

                eval_file.write("avg {:8.4g}".format(eval_result[cur_dir][metric]["avg"]))

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser("evaluation")
    parser.add_argument('--svr', action="store_true")
    parser.add_argument("--dirs", nargs="+", required=True)
    parser.add_argument("--overwrite_pts", action="store_true")
    parser.add_argument("--mse", action="store_true")
    opt = parser.parse_args()
    if opt.svr:
        evaluate_svr(opt.dirs, True, overwrite_pts=False)
    else:
        evaluate_deformation(opt.dirs, (not opt.mse), mse=opt.mse, overwrite_pts=opt.overwrite_pts)
