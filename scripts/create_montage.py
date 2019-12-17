import sys
import os
import shlex
import subprocess
from glob import glob
from multiprocessing.pool import ThreadPool
from pytorch_points.misc import logger

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR + "/..")

from common import find_files, call_proc

def create_montage(render_dir, output_cage):
    """render shapes inside a directory with thea"""
    files = find_files(render_dir, "png")
    output_dir = os.path.join(render_dir, "montage")
    os.makedirs(output_dir, exist_ok=True)
    source_files = [p for p in files if "Sa.png" in p]
    pool = ThreadPool(processes=4)
    results =[]
    for source in source_files:
        dir_path = os.path.dirname(source)
        sname, tname = os.path.basename(source).split("-")[:2]
        target = source.replace("Sa", "Sb")
        deformed = glob( os.path.join(dir_path, "{}-{}-Sab*.png".format(sname, tname)) )
        deformed = " ".join(deformed)
        if output_cage:
            cage1 = glob( os.path.join(dir_path, "{}-{}-cage1*.png".format(sname, tname)) )
            cage1 = " ".join(cage1)
            cage2 = glob( os.path.join(dir_path, "{}-{}-cage2*.png".format(sname, tname)) )
            cage2 = " ".join(cage2)
        else:
            cage1 = cage2 = ""
        output_file = os.path.join(output_dir, "{}-{}.png".format(sname, tname))
        # -gravity Center -crop 480x480+0+0 +repage
        results.append(pool.apply_async(call_proc,
                                        ("montage -geometry +2+0 -trim -tile x1 {} {} {} {} {} {}".format(source, target, deformed, cage1, cage2, output_file),)
                                        ))
    # Close the pool and wait for each running task to complete
    pool.close()
    pool.join()
    for result in results:
        out, err = result.get()
        if len(err) > 0:
            print("err: {}".format(err))

def create_comparison_montage(render_dirs, labels, output_dir, output_cage=False):
    files = find_files(render_dirs[0], "png")
    source_files = [p for p in files if "Sb.png" in p]
    logger.info("Found {} files".format(len(source_files)))
    pool = ThreadPool(processes=4)
    results =[]
    os.makedirs(output_dir, exist_ok=True)
    for source in source_files:
        sname, tname = os.path.basename(source).split("-")[:2]
        output_file = os.path.join(output_dir, "{}-{}.png".format(sname, tname))

        images = [glob( os.path.join(cur_dir, "{}-{}-Sab*.png".format(sname, tname)) ) for cur_dir in render_dirs]
        if not all([len(im_found)>0 for im_found in images]):
            indices = [i for i, x in enumerate(images) if len(x)==0]
            logger.warn("","Cannot find {} in {}".format("{}-{}-Sab*.png".format(sname, tname), ", ".join([render_dirs[i] for i in indices])))
            continue

        images = [im_found[0] for im_found in images]
        cages = []
        mylabels = labels[:]
        if output_cage:
            # find cages
            for i, dir_img in enumerate(zip(render_dirs, images)):
                cur_dir, image = dir_img
                cage1 = glob(image.replace("Sab", "cage1"))
                cage2 = glob(image.replace("Sab", "cage2"))
                if len(cage1) > 0 and len(cage2) > 0:
                    cages.append((i, cage1[0], cage2[0]))
            # insert cages to the correct position in images
            cnt = 0
            for offset, cage1, cage2 in cages:
                images.insert(offset+cnt+1, cage1)
                images.insert(offset+cnt+2, cage2)
                mylabels.insert(offset+cnt+1, mylabels[offset+cnt]+"_cage1")
                mylabels.insert(offset+cnt+2, mylabels[offset+cnt]+"_cage2")
                cnt += 2
            assert(len(images)==len(mylabels))
            image_strs = " ".join(["-label {} {}".format(l, i) for l, i in zip(mylabels, images)])
        else:
            image_strs = " ".join(["-label {} {}".format(l, i) for l, i in zip(mylabels, images)])

        num_cols = len(images) + 2
        target = source.replace("Sa", "Sb")
        results.append(pool.apply_async(call_proc,
                                        ("montage -geometry +0+0 -gravity Center -crop 420x450+0+0 +repage -tile {}x1 -label input {} -label target {} {} {}".format(num_cols, target, source, image_strs, output_file),)
                                        ))
    # Close the pool and wait for each running task to complete
    pool.close()
    pool.join()
    for result in results:
        out, err = result.get()
        if len(err) > 0:
            print("err: {}".format(err))

def create_two_row_comparison_montage(render_dirs, labels, output_dir, output_cage=True):
    files = find_files(render_dirs[0], "png")
    source_files = [p for p in files if "Sa.png" in p]
    logger.info("Found {} files".format(len(source_files)))
    pool = ThreadPool(processes=4)
    results =[]
    os.makedirs(output_dir, exist_ok=True)
    # first concatenate cage1-cage2
    for cur_dir in render_dirs:
        cage1s = glob(os.path.join(cur_dir, "*cage1*.png"))
        cage2s = [f.replace("cage1", "cage2") for f in cage1s]
        for cage1, cage2 in zip(cage1s, cage2s):
            if not (os.path.isfile(cage1) and os.path.isfile(cage2)):
                continue
            output_file = os.path.join(cage1.replace("cage1", "cages"))
            results.append(pool.apply_async(call_proc,
                                        ("montage -geometry +0+0 -gravity Center -crop 400x400+0+0 +repage -tile 2x1 {} {} {}".format(cage1, cage2, output_file),)
                                        ))
    pool.close()
    pool.join()
    for result in results:
        out, err = result.get()
        if len(err) > 0:
            print("err: {}".format(err))
    results.clear()
    pool = ThreadPool(processes=4)
    for source in source_files:
        sname, tname = os.path.basename(source).split("-")[:2]
        output_file = os.path.join(output_dir, "{}-{}.png".format(sname, tname))

        images = [glob( os.path.join(cur_dir, "{}-{}-Sab*.png".format(sname, tname)) ) for cur_dir in render_dirs]
        if not all([len(im_found)>0 for im_found in images]):
            indices = [i for i, x in enumerate(images) if len(x)==0]
            logger.warn("","Cannot find {} in {}".format("{}-{}-Sab*.png".format(sname, tname), ", ".join([render_dirs[i] for i in indices])))

        images = ["null:" if len(im_found)==0 else im_found[0] for im_found in images]

        cages = [glob( os.path.join(cur_dir, "{}-{}-cages*.png".format(sname, tname)) ) for cur_dir in render_dirs]
        if not all([len(im_found)>0 for im_found in cages]):
            indices = [i for i, x in enumerate(cages) if len(x)==0]
            logger.warn("","Cannot find {} in {}".format("{}-{}-cages*.png".format(sname, tname), ", ".join([render_dirs[i] for i in indices])))

        cages = ["null:" if len(im_found)==0 else im_found[0] for im_found in cages]

        mylabels = labels[:]

        assert(len(images)==len(mylabels))
        image_strs = " ".join(images)
        cage_str = " ".join(["-label {} {}".format(l, i) for l, i in zip(mylabels, cages)])

        num_cols = len(images) + 1
        target = source.replace("Sa", "Sb")
        results.append(pool.apply_async(call_proc,
                                        ("montage -geometry \'420x400>+0+0\' -tile {}x2 {} "
                                        "{} {} {} {}".format(num_cols, source, image_strs, target, cage_str, output_file),)
                                        ))
    # Close the pool and wait for each running task to complete
    pool.close()
    pool.join()
    for result in results:
        out, err = result.get()
        if len(err) > 0:
            print("err: {}".format(err))

    for cur_dir in render_dirs:
        call_proc("rm {}".format(os.path.join(cur_dir, "*.cages*.png")))


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser("create montage of renders for comparison")
    parser.add_argument("--dirs", nargs="+", required=True)
    parser.add_argument("--labels", nargs="*", help="labels for montage")
    parser.add_argument("--output_cage", action="store_true")
    parser.add_argument("--fancy", action="store_true")
    opt = parser.parse_args()
    if len(opt.dirs) == 1:
        create_montage(opt.dirs[0], opt.output_cage)
    else:
        if isinstance(opt.labels, str):
            opt.labels = [opt.labels]
        if opt.labels is not None:
            assert((len(opt.dirs)-1) == len(opt.labels))
        else:
            opt.labels = [str(i) for i in range(len(opt.dirs)-1)]

        if opt.fancy:
            create_two_row_comparison_montage(opt.dirs[:-1], opt.labels, opt.dirs[-1], True)
        else:
            create_comparison_montage(opt.dirs[:-1], opt.labels, opt.dirs[-1], opt.output_cage)
