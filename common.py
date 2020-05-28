import numpy as np
import torch
import os
import shlex
import subprocess
import pymesh
from multiprocessing.pool import ThreadPool
from pytorch_points.utils.geometry_utils import read_trimesh, write_trimesh
from pytorch_points.utils.pc_utils import save_ply, save_ply_property
from pytorch_points.network.geo_operations import mean_value_coordinates_3D, green_coordinates_3D, compute_face_normals_and_areas
from glob import glob

def is_type(file, file_ext):
    if isinstance(file_ext, str):
        file_ext = [file_ext]
    tmp = [os.path.splitext(file)[-1].lower()[1:] == ext for ext in file_ext]
    return any(tmp)

def find_files(source, file_ext=["txt",]):
    # If file_ext is a list
    if source is None:
        return []
    # Seamlessy pc_utils.load single file, list of files and files from directories.
    source_fns = []
    if isinstance(source, str):
        if os.path.isdir(source) or source[-1] == '*':
            if isinstance(file_ext, list):
                for fmt in file_ext:
                    source_fns += find_files(source, fmt)
            else:
                source_fns = sorted(glob("{}/**/*.{}".format(source, file_ext),recursive=True))
        elif os.path.isfile(source):
            source_fns = [source]
        assert (all([is_type(f, file_ext) for f in source_fns])), "Given files contain files with unsupported format"
    elif len(source) and isinstance(source[0], str):
        for s in source:
            source_fns.extend(find_files(s, file_ext=file_ext))
    return source_fns

def loadInitCage(templates):
    init_cage_Fs = []
    for i, template in enumerate(templates):
        init_cage_V, init_cage_F = read_trimesh(template)
        init_cage_V = torch.from_numpy(init_cage_V[:,:3].astype(np.float32)).unsqueeze(0).cuda()*2.0
        init_cage_F = torch.from_numpy(init_cage_F[:,:3].astype(np.int64)).unsqueeze(0).cuda()
        init_cage_Fs.append(init_cage_F)
    return init_cage_V, init_cage_Fs



def renderMeshes(shape_dir, forward=(0.5,0.5,0), pos=(-4,-4,0), up=(0,0,1), color=None, suffix="", img_size=(480, 480), other_method=False, otherStr=""):
    """render shapes inside a directory with thea"""
    # mycolor = "e0f2d79b"
    mycolor = "c2d2e9"
    try:
        len(img_size)
    except Exception as e:
        img_size = [img_size]*2
    finally:
        assert(len(img_size)==2)
    thea_render_bin = "RenderShape"
    output_dir = os.path.join(shape_dir, "renders")
    os.makedirs(output_dir, exist_ok=True)
    files = find_files(shape_dir, ["ply", "obj", "pts"])
    view_opt = ",".join([str(_) for _ in forward])+","+",".join([str(_) for _ in pos])+","+",".join([str(_) for _ in up])
    cage_view_opt = ",".join([str(_) for _ in forward])+","+",".join([str(_*1.8) for _ in pos])+","+",".join([str(_) for _ in up])

    pool = ThreadPool(processes=4)
    results =[]
    for input_file in files:
        myotherStr = otherStr
        output_file = os.path.join(output_dir, os.path.splitext(os.path.basename(input_file))[0]+".png")
        # ./MeshSample -n2048 -l LABEL INPUT OUTPUT
        if "Sa" in input_file and ("Sab" not in input_file):
            fname = os.path.basename(input_file).split('-')[0]
            mycolor = color or "f7d6bf"
        elif "Sab" in input_file:
            fname = os.path.basename(input_file).split('-')[0]
            mycolor = color or "c2d2e9"
            if not other_method:
                myotherStr = myotherStr + " -b f6f7e4 "
        elif "Sb" in input_file:
            fname = os.path.basename(input_file).split('-')[1]
            mycolor = color or "b0cac7"
        elif "cage1" in input_file:
            fname = os.path.splitext(os.path.basename(input_file))[0]
            overlay_file = input_file.replace(fname[fname.find("cage1"):], "Sa")
            overlay_file = glob(os.path.splitext(overlay_file)[0]+".*")
            mycolor = color or "c2d2e9"
            if len(overlay_file) == 1:
                overlay_file = overlay_file[0]
                results.append(pool.apply_async(call_proc, (thea_render_bin + " {} -0 -c {} -v {} -j 666660 -o {} -i 0 {} {} {} {}".format(
                    myotherStr, mycolor, cage_view_opt, input_file, overlay_file, output_file, img_size[0], img_size[1]),)))
            continue
        elif "cage2" in input_file:
            overlay_file = input_file.replace("cage2", "Sab")
            overlay_file = glob(os.path.splitext(overlay_file)[0]+".*")
            mycolor = color or "c2d2e9"
            if len(overlay_file) == 1:
                overlay_file = overlay_file[0]
                results.append(pool.apply_async(call_proc, (thea_render_bin + " {} -0 -c {} -v {} -j 666660 -o {} -i 0 {} {} {} {}".format(
                    myotherStr, mycolor, cage_view_opt, input_file, overlay_file, output_file, img_size[0], img_size[1]),)))
            continue

        if input_file[-4:] == ".pts":
            oname, oext = os.path.splitext(output_file)
            results.append(pool.apply_async(call_proc, (thea_render_bin + " {} -0 -p 4 -c {} -v {} {} {} {} {}".format(
                myotherStr, mycolor, view_opt, input_file, oname+"_pts"+oext, img_size[0], img_size[1]),)))
        else:
            results.append(pool.apply_async(call_proc, (thea_render_bin + " {} -0 -c {} -v {} {} {} {} {}".format(
                myotherStr, mycolor, view_opt, input_file, output_file, img_size[0], img_size[1]),)))

    # Close the pool and wait for each running task to complete
    pool.close()
    pool.join()
    for result in results:
        out, err = result.get()
        if len(err) > 0:
            print("err: {}".format(err))


def call_proc(cmd):
    """ This runs in a separate thread. """
    #subprocess.call(shlex.split(cmd))  # This will block until cmd finishes
    p = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    return (out, err)

def deform_with_MVC(cage, cage_deformed, cage_face, query, verbose=False):
    """
    cage (B,C,3)
    cage_deformed (B,C,3)
    cage_face (B,F,3) int64
    query (B,Q,3)
    """
    weights, weights_unnormed = mean_value_coordinates_3D(query, cage, cage_face, verbose=True)
    weights = weights.detach()
    deformed = torch.sum(weights.unsqueeze(-1)*cage_deformed.unsqueeze(1), dim=2)
    if verbose:
        return deformed, weights, weights_unnormed
    return deformed

# def deform_with_GC(cage, cage_deformed, cage_face, query, verbose=False):
#     cage_FN, _ = compute_face_normals_and_areas(cage, cage_face)
#     coords_V, coords_F, is_exterior = green_coordinates_3D(query, cage, cage_face, face_normals=cage_FN, verbose=True)
#     # (B,P,N)*(B,N,3)->(B,N,3)
#     cage_deformed_FN, _ = compute_face_normals_and_areas(cage_deformed, cage_face)
#     deformed = torch.sum(coords_V.unsqueeze(-1)*cage_deformed.unsqueeze(1), dim=-2) + torch.sum(coords_F.unsqueeze(-1)*cage_deformed_FN.unsqueeze(1), dim=-2)
#     # if verbose:
#     #     return deformed, weights, is_exterior
#     return deformed

def load_shapenet_cat():
    """return two dictionaries: name-to-id and id-to-name"""
    namecat2numbercat = {}
    numbercat2namecat = {}
    with open(os.path.join("data","synsetoffset2category.txt"), 'r') as f:
        for line in f:
            ls = line.strip().split()
            namecat2numbercat[ls[0]] = ls[1]
            numbercat2namecat[ls[1]] = ls[1]
    return (namecat2numbercat, numbercat2namecat)

def build_dataset(opt):
    import datasets
    if opt.target_model is not None and opt.source_model is not None:
        normalization_fn = None
        if opt.dataset == "COSEG":
            normalization_fn = datasets.CoSegDataset.normalize
            render_fn = datasets.CoSegDataset.render_result
        elif opt.dataset == "FAUST":
            normalization_fn = datasets.FaustDataset.normalize
            render_fn = datasets.FaustDataset.render_result
        elif opt.dataset == "SHAPENET":
            normalization_fn = datasets.ShapeNetSeg.normalize
            render_fn = datasets.ShapeNetSeg.render_result
        elif opt.dataset == "SHAPENETV2":
            normalization_fn = datasets.ShapeNetV2.normalize
            render_fn = datasets.ShapeNetV2.render_result
        dataset = datasets.FileListDataset(opt, normalization_fn)
        dataset.render_result = render_fn

    elif opt.dataset == "COSEG":
        dataset = datasets.CoSegDataset(root_dir=opt.data_dir, cat=opt.data_cat, phase=opt.phase, max=(opt.data_max if opt.phase=="test" else -1))
        data_dict = dataset[0]
        opt.sym_plane = dataset.sym_plane
    elif opt.dataset == "FAUST":
        dataset = datasets.FaustDataset(root_dir=opt.data_dir, phase=opt.phase, npoints=opt.num_point,
                                        template=opt.template,
                                        source=opt.source_model,
                                        max=opt.data_max, regular_sampling=opt.regular_sampling,
                                        normalization=False)
        opt.sym_plane = None
    elif opt.dataset == "SHAPENET":
        dataset = datasets.ShapeNetSeg(root_dir=opt.data_dir,
                                       phase=opt.phase,
                                       shuffle=(opt.phase=="train"),
                                       knn=False,
                                       num_neighbors=40,
                                       normalization="BoundingBox",
                                       class_choice=opt.data_cat,
                                       data_augmentation_Z_rotation=False,
                                       data_augmentation_Z_rotation_range=40,
                                       anisotropic_scaling=(opt.phase=="train"),
                                       npoints=opt.num_point,
                                       random_translation=False,
                                       use_fixed_pairs=(opt.phase=="test"),
                                       num_samples=(opt.data_max if opt.phase=="test" else -1),
                                       isV2=opt.isV2,
                                       use_preprocessed=opt.use_preprocessed)
        opt.sym_plane = dataset.sym_plane
    elif opt.dataset == "SHAPENETV2":
        dataset = datasets.ShapeNetV2(phase=opt.phase,
                                       shuffle=(opt.phase=="train"),
                                       knn=False,
                                       num_neighbors=40,
                                       normalization="BoundingBox",
                                       class_choice=opt.data_cat,
                                       data_augmentation_Z_rotation=False,
                                       data_augmentation_Z_rotation_range=40,
                                       anisotropic_scaling=(opt.phase=="train"),
                                       npoints=opt.num_point,
                                       random_translation=False,
                                       use_fixed_pairs=(opt.phase=="test"),
                                       num_samples=(opt.data_max if opt.phase=="test" else -1),
                                       use_preprocessed=opt.use_preprocessed)
        opt.sym_plane = dataset.sym_plane
    elif opt.dataset == "SURREAL":
        dataset = datasets.PairedSurreal(root_dir=opt.data_dir,
                                         template=opt.template,
                                         source=opt.source_model,
                                         max=(opt.data_max if opt.phase=="test" else -1),
                                         regular_sampling=opt.regular_sampling,
                                         phase=opt.phase,
                                         data_augmentation_Z_rotation=False,
                                         data_augmentation_Z_rotation_range=40,
                                         npoints=opt.num_point,
                                        )
        opt.sym_plane = dataset.sym_plane
    elif opt.dataset == "MNIST_MIXED":
        dataset = datasets.CrossCategoryPairsDataset(opt.data_dir, phase=opt.phase, num_point=opt.num_point, training_size=0.9,
                                            source_digits=opt.source_digit, target_digits=opt.target_digit, max=(100 if opt.phase=="test" else -1))
        opt.sym_plane = None
    elif opt.dataset == "MNIST_SINGLE":
        dataset = datasets.SameCategoryPairsDataset(opt.data_dir, phase=opt.phase, num_point=opt.num_point, training_size=0.9,
                                                    categories=opt.source_digit, max=(100 if opt.phase=="test" else -1))
        opt.sym_plane = None
    else:
        raise ValueError("Unsupported dataset")

    opt.mesh_data = dataset.mesh_data
    return dataset

def log_outputs(opt, step, all_outputs, all_inputs):
    # Source
    color = all_inputs["source_shape"][:,:,1].cpu().numpy()
    save_ply_property(os.path.join(opt.log_dir,"step-{:06d}-Sa.ply".format(step)), all_inputs["source_shape"][0].detach().cpu().numpy(), color[0], cmap_name="rainbow")
    # Target
    save_ply_property(os.path.join(opt.log_dir,"step-{:06d}-Sb.ply".format(step)), all_inputs["target_shape"][0].detach().cpu().numpy(), color[0], cmap_name="rainbow")
    for batch in range(0, all_outputs["cage"].shape[0], opt.batch_size):
        if batch // opt.batch_size == 0:
            tag = "StoT"
        elif batch // opt.batch_size == 2:
            tag = "StoS"
        elif batch // opt.batch_size == 1:
            tag = "TtoS"
        elif batch // opt.batch_size == 3:
            tag = "TtoT"

        # deformed and cage
        save_ply_property(os.path.join(opt.log_dir,"step-{:06d}-{}-Sab.ply".format(step, tag)),
                    all_outputs["deformed"][batch].detach().cpu().numpy(), color[batch], cmap_name="rainbow")
        write_trimesh(os.path.join(opt.log_dir, "step-{:06d}-{}-cage1.ply".format(step, tag)),
                    all_outputs["cage"][batch].detach().cpu(), all_outputs["cage_face"][0].detach().cpu(), binary=True)
        write_trimesh(os.path.join(opt.log_dir, "step-{:06d}-{}-cage2.ply".format(step, tag)),
                    all_outputs["new_cage"][batch].detach().cpu(), all_outputs["cage_face"][0].detach().cpu(), binary=True)

        # if using network2
        if "cage_surface" in all_outputs:
            save_ply(os.path.join(opt.log_dir,"step-{:06d}-{}-cage_surface1.ply".format(step, tag)), all_outputs["cage_surface"][batch].detach().cpu().numpy())
            save_ply(os.path.join(opt.log_dir,"step-{:06d}-{}-cage_surface2.ply".format(step, tag)), all_outputs["new_cage_surface"][batch].detach().cpu().numpy())

def remesh(path1):
    """
    This function takes a path to the orginal shapenet model and subsample it nicely
    """
    obj1 = pymesh.load_mesh(path1)
    obj1, info = pymesh.remove_isolated_vertices(obj1)
    print("Removed {} isolated vertices".format(info["num_vertex_removed"]))
    obj1, info = pymesh.remove_duplicated_vertices(obj1)
    print("Merged {} duplicated vertices".format(info["num_vertex_merged"]))
    obj1, _ = pymesh.remove_degenerated_triangles(obj1)
    if len(obj1.vertices)<5000:
        while len(obj1.vertices)<5000:
            obj1 = pymesh.subdivide(obj1)
    obj1 = pymesh.form_mesh(obj1.vertices, obj1.faces)
    return obj1

def read_trimesh(path, normal=False, clean=True):
    mesh = pymesh.load_mesh(path)
    if clean:
        mesh, info = pymesh.remove_isolated_vertices(mesh)
        print("Removed {} isolated vertices".format(info["num_vertex_removed"]))
        mesh, info = pymesh.remove_duplicated_vertices(mesh)
        print("Merged {} duplicated vertices".format(info["num_vertex_merged"]))
        mesh, info = pymesh.remove_degenerated_triangles(mesh)
        mesh = pymesh.form_mesh(mesh.vertices, mesh.faces)

    vertices = mesh.vertices
    if normal:
        mesh.add_attribute("vertex_normal")
        vertex_normals = mesh.get_attribute("vertex_normal").reshape(-1, 3)
        vertices = np.concatenate([vertices, vertex_normals], axis=-1)
    return vertices, mesh.faces

def crisscross_input(data):
    _source_shape, _source_normals, _source_face, _source_filename, \
    _target_shape, _target_normals, _target_face, _target_filename = \
        data["source_shape"], data["source_normals"], data["source_face"], data["source_file"], \
        data["target_shape"], data["target_normals"], data["target_face"], data["target_file"] \

    data["source_shape"] = torch.cat([_source_shape, _target_shape, _source_shape], dim=0).contiguous()
    data["target_shape"] = torch.cat([_target_shape, _source_shape, _source_shape], dim=0).contiguous()
    data["source_normals"] = torch.cat([_source_normals, _target_normals, _source_normals], dim=0).contiguous()
    data["target_normals"] = torch.cat([_target_normals, _source_normals, _source_normals], dim=0).contiguous()
    data["source_file"] = [_source_filename] + [_target_filename] + [_source_filename]
    data["target_file"] = [_target_filename] + [_source_filename] + [_source_filename]

    if _source_face is not None and _target_face is not None:
        data["source_face"] = torch.cat([_source_face, _target_face,  _source_face], dim=0).contiguous()
        data["target_face"] = torch.cat([_target_face, _source_face,  _source_face], dim=0).contiguous()

    if "source_label" in data and data["source_label"] is not None and "source_label" in data and data["target_label"] is not None:
        _source_label = data["source_label"]
        _target_label = data["target_label"]
        data["source_label"] = torch.cat([_source_label, _target_label, _source_label], dim=0).contiguous()
        data["target_label"] = torch.cat([_target_label, _source_label, _source_label], dim=0).contiguous()


    return data
