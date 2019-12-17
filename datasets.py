import torch
import os
from glob import glob
import itertools
from collections import OrderedDict
import numpy as np
import json
import openmesh as om
import time
from collections import defaultdict
import pickle
# from sklearn.neighbors import NearestNeighbors
import joblib
import pymesh
from multiprocessing.pool import ThreadPool
import pytorch_points.utils.pc_utils as pc_utils
from pytorch_points.network.geo_operations import normalize_point_batch_to_sphere, compute_face_normals_and_areas, furthest_point_sample
from pytorch_points.network.operations import scatter_add
from pytorch_points.utils.pytorch_utils import check_values, AverageValueMeter
from pytorch_points.utils.geometry_utils import Mesh, get_edge_points
from pytorch_points.misc import logger
from losses import LabeledChamferDistance
from common import find_files, renderMeshes, call_proc, read_trimesh

labeled_chamfer_distance = LabeledChamferDistance(beta=1.0, gamma=1, delta=0)

def _numpy_chamfer(P1, P2):
    """
    used for sklearn NearestNeighbor P1, P2 (1, 3xP)
    """
    P1 = torch.from_numpy(P1).view(-1, 3).unsqueeze(0).cuda().float()
    P2 = torch.from_numpy(P2).view(-1, 3).unsqueeze(0).cuda().float()
    size_1 = P1.size(1)
    orig_size_tmp_1 = (P1.view(-1)[P1.view(-1) == 100]).nelement()
    orig_size_1 = orig_size_tmp_1 / 3
    assert orig_size_1 * 3 == orig_size_tmp_1
    orig_size_1 = size_1 - orig_size_1

    size_2 = P2.size(1)
    orig_size_tmp_2 = (P2.view(-1)[P2.view(-1) == 100]).nelement()
    orig_size_2 = orig_size_tmp_2 / 3
    assert orig_size_2 * 3 == orig_size_tmp_2
    orig_size_2 = size_2 - orig_size_2

    dist, _, _ = labeled_chamfer_distance(P1, P2)
    dist = dist.cpu().numpy()
    return dist

class FileListDataset(torch.utils.data.Dataset):
    def __init__(self, opt, normalization_function=lambda x: pc_utils.normalize_to_box(x)[0]):
        super().__init__()
        self.mesh_data = True
        self.source_files = opt.source_model
        self.target_files = opt.target_model
        pref = os.path.commonpath([os.path.dirname(f) for f in self.source_files])
        self.source_names = [os.path.relpath(f, pref) for f in self.source_files]

        pref = os.path.commonpath([os.path.dirname(f) for f in self.target_files])
        self.target_names = [os.path.relpath(f, pref) for f in self.target_files]
        assert(len(self.source_files) == len(self.target_files))
        self.opt = opt
        self.normalize = normalization_function or (lambda x: pc_utils.normalize_to_box(x)[0])

    def __len__(self):
        return len(self.target_files)

    def __getitem__(self,idx):
        source_file = self.source_files[idx]
        target_file = self.target_files[idx]
        # TODO handle point inputs
        if source_file[-4:] in (".pts"):
            source_mesh = pc_utils.load(source_file)
            source_face = None
        else:
            if self.opt.is_poly:
                source_mesh = om.read_polymesh(source_file)
                source_mesh = source_mesh.points().copy()
                source_face = None
            else:
                source_mesh, source_face = read_trimesh(source_file,clean=False)
                source_face = torch.from_numpy(source_face).to(torch.int64)

        source_mesh = torch.from_numpy(source_mesh[:,:3].astype(np.float32))
        source_mesh = self.normalize(source_mesh)

        if target_file[-4:] in (".pts"):
            target_mesh = pc_utils.load(target_file)
            target_face = None
        else:
            if self.opt.is_poly:
                target_mesh = om.read_polymesh(target_file)
                target_mesh = target_mesh.points().copy()
                target_face = None
            else:
                target_mesh, target_face = read_trimesh(target_file,clean=False)
                target_face = torch.from_numpy(target_face).to(torch.int64)

        target_mesh = torch.from_numpy(target_mesh[:,:3].astype(np.float32))
        target_mesh = self.normalize(target_mesh)
        # sample source point
        if source_mesh.shape[0] != self.opt.num_point:
            source_shape = furthest_point_sample(source_mesh.unsqueeze(0).cuda(), self.opt.num_point, NCHW=False)[1]
            source_shape = source_shape.squeeze(0)
        else:
            source_shape = source_mesh.clone()

        if target_mesh.shape[0] != self.opt.num_point:
            target_shape = furthest_point_sample(target_mesh.unsqueeze(0).cuda(), self.opt.num_point, NCHW=False)[1]
            target_shape = target_shape.squeeze(0)
        else:
            target_shape = target_mesh.clone()

        source_fn = os.path.splitext(self.source_names[idx])[0].replace("/","_")
        target_fn = os.path.splitext(self.target_names[idx])[0].replace("/","_")
        if self.opt.is_poly:
            source_mesh = source_file
            target_mesh = target_file
            return {"source_shape": source_shape, "source_mesh": source_mesh, "source_file": source_fn,
                    "target_shape": target_shape, "target_mesh": target_mesh, "target_file": target_fn,}

        return {"source_shape": source_shape, "source_face": source_face, "source_mesh": source_mesh, "source_file": source_fn,
                "target_shape": target_shape, "target_face": target_face, "target_mesh": target_mesh, "target_file": target_fn,}

    @staticmethod
    def uncollate(batch_data):
        source_shape = batch_data["source_shape"][:,:,:3].detach().cuda().contiguous()
        target_shape = batch_data["target_shape"][:,:,:3].detach().cuda().contiguous()
        if "source_mesh" not in batch_data:
            source_mesh = None
        elif isinstance(batch_data["source_mesh"], torch.Tensor):
            source_mesh = batch_data["source_mesh"].cuda().detach()
        else:
            source_mesh = batch_data["source_mesh"]

        if "target_mesh" not in batch_data:
            target_mesh = None
        elif isinstance(batch_data["target_mesh"], torch.Tensor):
            target_mesh = batch_data["target_mesh"].cuda().detach()
        else:
            target_mesh = batch_data["target_mesh"]

        source_face = batch_data["source_face"].cuda().detach() if "source_face" in batch_data else None
        target_face = batch_data["target_face"].cuda().detach() if "target_face" in batch_data else None

        return {
            "source_shape": source_shape, "source_file": batch_data["source_file"],
            "target_shape": target_shape, "target_file": batch_data["target_file"],
            "source_mesh": source_mesh, "source_face": source_face, "target_mesh": target_mesh, "target_face": target_face
        }

class FaustDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir="/home/mnt/points/data/MPI-FAUST", phase="train", npoints=6890, max=-1,
                 normalization=True, regular_sampling=False, pair_list=None,
                 template="./vanilla_data/surreal_template_v77.ply", source="./vanilla_data/surreal_template.ply",
                 **kwargs):
        super().__init__()
        self.mesh_data = True
        self.npoints = npoints
        self.regular_sampling = regular_sampling
        self.mesh_vertex, self.mesh_face = read_trimesh(source)
        self.mesh_vertex = self.mesh_vertex[:,:3]
        self.mesh_face = torch.from_numpy(self.mesh_face[:,:3].astype(np.int64))
        self.mesh_vertex = torch.from_numpy(self.mesh_vertex)
        template = template or "./vanilla_data/surreal_template_v77.ply"
        source = source or "./vanilla_data/surreal_template.ply"
        _, farea = compute_face_normals_and_areas(self.mesh_vertex.unsqueeze(0), self.mesh_face.unsqueeze(0))
        v_area = scatter_add(farea.view(-1,1).expand(-1,3).contiguous().view(-1), self.mesh_face.view(-1), 0, out_size=(self.mesh_vertex.shape[0],))
        self.prob = (v_area / torch.sum(v_area)).numpy()

        self.cage_vertex, self.cage_face = read_trimesh(template)
        self.cage_vertex = self.cage_vertex[:,:3]+0.05*self.cage_vertex[:,3:6]
        self.cage_face = torch.from_numpy(self.cage_face[:,:3].astype(np.int64))
        self.cage_vertex = torch.from_numpy(self.cage_vertex)

        if normalization:
            self.mesh_vertex, center, scale = pc_utils.normalize_to_box(self.mesh_vertex)
            self.cage_vertex = (self.cage_vertex - center)/scale
        else:
            self.mesh_vertex, center, scale = pc_utils.center_bounding_box(self.mesh_vertex)
            self.cage_vertex = (self.cage_vertex - center)

        if phase=="train":
            self.source_files = glob(os.path.join(root_dir, "training", "registrations", "*[!8].ply"))  # the 8th pose has global rotation
            self.target_files = self.source_files[:]
            np.random.shuffle(self.target_files)
        else:
            self.source_files, self.target_files = np.loadtxt(os.path.join(root_dir,"test", "challenge_pairs", "inter_challenge.txt"),
                unpack=True, dtype=str, delimiter="_")
            self.source_files = [os.path.join(root_dir, "test", "scans", "test_scan_{}.ply".format(f)) for f in self.source_files]
            self.target_files = [os.path.join(root_dir, "test", "scans", "test_scan_{}.ply".format(f)) for f in self.target_files]

        if max > 0:
            self.source_files = self.source_files[:max]
            self.target_files = self.target_files[:max]
        assert(len(self.source_files)>0 and len(self.source_files)>0)
        self.isTrain = phase == "train"
        self.normalization = normalization

    @staticmethod
    def normalize(shape):
        shape = pc_utils.normalize_to_box(shape)[0]
        return shape

    def __getitem__(self, idx):
        TARGET_IDX = idx
        INPUT_IDX = idx
        if self.isTrain:
            INPUT_IDX = np.random.randint(0, self.__len__())

        source_file = self.source_files[INPUT_IDX]
        target_file = self.target_files[TARGET_IDX]

        # source
        mesh_source = Mesh(source_file)
        # target
        mesh_target = Mesh(target_file)

        if self.normalization:
            V, _, _ = pc_utils.normalize_to_box(mesh_source.vs)
            mesh_source.vs = V
            V, _, _ = pc_utils.normalize_to_box(mesh_target.vs)
            mesh_target.vs = V
        else:
            V, _, _ = pc_utils.center_bounding_box(mesh_source.vs)
            mesh_source.vs = V
            V, _, _ = pc_utils.center_bounding_box(mesh_target.vs)
            mesh_target.vs = V

        # sample poitns
        if self.npoints != 6890:
            if self.regular_sampling:
                random_sample = np.random.choice(6890, size=self.npoints, p=self.prob)
            else:
                random_sample = np.random.choice(6890, size=self.npoints, replace=False)

            target_points = mesh_target.vs[random_sample]
            source_points = mesh_source.vs[random_sample]

            return (mesh_source, os.path.basename(source_file),
                    mesh_target, os.path.basename(target_file),
                    source_points,
                    target_points,
                    torch.from_numpy(random_sample.astype(np.int64)),
                    )


        return mesh_source, os.path.basename(source_file)[:-4], mesh_target, os.path.basename(target_file)[:-4]


    def __len__(self):
        return len(self.target_files)

    @staticmethod
    def uncollate(batch_data):
        source_mesh, source_filename, target_mesh, target_filename = batch_data[:4]
        source_shape = source_mesh['vs'].cuda().detach()
        source_normals = source_mesh['vn'].cuda().detach()
        target_face = target_mesh['fs'].cuda().detach()

        target_shape = target_mesh['vs'].cuda().detach()
        target_normals = target_mesh['vn'].cuda().detach()
        source_face = source_mesh['fs'].cuda().detach()

        return_d = {"source_shape": source_shape, "source_label": None, "source_file": source_filename, "source_normals": source_normals,
                    "target_shape": target_shape, "target_label": None, "target_file": target_filename, "target_normals": target_normals,
                    "source_mesh":  source_shape, "source_face":  source_face,
                    "target_mesh":  target_shape, "target_face": target_face }

        if len(batch_data) == 7:
            return_d["source_shape"] = batch_data[4].cuda().detach()
            return_d["target_shape"] = batch_data[5].cuda().detach()
            return_d["sample_idx"] = batch_data[6].cuda().detach()
            return return_d
        else:
            return return_d

    @staticmethod
    def render_result(shape_dir, **kwargs):
        renderMeshes(shape_dir, up=(0,1,0), forward=(0.2,0,-1), pos=(-1.2, 6), **kwargs)

class CoSegDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir="/home/mnt/points/data/Coseg_Wang/Coseg_Wang/", use_init_cage=False, cat="Vase300", phase="train", max=-1):
        super().__init__()
        self.mesh_data = False
        self.use_init_cage = use_init_cage
        self.root_dir = root_dir
        self.cat = cat
        self.label_dir = os.path.join(self.root_dir,"{}GT".format(self.cat))
        self.sym_plane = ("yz", "xz")
        if phase=="train":
            # self.training_source = glob(os.path.join(root_dir, "training", "registrations", "tr_reg_*.ply"))
            self.training_source = glob(os.path.join(root_dir, "{}Points".format(cat), "*.pts"))
            self.training_target = self.training_source[::-1]
            assert(len(self.training_source) > 0)
            assert(len(self.training_target) > 0)
            if max > 0:
                self.training_source = self.training_source[:max]
                self.training_target = self.training_target[:max]
            if use_init_cage:
                # self.cage_dir = os.path.join(root_dir, "training", "simplified_registrations")
                RuntimeWarning("Coseg dataset doesn't support initial cage")
                self.use_init_cage = False
            # self.training_target = self.training_source[:]
        else:
            test_list = os.path.join(root_dir, "{}Shapes".format(cat), "test.txt")
            self.testing_source, self.testing_target = np.loadtxt(os.path.join(root_dir, "{}Shapes".format(cat), "test.txt"), dtype=str, unpack=True)
            self.testing_source = self.testing_source[:max]
            self.testing_target = self.testing_target[:max]
            self.testing_dir = os.path.join(root_dir, "{}Shapes".format(cat))
            self.get_testing_source = lambda x: os.path.join(root_dir, "{}Points".format(cat), "{}.pts".format(self.testing_source[x]))
            self.get_testing_target = lambda x: os.path.join(root_dir, "{}Points".format(cat), "{}.pts".format(self.testing_target[x]))
            # self.get_source_mesh = lambda x: os.path.join(root_dir, "{}Shapes".format(cat), "{}.obj".format(self.testing_source[x]))
            # self.get_target_mesh = lambda x: os.path.join(root_dir, "{}Shapes".format(cat), "{}.obj".format(self.testing_target[x]))
            self.get_source_mesh = lambda x: os.path.join(root_dir, "{}Shapes".format(cat), "{}.off".format(self.testing_source[x]))
            self.get_target_mesh = lambda x: os.path.join(root_dir, "{}Shapes".format(cat), "{}.off".format(self.testing_target[x]))

        self.isTrain = phase == "train"
        self.get_default_label_path = lambda fname: os.path.join(self.label_dir,"{}.seg".format(fname))
        self.rot_mat = torch.from_numpy(pc_utils.get_3D_rot_matrix(0, -np.pi/2).astype(np.float32))

    @staticmethod
    def normalize(shape):
        shape = pc_utils.normalize_to_box(shape)[0]
        return shape

    def __getitem__(self, idx):
        if self.isTrain:
            TARGET_IDX = idx
            INPUT_IDX = np.random.randint(0, len(self.training_source))
            source_file = self.training_source[INPUT_IDX]
            target_file = self.training_target[TARGET_IDX]
        else:
            source_file = self.get_testing_source(idx)
            target_file = self.get_testing_target(idx)

        # source
        # point coordinates (Nx3), normals (Nx3) and labels (Nx1)
        try:
            pnl = np.loadtxt(source_file, dtype=np.float32, converters={6: lambda x: np.float32(x[1:-1])})
            V = pnl[:,:3]
            # seems that the normals are inverted
            V_n = -pnl[:,3:6]
            V_l = pnl[:,6:]
        except IndexError:
            pnl = np.loadtxt(source_file, dtype=np.float32)
            V = pnl[:,:3]
            V_n = pnl[:,3:6]
            V_l = np.zeros_like(pnl[:,:1])

        V, center, scale = pc_utils.normalize_to_box(V)
        V =  torch.from_numpy(V)
        V_n =  torch.from_numpy(V_n)
        V_l =  torch.from_numpy(V_l)

        # target
        try:
            pnl = np.loadtxt(target_file, dtype=np.float32, converters={6: lambda x: np.float32(x[1:-1])})
            V_target = pnl[:,:3]
            # seems that the normals are inverted
            V_n_target = -pnl[:,3:6]
            V_l_target = pnl[:,6:]
        except IndexError:
            pnl = np.loadtxt(target_file, dtype=np.float32)
            V_target = pnl[:,:3]
            V_n_target = pnl[:,3:6]
            V_l_target = np.zeros_like(pnl[:,:1])

        V_target, _, _ = pc_utils.normalize_to_box(V_target)
        V_target =  torch.from_numpy(V_target)
        V_n_target =  torch.from_numpy(V_n_target)
        V_l_target =  torch.from_numpy(V_l_target)

        # V = V.matmul(self.rot_mat)
        # V_n = V_n.matmul(self.rot_mat)
        # V_target = V_target.matmul(self.rot_mat)
        # V_n_target = V_n_target.matmul(self.rot_mat)

        V = torch.cat([V, V_n], dim=-1)
        V_target = torch.cat([V_target, V_n_target], dim=-1)
        # source_label_file = self.get_testing_source_label(idx)
        # target_label_file = self.get_testing_target_label(idx)

        if self.isTrain:
            return {"source_shape": V,        "source_label": V_l,        "source_file": os.path.basename(source_file)[:-4],
                    "target_shape": V_target, "target_label": V_l_target, "target_file": os.path.basename(target_file)[:-4]}

        V_mesh, F_mesh = read_trimesh(self.get_source_mesh(idx))
        V_mesh = V_mesh[:,:3]
        F_mesh = F_mesh[:,:3]
        # V_mesh = (V_mesh - center)/scale
        V_mesh, _, _ = pc_utils.normalize_to_box(V_mesh)

        V_mesh = torch.from_numpy(V_mesh).to(dtype=torch.float)
        F_mesh = torch.from_numpy(F_mesh).to(dtype=torch.int64)
        assert(F_mesh.max() < V_mesh.shape[0])

        V_mesh_target, F_mesh_target = read_trimesh(self.get_target_mesh(idx))
        V_mesh_target = V_mesh_target[:,:3]
        F_mesh_target = F_mesh_target[:,:3]
        # V_mesh_target = (V_mesh_target - center)/scale
        V_mesh_target, _, _ = pc_utils.normalize_to_box(V_mesh_target)

        V_mesh_target = torch.from_numpy(V_mesh_target).to(dtype=torch.float)
        F_mesh_target = torch.from_numpy(F_mesh_target).to(dtype=torch.int64)
        assert(F_mesh_target.max() < V_mesh_target.shape[0])

        # V_mesh_target = V_mesh_target.matmul(self.rot_mat)
        # V_mesh = V_mesh.matmul(self.rot_mat)

        return {"source_shape": V,        "source_label": V_l,        "source_file": os.path.basename(source_file)[:-4],
                "target_shape": V_target, "target_label": V_l_target, "target_file": os.path.basename(target_file)[:-4],
                "source_mesh":  V_mesh,   "source_face":  F_mesh,
                "target_mesh":  V_mesh_target, "target_face": F_mesh_target }

    @staticmethod
    def uncollate(batch_data):
        source_normals = batch_data["source_shape"][:,:,3:6].detach().cuda().contiguous()
        target_normals = batch_data["target_shape"][:,:,3:6].detach().cuda().contiguous()
        source_shape = batch_data["source_shape"][:,:,:3].detach().cuda().contiguous()
        target_shape = batch_data["target_shape"][:,:,:3].detach().cuda().contiguous()
        source_mesh = batch_data["source_mesh"].cuda().detach() if "source_mesh" in batch_data else None
        source_face = batch_data["source_face"].cuda().detach() if "source_face" in batch_data else None
        target_mesh = batch_data["target_mesh"].cuda().detach() if "target_mesh" in batch_data else None
        target_face = batch_data["target_face"].cuda().detach() if "target_face" in batch_data else None

        return {
            "source_shape": source_shape, "source_normals": source_normals, "source_label": batch_data["source_label"].detach().cuda(), "source_file": batch_data["source_file"],
            "target_shape": target_shape, "target_normals": target_normals, "target_label": batch_data["target_label"].detach().cuda(), "target_file": batch_data["target_file"],
            "source_mesh": source_mesh, "source_face": source_face, "target_mesh": target_mesh, "target_face": target_face
        }

    def __len__(self):
        if self.isTrain:
            return len(self.training_source)
        else:
            return len(self.testing_source)

    @staticmethod
    def render_result(shape_dir, **kwargs):
        """render shapes inside a directory with thea"""
        forward = (-.5,-.5,0)
        pos = (5,5,0)
        up = (0,0,1)
        renderMeshes(shape_dir, forward=forward, pos=pos, up=up, **kwargs)

def _unwrap_self(arg, **kwarg):
    return arg[0]._getitem(*(arg[1:]), **kwarg)

class ShapeNetSeg(torch.utils.data.Dataset):
    def __init__(self, root_dir="/home/mnt/points/data/ShapeNet/PartSeg_v0/shapenetcore_partanno_segmentation_benchmark_v0_normal",
                 mesh_dir="/home/mnt/points/data/ShapeNet/ShapeNetCore.v1.remesh",
                 num_samples=-1, normalization="BoundingBox", knn=False,
                 num_neighbors=20,
                 class_choice=None, phase="train", normal=False, npoints=2500, sample=True,
                 data_augmentation_Z_rotation=False, data_augmentation_Z_rotation_range=360,
                 data_augmentation_3D_rotation=False, random_translation=False, anisotropic_scaling=False,
                 shuffle=False, use_fixed_pairs=False, use_preprocessed=True, isV2=False, **kwargs):
        self.mesh_data = False
        self.use_init_cage = False
        self.use_preprocessed = use_preprocessed
        self.isV2 = isV2
        self.v2_rot_mat = torch.from_numpy(pc_utils.get_3D_rot_matrix(1, np.pi/2).astype(np.float32))
        self.dataset_string_args = str(phase) + "_" + \
                                   str(class_choice) + \
                                   "_" + str(num_samples) + \
                                   "_" + str(normalization) + \
                                   "_" + str(knn) + \
                                   "_" + str(num_neighbors) + \
                                   "_" + str(shuffle) + \
                                   "_" + str(normal) + \
                                   "_" + str(2500)
        self.mesh_dir = mesh_dir
        self.path_dataset = os.path.join("./data/processed_shapenetseg", self.dataset_string_args)
        os.makedirs(os.path.dirname(self.path_dataset), exist_ok=True)
        self.shuffle = shuffle
        self.num_samples = num_samples
        self.anisotropic_scaling = anisotropic_scaling
        self.fine_tune = (phase == 'fine_tune_test')
        if phase == 'fine_tune_test':
            phase = 'ALLDATA'
        self.knn = knn
        self.num_neighbors = num_neighbors
        self.normalization = normalization
        self.random_translation = random_translation
        self.data_augmentation_Z_rotation = data_augmentation_Z_rotation
        self.data_augmentation_Z_rotation_range = data_augmentation_Z_rotation_range  # range in degree of random rotation
        self.data_augmentation_3D_rotation = data_augmentation_3D_rotation
        self.sym_plane = ("xy", )
        self.npoints = npoints
        self.sample = sample
        self.normal = normal
        self.phase = phase
        self.root = root_dir
        self.datapath = []  # List to store all path of dataset files
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.namecat2numbercat = {}
        self.numbercat2namecat = {}
        self.numsamples_by_cat = {}
        self.class_choice = class_choice
        self.meta = {}
        self.data = []
        self.use_fixed_pairs = use_fixed_pairs
        # if self.shuffle and self.use_fixed_pairs:
        #     logger.warn("{}: shuffle and use_fixed_pairs is exclusive, shuffle to False".format(__name__))
        # if not os.path.exists(self.root):
        #     print("Downloading Shapetnet for segmentation...")
        #     os.system('chmod +x ./data/download_dataset_shapenet.sh')
        #     os.system('./data/download_dataset_shapenet.sh')
        # ----------------------------------------------------------#
        ## Create dictionaries with keys : name of class, value : name of folder and vice-versa
        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.numsamples_by_cat[ls[0]] = 0
                self.namecat2numbercat[ls[0]] = ls[1]
                self.numbercat2namecat[ls[1]] = ls[0]
                self.meta[ls[0]] = []  # List to store all path of files per category

        # ----------------------------------------------------------#
        # add all paths to the same list, keep track of sizes
        for cat in self.meta.keys():
            for file in self.meta[cat]:
                self.datapath.append(file)

        # fixed pairs? create or load a dict of source_target_pairs storing the cat/file_name: cat/file_name
        if self.use_fixed_pairs and self.phase in ("val", "test", "svr_test"):
            if os.path.exists(self.path_dataset + "_pairs.txt"):
                source_target_file = np.loadtxt(self.path_dataset + "_pairs.txt", dtype=str)
                self.source_target_pairs = source_target_file
                self.datas = [(None, self.numbercat2namecat[source.split("/")[0]], os.path.join(self.root, source)) for source, _ in self.source_target_pairs]
                self.datas += [(None, self.numbercat2namecat[source.split("/")[0]], os.path.join(self.root, source)) for _, source in self.source_target_pairs]
                self.datapath = self.source_target_pairs

        len(self)
        if not use_fixed_pairs or len(self.source_target_pairs)==0:
            if self.phase == "train":
                with open(os.path.join(os.path.join(self.root, "train_test_split"), 'shuffled_train_file_list.json')) as f:
                    file_list = json.load(f)
            if self.phase == "val":
                with open(os.path.join(os.path.join(self.root, "train_test_split"), 'shuffled_val_file_list.json')) as f:
                    file_list = json.load(f)
            if "test" in self.phase :
                with open(os.path.join(os.path.join(self.root, "train_test_split"), 'shuffled_test_file_list.json')) as f:
                    file_list = json.load(f)

            for file in file_list:
                # Typical example : shape_data/03001627/355fa0f35b61fdd7aa74a6b5ee13e775 so remove 'shape_data/' and add '.txt'
                file_path = os.path.join(self.root, file[11:]) + ".txt"
                number_category = file[11:19] + ""
                if (class_choice is None) or (self.numbercat2namecat[number_category] in class_choice):
                    self.meta[self.numbercat2namecat[number_category]].append(
                        (file_path, self.numbercat2namecat[number_category]))
                    self.numsamples_by_cat[self.numbercat2namecat[number_category]] = self.numsamples_by_cat[
                                                                                          self.numbercat2namecat[
                                                                                              number_category]] + 1
            # ----------------------------------------------------------#
            # add all paths to the same list, keep track of sizes
            for cat in self.meta.keys():
                for file in self.meta[cat]:
                    self.datapath.append(file)

            if self.use_preprocessed:
                self.preprocess()
            else:
                self.datas = [(None, file[1], file[0]) for file in self.datapath]

            if self.knn:
                start = time.time()
                logger.info(None,
                    "Computing nearest neighbors graph... (can take some time if it's not already precomputed.)")
                self.compute_nearest_neighbors_graph()
                logger.success('Done!')
                end = time.time()
                logger.info("Ellapsed time : " + '"%.2f' % (end - start))

            else:
                self.source_target_pairs = []
                for i in range(len(self.datapath)):
                    _, cat, filename = self.datas[i]
                    filename = os.path.basename(filename)
                    if self.knn:
                        j = self.indices[i][np.random.randint(self.num_neighbors)]
                    else:
                        j = np.random.randint(len(self.datas))
                    _, cat2, filename2 = self.datas[j]
                    filename2 = os.path.basename(filename2)
                    self.source_target_pairs = (self.namecat2numbercat[cat]+"/"+filename, self.namecat2numbercat[cat2]+"/"+filename2)
                with open(self.path_dataset + "_pairs.txt","w") as f:
                    for key, value in self.source_target_pairs:
                        f.write("{} {}\n".format(key, value))

        # Shuffle_list
        self.shuffle_list = [i for i in range(len(self))]
        if self.shuffle:
            np.random.shuffle(self.shuffle_list)

    @staticmethod
    def _uniformize_sizes(knn):
        """
        input : knn :
        -- list of tensors of size(npoints x 3)
        N_point can be different for each sample
        It is used to compute the KNN in the dataloader.
        Since all pointcloud have a point (100, 100, 100) it doesn't change their chamfer distance and doesn't affect the nearest neighbors.
        :param knn: list of tensors of size(npoints x 3)
        :return: Return a tensor of size B, max_npoints, 3. The missing values are filled with 100.
        """

        sizes = [data.size(0) for data in knn]
        scale = [data.max() for data in knn]
        scale = max(scale)
        max_sizes = max(sizes) + 1
        knn_new_list = []
        for data in knn:
            data = data.squeeze()
            new_tensor = torch.zeros((max_sizes, 3)) + 100
            new_tensor[:data.size(0)] = data[:, :3]
            knn_new_list.append(new_tensor.unsqueeze(0).float())

        return torch.cat(knn_new_list, 0)

    def compute_nearest_neighbors_graph(self):
        if not os.path.exists(self.path_dataset + "_knn_indices.npy"):
            knn = [data[0] for data in self.datas]
            knn = self._uniformize_sizes(knn)
            self.num_neighbors = min(self.num_neighbors, len(self.datas))
            nbrs = NearestNeighbors(n_neighbors=self.num_neighbors, algorithm='ball_tree', metric=_numpy_chamfer,
                                    n_jobs=4).fit(knn.view(knn.size(0), -1).numpy())
            distances, indices = nbrs.kneighbors(knn.view(knn.size(0), -1).numpy())
            self.indices = indices
            self.distances = distances
            np.save(self.path_dataset + "_knn_indices.npy", self.indices)
            np.save(self.path_dataset + "_knn_distances.npy", self.distances)
        else:
            self.indices = np.load(self.path_dataset + "_knn_indices.npy")
            self.distances = np.load(self.path_dataset + "_knn_distances.npy")

    def _getitem(self, index):
        """return points (N,3), category name string, path string"""
        file = self.datapath[index]
        try:
            points = pc_utils.load(file[0])
        except Exception as e:
            logger.warn("Could not read {}".format(file[0]))
            return None
        points = torch.from_numpy(points).float()
        # Normalization is done before resampling !
        points[:, :3] = self.normalize(points[:, :3], self.isV2)
        return points, file[1], file[0]

    def preprocess(self):
        start = time.time()
        if os.path.exists(self.path_dataset + ".pkl"):
            print("Reload dataset : ", self.path_dataset)
            with open(self.path_dataset + ".pkl", "rb") as fp:
                self.datas = pickle.load(fp)
        else:
            logger.info("preprocess dataset...")

            class BatchCompletionCallBack(object):
                completed = defaultdict(int)

                def __init__(se, time, index, parallel):
                    se.index = index
                    se.parallel = parallel

                def __call__(se, index):
                    BatchCompletionCallBack.completed[se.parallel] += 1
                    if BatchCompletionCallBack.completed[se.parallel] % 100 == 0:
                        end = time.time()
                        etl = (end - start) * (
                                len(self.datapath) / float(BatchCompletionCallBack.completed[se.parallel])) - (
                                      end - start)
                        print('\r' + "Progress : %f %% " %
                              float(BatchCompletionCallBack.completed[
                                        se.parallel] * 100 / len(self.datapath)) + "ETL %d seconds" % int(etl), end='')
                    if se.parallel._original_iterator is not None:
                        se.parallel.dispatch_next()

            joblib.parallel.BatchCompletionCallBack = BatchCompletionCallBack
            self.datas = joblib.Parallel(n_jobs=-1, backend="multiprocessing")(
                joblib.delayed(_unwrap_self)(i) for i in zip([self] * len(self.datapath), range(len(self.datapath))))
            self.datas = list(filter(None, self.datas))

            with open(self.path_dataset + ".pkl", "wb") as fp:  # Pickling
                pickle.dump(self.datas, fp)

        logger.info("dataset : " + str(len(self.datas)))
        end = time.time()
        logger.info("Ellapsed time : " + '"%.2f' % (end - start))

    def generate_parts_by_cat(self):
        parts_by_cat = {}
        for cat in self.meta.keys():
            labels = set([])
            for file in self.meta[cat]:
                points = np.loadtxt(file[0])
                labels = labels | set(points[:, 6])
            print(cat, labels)
            parts_by_cat[cat] = list(labels)
        return parts_by_cat

    @staticmethod
    def normalize(x, isV2=False):
        v2_rot_mat = pc_utils.get_3D_rot_matrix(1, np.pi/2).astype(np.float32)
        pc = pc_utils.normalize_to_box(x)[0]
        # place on the "ground"
        ground = pc[:,1].min()
        pc[:,1] -= (ground+1.0)
        if isV2:
            x[:,:3] = x[:,:3].dot(v2_rot_mat)
            if x.shape[1] > 3:
                x[:,3:] = x[:,3:].dot(v2_rot_mat)
        return pc


    def getAnItem(self, index):
        """
        returns:
            points      : float (N,6) positions and normals
            cat         : string category
            file_path   : string
        """
        # ----------------------------------------------------------#
        points, cat, file_path = self.datas[index]
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        if points is None or not self.use_preprocessed:
            found = glob(os.path.splitext(file_path)[0]+".*")
            if len(found) == 0:
                # use mesh
                mesh_path = os.path.join(self.mesh_dir, self.namecat2numbercat[cat], file_name, "model.obj")
                mesh = pymesh.load_mesh(mesh_path)
                mesh = pymesh.split_long_edges(mesh, 0.02)[0]
                points = mesh.vertices
            else:
                points = pc_utils.load(found[0])

            points = self.normalize(points[:,:3])
            points = torch.from_numpy(points).float()

        points = points.clone()
        # Resample
        if self.sample:
            choice = np.random.choice(points.size(0), self.npoints, replace=True)
            points = points[choice, :]

        rot_matrix = pc_utils.uniform_rotation_axis_matrix(axis=1, range_rot=self.data_augmentation_Z_rotation_range)
        if self.data_augmentation_Z_rotation:
            # Uniform random Rotation of axis Y
            points, rot_matrix = pc_utils.uniform_rotation_axis(points, axis=1, normals=self.normal,
                                                                        range_rot=self.data_augmentation_Z_rotation_range)
        if self.anisotropic_scaling:
            # Data augmentation : anisotropic scaling
            points[:, :3] = pc_utils.anisotropic_scaling(points[:, :3]).contiguous()
            points[:, :3] = self.normalize(points[:, :3])

        if self.data_augmentation_3D_rotation:
            #  Uniform random 3D rotation of the sphere.
            points, rot_matrix = pc_utils.uniform_rotation_sphere(points, normals=self.normal)
        # Remark : if input of data_augmentation is normalized to unit ball and centered, the output rotated is as well.
        if self.random_translation:
            points = pc_utils.add_random_translation(points, scale=0.03)
        # also load mesh during test
        if self.phase == "test" and self.mesh_dir is not None:
            mesh_path = os.path.join(self.mesh_dir, self.namecat2numbercat[cat], file_name, "model.obj")
            V_mesh, F_mesh = read_trimesh(mesh_path)
            V_mesh = V_mesh[:,:3]
            F_mesh = F_mesh[:,:3]
            # V_mesh = (V_mesh - center)/scale
            V_mesh = self.normalize(V_mesh)

            V_mesh = torch.from_numpy(V_mesh).to(dtype=torch.float)
            F_mesh = torch.from_numpy(F_mesh).to(dtype=torch.int64)
            return points, V_mesh, F_mesh, cat, file_name
        else:
            return points, cat, file_name
        # ----------------------------------------------------------#

    def __getitem__(self, index):
        if self.use_fixed_pairs:
            # _, cat, filename = self.datas[index]
            # target = self.source_target_pairs[self.namecat2numbercat[cat]+'/'+os.path.basename(filename)]
            # index_2 = [i for i, tupl in enumerate(self.datas) if ((self.namecat2numbercat[tupl[1]]+'/'+os.path.basename(tupl[2])) == target)]
            # assert(len(index_2) == 1)
            # index_2 = index_2[0]
            index_2 = index + len(self.source_target_pairs)//2
        else:
            index = self.shuffle_list[index]
            if self.knn:
                index_2 = self.indices[index][np.random.randint(self.num_neighbors)]
            else:
                index_2 = np.random.randint(self.__len__())

        source_data = self.getAnItem(index)
        target_data = self.getAnItem(index_2)

        if len(source_data)==5 and len(target_data) == 5:
            return {"source_shape": source_data[0], "source_cat": source_data[3], "source_file": source_data[4], "source_mesh": source_data[1], "source_face": source_data[2],
                    "target_shape": target_data[0], "target_cat": target_data[3], "target_file": target_data[4], "target_mesh": target_data[1], "target_face": target_data[2]}
        else:
            return {"source_shape": source_data[0], "source_cat": source_data[1], "source_file": source_data[2],
                    "target_shape": target_data[0], "target_cat": target_data[1], "target_file": target_data[2]}

    @staticmethod
    def uncollate(batch_data):
        source_shape, target_shape = batch_data["source_shape"], batch_data["target_shape"]
        source_normals = source_shape[:,:,3:6].detach().contiguous().cuda()
        target_normals = target_shape[:,:,3:6].detach().contiguous().cuda()
        source_label = source_shape[:,:,-1].detach().contiguous().cuda()
        target_label = target_shape[:,:,-1].detach().contiguous().cuda()
        source_shape = source_shape[:,:,:3].detach().contiguous().cuda()
        target_shape = target_shape[:,:,:3].detach().contiguous().cuda()
        source_shape = source_shape.detach().cuda()
        target_shape = target_shape.detach().cuda()
        source_mesh = None if "source_mesh" not in batch_data else batch_data["source_mesh"].detach().cuda()
        target_mesh = None if "target_mesh" not in batch_data else batch_data["target_mesh"].detach().cuda()
        source_face = None if "source_face" not in batch_data else batch_data["source_face"].detach().cuda()
        target_face = None if "target_face" not in batch_data else batch_data["target_face"].detach().cuda()
        return {"source_shape": source_shape, "source_normals": source_normals, "source_label": source_label, "source_file": batch_data["source_file"],
                "target_shape": target_shape, "target_normals": target_normals, "target_label": target_label, "target_file": batch_data["target_file"],
                "source_mesh": source_mesh, "source_face": source_face, "target_mesh": target_mesh, "target_face": target_face}

    def __len__(self):
        if self.use_fixed_pairs and hasattr(self, "source_target_pairs"):
            if self.num_samples > 0:
                return min(self.num_samples, len(self.source_target_pairs))
            return len(self.source_target_pairs)

        if self.num_samples > 0:
            return min(len(self.datas), self.num_samples)
        else:
            return len(self.datas)

    @staticmethod
    def render_result(shape_dir, **kwargs):
        """render shapes inside a directory with thea"""
        forward = (-1,-0.4,-1)
        pos = (4,1.4,4)
        up = (-0.1,0.2,-0.1)
        # if "color" not in kwargs:
        #     kwargs["color"] = "eebad6a9"
        renderMeshes(shape_dir, up=up, forward=forward, pos=pos, **kwargs)

class ShapeNetV2(ShapeNetSeg):
    def __init__(self, root_dir="/home/mnt/points/data/ShapeNet/ShapeNetCore.v2.5000p",
                 mesh_dir="/home/mnt/points/data/ShapeNet/ShapeNetCore.v2.remesh",
                 num_samples=-1, normalization="BoundingBox", knn=False,
                 num_neighbors=40,
                 class_choice=None, phase="train", normal=False, npoints=5000, sample=True,
                 data_augmentation_Z_rotation=False, data_augmentation_Z_rotation_range=360,
                 data_augmentation_3D_rotation=False, random_translation=False, anisotropic_scaling=False,
                 shuffle=False, use_fixed_pairs=False, **kwargs):
        torch.utils.data.Dataset.__init__(self)
        self.mesh_data = False
        self.mesh_dir = mesh_dir
        self.dataset_string_args = str(phase) + "_" + \
                                   str(class_choice) + \
                                   "_" + str(num_samples) + \
                                   "_" + str(normalization) + \
                                   "_" + str(knn) + \
                                   "_" + str(num_neighbors) + \
                                   "_" + str(normal) + \
                                   "_" + str(5000) + \
                                   "_" + str(sample)
        self.path_dataset = os.path.join("./data/processed_shapenetv2", self.dataset_string_args)
        os.makedirs(os.path.dirname(self.path_dataset), exist_ok=True)
        self.shuffle = shuffle
        self.num_samples = num_samples
        self.anisotropic_scaling = anisotropic_scaling
        self.knn = knn
        self.num_neighbors = num_neighbors
        self.normalization = normalization
        self.random_translation = random_translation
        self.data_augmentation_Z_rotation = data_augmentation_Z_rotation
        self.data_augmentation_Z_rotation_range = data_augmentation_Z_rotation_range  # range in degree of random rotation
        self.data_augmentation_3D_rotation = data_augmentation_3D_rotation
        self.sym_plane = ("yz", )
        self.npoints = npoints
        self.sample = sample
        self.normal = normal
        self.phase = phase
        self.root = root_dir
        self.datapath = []  # List to store all path of dataset files
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.namecat2numbercat = {}
        self.numbercat2namecat = {}
        self.numsamples_by_cat = {}
        self.class_choice = class_choice
        self.meta = {}
        self.data = []
        self.use_fixed_pairs = use_fixed_pairs
        # if self.shuffle and self.use_fixed_pairs:
        #     logger.warn("{}: shuffle and use_fixed_pairs is exclusive, shuffle to False".format(__name__))
        # if not os.path.exists(self.root):
        #     print("Downloading Shapetnet for segmentation...")
        #     os.system('chmod +x ./data/download_dataset_shapenet.sh')
        #     os.system('./data/download_dataset_shapenet.sh')
        # ----------------------------------------------------------#
        ## Create dictionaries with keys : name of class, value : name of folder and vice-versa
        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.numsamples_by_cat[ls[0]] = 0
                self.namecat2numbercat[ls[0]] = ls[1]
                self.numbercat2namecat[ls[1]] = ls[0]
                self.meta[ls[0]] = []  # List to store all path of files per category

        # ----------------------------------------------------------#
        # create list of files datapaths
        # cat/name 0/1 (0:train 1:test)
        file_list = []
        split_file = None
        if class_choice is None:
            split_files = glob(os.path.join(self.root, "*_shuffled_train_test_split.txt"))
        else:
            split_files = [os.path.join(self.root, "{}_shuffled_train_test_split.txt".format(synnum))
                for synname, synnum in self.namecat2numbercat.items() if synname in class_choice]
        for split_file in split_files:
            if os.path.isfile(split_file):
                phase_token = "0" if self.phase=="train" else "1"
                with open(split_file, "r") as f:
                    for l in f:
                        if l[-2:-1] == phase_token:
                            file_list.append(l.split()[0])
            else:
                with open(split_file, "w") as f:
                    remove_str = len("/models/model_normalized.pts")
                    for k in self.numbercat2namecat:
                        all_files_in_cat = find_files(os.path.join(self.root, k), ["ply","pts"])
                        np.random.shuffle(all_files_in_cat)
                        num_test = max(int(0.05 * len(all_files_in_cat)), 10)
                        for i, l in enumerate(all_files_in_cat):
                            l = os.path.relpath(l[:-remove_str], self.root)
                            f.write(l+" "+str(int(i<num_test))+"\n")
                            if (self.phase == "train" and i >= num_test) or (self.phase=="test" and i<num_test):
                                file_list.append(l)

        for file in file_list:
            # Typical example : 03001627/355fa0f35b61fdd7aa74a6b5ee13e775 so remove 'shape_data/' and add '.txt'
            file_path = glob(os.path.join(self.root, file, "models", "model_normalized.*"))[0]
            number_category = file[:8]
            if (class_choice is None) or (self.numbercat2namecat[number_category] in class_choice):
                self.meta[self.numbercat2namecat[number_category]].append(
                    (file_path, self.numbercat2namecat[number_category]))
                self.numsamples_by_cat[self.numbercat2namecat[number_category]] = self.numsamples_by_cat[
                                                                                      self.numbercat2namecat[
                                                                                          number_category]] + 1
        # ----------------------------------------------------------#
        # add all paths to the same list, keep track of sizes
        for cat in self.meta.keys():
            for file in self.meta[cat]:
                self.datapath.append(file)

        logger.info("size of " + self.phase + " dataset : " + str(len(self.datapath)))


        self.preprocess()
        if self.knn:
            start = time.time()
            logger.info(None,
                "Computing nearest neighbors graph... (can take some time if it's not already precomputed.)")
            self.compute_nearest_neighbors_graph()
            logger.success('Done!')
            end = time.time()
            logger.info("Ellapsed time : " + '"%.2f' % (end - start))

        # fixed pairs? create or load a dict of source_target_pairs storing the cat/file_name: cat/file_name
        if self.use_fixed_pairs and self.phase in ("val", "test"):
            if os.path.exists(self.path_dataset + "_pairs.txt"):
                source_target_file = np.loadtxt(self.path_dataset + "_pairs.txt", dtype=str)
                self.source_target_pairs = dict(source_target_file)
            else:
                self.source_target_pairs = dict()
                for i in range(len(self.datas)):
                    _, cat, filename = self.datas[i]
                    remove_str = len("/models/model_normalized.pts")
                    filename = os.path.basename(filename[:-remove_str])
                    if self.knn:
                        j = self.indices[i][np.random.randint(self.num_neighbors)]
                    else:
                        j = np.random.randint(len(self.datas))
                    _, cat2, filename2 = self.datas[j]
                    filename2 = os.path.basename(filename2[:-remove_str])
                    self.source_target_pairs[self.namecat2numbercat[cat]+"/"+filename] = self.namecat2numbercat[cat2]+"/"+filename2
                with open(self.path_dataset + "_pairs.txt","w") as f:
                    for key, value in self.source_target_pairs.items():
                        f.write("{} {}\n".format(key, value))


        # Shuffle_list
        self.len_data = len(self.datas)
        self.shuffle_list = [i for i in range(self.len_data)]
        if self.shuffle:
            np.random.shuffle(self.shuffle_list)


    @staticmethod
    def normalize(x):
        pc = pc_utils.normalize_to_box(x)[0]
        return pc

    def getAnItem(self, index):
        """
        returns:
            points      : float (N,6) positions and normals
            cat         : string category
            file_path   : string
        """
        # ----------------------------------------------------------#
        points, cat, file_path = self.datas[index]
        remove_str = len("/models/model_normalized.pts")
        file_name = os.path.basename(file_path[:-remove_str])
        points = points.clone()
        # Resample
        if self.sample:
            choice = np.random.choice(points.size(0), self.npoints, replace=True)
            points = points[choice, :]

        rot_matrix = pc_utils.uniform_rotation_axis_matrix(axis=1, range_rot=self.data_augmentation_Z_rotation_range)
        if self.anisotropic_scaling:
            # Data augmentation : anisotropic scaling
            points[:, :3] = pc_utils.anisotropic_scaling(points[:, :3]).contiguous()
            points[:, :3] = self.normalize(points[:, :3])

        # Remark : if input of data_augmentation is normalized to unit ball and centered, the output rotated is as well.
        if self.random_translation:
            points = pc_utils.add_random_translation(points, scale=0.03)
        # also load mesh during test
        if self.phase == "test" and self.mesh_dir is not None:
            mesh_path = os.path.join(self.mesh_dir, self.namecat2numbercat[cat], file_name, "models", "model_normalized.obj")
            V_mesh, F_mesh = read_trimesh(mesh_path)
            V_mesh = V_mesh[:,:3]
            F_mesh = F_mesh[:,:3]
            # V_mesh = (V_mesh - center)/scale
            V_mesh = self.normalize(V_mesh)

            V_mesh = torch.from_numpy(V_mesh).to(dtype=torch.float)
            F_mesh = torch.from_numpy(F_mesh).to(dtype=torch.int64)
            return points, V_mesh, F_mesh, cat, file_name
        else:
            return points, cat, file_name

    def __getitem__(self, index):
        if self.use_fixed_pairs:
            _, cat, filename = self.datas[index]
            remove_str = len("/models/model_normalized.pts")
            target = self.source_target_pairs[self.namecat2numbercat[cat]+'/'+os.path.basename(filename[:-remove_str])]
            index_2 = [i for i, tupl in enumerate(self.datas) if ((self.namecat2numbercat[tupl[1]]+'/'+os.path.basename(tupl[2][:-remove_str])) == target)]
            assert(len(index_2) == 1)
            index_2 = index_2[0]
        else:
            index = self.shuffle_list[index]
            if self.knn:
                index_2 = self.indices[index][np.random.randint(self.num_neighbors)]
            else:
                index_2 = np.random.randint(self.__len__())

        source_data = self.getAnItem(index)
        target_data = self.getAnItem(index_2)
        if len(source_data)==5 and len(target_data) == 5:
            return {"source_shape": source_data[0], "source_cat": source_data[3], "source_file": source_data[4], "source_mesh": source_data[1], "source_face": source_data[2],
                    "target_shape": target_data[0], "target_cat": target_data[3], "target_file": target_data[4], "target_mesh": target_data[1], "target_face": target_data[2]}
        else:
            return {"source_shape": source_data[0], "source_cat": source_data[1], "source_file": source_data[2],
                    "target_shape": target_data[0], "target_cat": target_data[1], "target_file": target_data[2]}

    @staticmethod
    def render_result(shape_dir, **kwargs):
        """render shapes inside a directory with thea"""
        forward = (-1,-0.65,1)
        pos = (3.05,1.83,-3.05)
        up = (0.1,0.4,-0.1)
        renderMeshes(shape_dir, up=up, forward=forward, pos=pos, **kwargs)

class PairedSurreal(torch.utils.data.Dataset):
    def __init__(self, root_dir, phase="train", npoints=6890, regular_sampling=False, normal=False, data_augmentation_Z_rotation=False,
                 data_augmentation_Z_rotation_range=360, data_augmentation_3D_rotation=False, max=-1,
                 template="./vanilla_data/surreal_template_v77.ply", source="./vanilla_data/surreal_template.ply"):
        super().__init__()
        self.data_augmentation_Z_rotation = data_augmentation_Z_rotation
        self.data_augmentation_Z_rotation_range = data_augmentation_Z_rotation_range
        self.data_augmentation_3D_rotation = data_augmentation_3D_rotation
        self.sym_plane = None
        self.normal = normal
        self.phase = phase
        self.train = self.phase == "train"
        self.regular_sampling = regular_sampling  # sample points uniformly or proportionaly to their adjacent area
        self.npoints = npoints
        template = template or "./vanilla_data/surreal_template_v77.ply"
        source = source or "./vanilla_data/surreal_template.ply"

        self.datas = []
        start = time.time()
        if self.phase == "train":
            self.datas = torch.load(os.path.join(root_dir, "datas_surreal_train.pth"))
        else:
            self.datas = torch.load(os.path.join(root_dir, "datas_surreal_test.pth"))

        end = time.time()
        print("Ellapsed time to load dataset: ", end - start)
        # template
        # if not os.path.exists("./data/template/template.ply"):
        #     os.system("chmod +x ./data/download_template.sh")
        #     os.system("./data/download_template.sh")

        self.mesh_data = True
        if self.phase == "train":
            self.len = len(self.datas) if max < 0 else max # self.datas.shape[0] 230000
        else:
            self.len = 100 if max < 0 else max

        self.mesh_vertex, self.mesh_face = read_trimesh(source)
        self.mesh_vertex = self.mesh_vertex[:,:3].astype(np.float32)
        self.mesh_face = torch.from_numpy(self.mesh_face[:,:3].astype(np.int64))
        self.mesh_vertex = torch.from_numpy(self.mesh_vertex)
        V, center, scale = pc_utils.center_bounding_box(self.mesh_vertex)
        self.mesh_vertex = V.squeeze(0)
        center = center.squeeze(0)
        scale = scale.squeeze(0)

        self.cage_vertex, self.cage_face = read_trimesh(template, normal=True)
        if "surreal_template_" in template :
            self.cage_vertex = self.cage_vertex[:,:3].astype(np.float32) + 0.05*self.cage_vertex[:,3:6].astype(np.float32)
        else:
            self.cage_vertex = self.cage_vertex[:,:3].astype(np.float32)
        self.cage_face = torch.from_numpy(self.cage_face[:,:3].astype(np.int64))
        self.cage_vertex = torch.from_numpy(self.cage_vertex)
        self.cage_vertex = (self.cage_vertex - center)

        _, farea = compute_face_normals_and_areas(self.mesh_vertex.unsqueeze(0), self.mesh_face.unsqueeze(0))
        v_area = scatter_add(farea.view(-1,1).expand(-1,3).contiguous().view(-1), self.mesh_face.view(-1), 0, out_size=(self.mesh_vertex.shape[0],))
        self.prob = (v_area / torch.sum(v_area)).numpy()


    def getPairIdx(self, index):
        target_index = index
        source_index = np.random.choice(self.len)
        return (source_index, target_index)

    def __getitem__(self, index):
        # Clone it to keep the cached data safe
        outputs = {}
        # Resample according to triangles area
        random_sample = None
        if self.npoints != 6890:
            if self.regular_sampling:
                random_sample = np.random.choice(6890, size=self.npoints, p=self.prob)
            else:
                random_sample = np.random.choice(6890, size=self.npoints, replace=False)
            outputs["sample_idx"] = torch.from_numpy(random_sample.astype(np.int64))

        for i, index in enumerate(self.getPairIdx(index)):
            tag = "source" if i == 0 else "target"

            # LOAD a training sample
            points = self.datas[index].squeeze()
            points = points.clone()
            points, _, _ = pc_utils.center_bounding_box(points)
            # Add small random translation
            # if self.train:
            #     points = pc_utils.add_random_translation(points, scale=0.03)

            outputs["%s_mesh" % tag] = points
            if random_sample is not None:
                points = points[random_sample]

            outputs["%s_shape" % tag] = points
            # append 4 char because
            outputs["%s_file" % tag] = "{:04d}".format(index)

        outputs["source_face"] = self.mesh_face.clone()
        outputs["target_face"] = self.mesh_face.clone()
        return outputs

    @staticmethod
    def uncollate(batch_data):
        source_shape, target_shape = batch_data["source_shape"], batch_data["target_shape"]
        source_normals = None
        target_normals = None
        source_label = source_shape[:,:,-1].detach().contiguous().cuda()
        target_label = target_shape[:,:,-1].detach().contiguous().cuda()
        source_shape = source_shape[:,:,:3].detach().contiguous().cuda()
        target_shape = target_shape[:,:,:3].detach().contiguous().cuda()
        source_shape = source_shape.detach().cuda()
        target_shape = target_shape.detach().cuda()
        source_mesh = None if "source_mesh" not in batch_data else batch_data["source_mesh"].detach().cuda()
        target_mesh = None if "target_mesh" not in batch_data else batch_data["target_mesh"].detach().cuda()
        source_face = None if "source_face" not in batch_data else batch_data["source_face"].detach().cuda()
        target_face = None if "target_face" not in batch_data else batch_data["target_face"].detach().cuda()

        return_d = {"source_shape": source_shape, "source_normals": source_normals, "source_label": source_label, "source_file": batch_data["source_file"],
                "target_shape": target_shape, "target_normals": target_normals, "target_label": target_label, "target_file": batch_data["target_file"],
                "source_mesh": source_mesh, "source_face": source_face, "target_mesh": target_mesh, "target_face": target_face}
        if "sample_idx" in batch_data:
            return_d["sample_idx"] = batch_data["sample_idx"].detach().cuda()

        return return_d

    def __len__(self):
        return self.len

    @staticmethod
    def render_result(shape_dir, **kwargs):
        # "e0f2d79b" mustard yellow
        renderMeshes(shape_dir, up=(0,1,0), forward=(0.2,0,-1), pos=(-1.2, 0, 6), **kwargs)


if __name__ == "__main__":
    dataset = PairedSurreal("/home/mnt/points/data/Surreal", npoints=6890, regular_sampling=False)
    for i in range(len(dataset)):
        data = dataset[i]
        # pc_utils.save_ply_property(data["source_shape"][:,:3].numpy(), data["source_shape"][:,2].numpy(), "./surreal_source.ply", cmap_name="rainbow")
        # pc_utils.save_ply_property(data["target_shape"][:,:3].numpy(), data["source_shape"][:,2].numpy(), "./surreal_target.ply", cmap_name="rainbow")
        pc_utils.save_ply_property(data["target_shape"][:,:3].numpy(), data["target_shape"][:,2].numpy(), "/home/mnt/points/data/Surreal/all/{}.ply".format(i), cmap_name="rainbow")
        # mesh_points = torch.gather(dataset.mesh_vertex, 0, data["sample_idx"].unsqueeze(-1).expand(-1,3))
        # pc_utils.save_ply_property(mesh_points.numpy(), data["source_shape"][:,2].numpy(), "./surreal_template.ply", cmap_name="rainbow")
        if (i+1) % 100 == 0:
            print("{}/{}".format(i, len(dataset)))

    # namecat2numbercat = {}
    # numbercat2namecat = {}
    # shapenet_path = "/home/mnt/points/data/ShapeNet/PartSeg_v0/shapenetcore_partanno_segmentation_benchmark_v0_normal"
    # with open(os.path.join(shapenet_path,"synsetoffset2category.txt"), 'r') as f:
    #     for line in f:
    #         ls = line.strip().split()
    #         namecat2numbercat[ls[0]] = ls[1]
    #         numbercat2namecat[ls[1]] = ls[1]
    # for phase in ("train", "test"):
    #     for catname in ("Chair", "Car", "Lamp", "Table"):
    #         dataset = ShapeNetSeg(root_dir=shapenet_path,
    #                       phase=phase,
    #                       knn=True,
    #                       num_neighbors=40,
    #                       normalization="BoundingBox",
    #                       class_choice=catname,
    #                       data_augmentation_Z_rotation=False,
    #                       data_augmentation_Z_rotation_range=40,
    #                       anisotropic_scaling=1,
    #                       npoints=2500,
    #                       random_translation=False)
    #         dataset = ShapeNetSeg(root_dir=shapenet_path,
    #                       phase=phase,
    #                       knn=False,
    #                       num_neighbors=40,
    #                       normalization="BoundingBox",
    #                       class_choice="Airplane",
    #                       data_augmentation_Z_rotation=False,
    #                       data_augmentation_Z_rotation_range=40,
    #                       anisotropic_scaling=1,
    #                       npoints=2500,
    #                       random_translation=False)

    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=(PHASE=="train"), drop_last=(PHASE=="train"))
    # cnt = 0
    # for data in dataloader:
    #     source_shape, source_cat, _, source_filenames, target_shape, target_cat, _, target_filenames = data
    #     s_filename = source_filenames[0]
    #     t_filename = target_filenames[0]
    #     source_cat = source_cat[0]
    #     target_cat = target_cat[0]
    #     source_shape = source_shape.numpy()[0]
    #     target_shape = target_shape.numpy()[0]
    #     pc_utils.save_ply_property(source_shape[:,:3], source_shape[:,6], os.path.join("./data-test", "shapenetseg", "{}-Sa-{}-{}.ply".format(cnt, source_cat, s_filename)), normals=source_shape[:,3:6])
    #     pc_utils.save_ply_property(target_shape[:,:3], target_shape[:,6], os.path.join("./data-test", "shapenetseg", "{}-Sb-{}-{}.ply".format(cnt, target_cat, t_filename)), normals=target_shape[:,3:6])
    #     cnt += 1
    #     if cnt > 3:
    #         break

    # DATA_DIR = "/home/yifan/Data/font_skeletons/font_skeleton_normal"
    # dataset = CrossCategoryPairsDataset(DATA_DIR, phase=PHASE, num_point=800, training_size=0.9, source_digits=["Aup"], target_digits=["Aup"])
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=(PHASE=="train"), drop_last=(PHASE=="train"))
    # cnt = 0
    # for data in dataloader:
    #     source_shape, target_shape, source_filenames, target_filenames = data
    #     # source_shape, source_normals = torch.chunk(source_shape, 2, dim=1)
    #     # target_shape, target_normals = torch.chunk(target_shape, 2, dim=1)
    #     source_shape = source_shape.transpose(1,2).numpy()[0]
    #     target_shape = target_shape.transpose(1,2).numpy()[0]
    #     s_filename = source_filenames[0]
    #     t_filename = target_filenames[0]
    #     pc_utils.save_ply(source_shape[:,:2], os.path.join("./data-test", "crossdata-{}-Sa.ply".format(s_filename)), normals=source_shape[:,2:])
    #     pc_utils.save_ply(target_shape[:,:2], os.path.join("./data-test", "crossdata-{}-Sb.ply".format(t_filename)), normals=target_shape[:,2:])
    #     cnt += 1
    #     if cnt > 3:
    #         break

    # dataset = SameCategoryPairsDataset(DATA_DIR, phase=PHASE, num_point=800, training_size=0.9, categories=["Aup", "Bup"])
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=(PHASE=="train"), drop_last=(PHASE=="train"))
    # cnt = 0
    # for data in dataloader:
    #     source_shape, target_shape, source_filenames, target_filenames, cat_id = data
    #     # source_shape, source_normals = torch.chunk(source_shape, 2, dim=1)
    #     # target_shape, target_normals = torch.chunk(target_shape, 2, dim=1)
    #     source_shape = source_shape.transpose(1,2).numpy()[0]
    #     target_shape = target_shape.transpose(1,2).numpy()[0]
    #     s_filename = source_filenames[0]
    #     t_filename = target_filenames[0]
    #     pc_utils.save_ply(source_shape[:,:2], os.path.join("./data-test", "samedata-{}-Sa.ply".format(cnt)), normals=source_shape[:,2:])
    #     pc_utils.save_ply(target_shape[:,:2], os.path.join("./data-test", "samedata-{}-Sb.ply".format(cnt)), normals=target_shape[:,2:])
    #     cnt += 1
    #     if cnt > 3:
    #         break
