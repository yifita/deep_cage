""" Optimize the initial cage for a new source shape """
from __future__ import print_function
from pprint import pprint
import traceback
import sys
import datetime
import shutil
import torch
import os
import numpy as np
import pymesh
from pytorch_points.misc import logger
from pytorch_points.network.operations import faiss_knn
from pytorch_points.network.geo_operations import mean_value_coordinates_3D, edge_vertex_indices
from pytorch_points.utils.pc_utils import load, save_ply, save_ply_with_face, center_bounding_box
from pytorch_points.utils.geometry_utils import get_edge_points
from pytorch_points.utils.pytorch_utils import weights_init, check_values, save_network, load_network, \
    clamp_gradient_norm, tolerating_collate, clamp_gradient, fix_network_parameters
from pytorch_points.network.model_loss import MeshLaplacianLoss
import network2 as networks
from common import loadInitCage, build_dataset, deform_with_MVC, read_trimesh
from losses import MVCRegularizer
from option import DeformationOptions
from matplotlib.colors import Normalize
from matplotlib import cm
import openmesh as om


class MyOptions(DeformationOptions):
    def initialize(self, parser):
        parser.add_argument(
            "--model", type=str, default="/home/mnt/points/data/MPI-FAUST/training/registrations/tr_reg_010.ply")
        parser.add_argument("--use_cage", type=str,
                            help="path to optimized cage")
        parser.add_argument("--opt_lap", action="store_true",
                            help="optimize deformed shape using laplacian")
        return super().initialize(parser)

    def parse(self):
        super().parse()
        assert(self.opt.source_model is not None), "source model is required for optimize_cage"
        if not isinstance(self.opt.source_model, str):
            self.opt.source_model = self.opt.source_model[0]
        return self.opt


def visualize_correspondence(opt, source_shape, source_face, target_shape, target_face, corres_1, corres_2):
    """
    source_shape (1,N,3)
    source_face (1,F,3)
    target_shape (1,N2,3)
    target_face (1,F2,3)
    corres_face_1 (P)
    corres_face_2 (P)
    """
    # save these points with color codes
    P = corres_2.shape[0]
    assert(corres_1.shape[0] == corres_2.shape[0])
    corres_1 = corres_1.cpu().numpy().reshape(-1)
    corres_2 = corres_2.cpu().numpy().reshape(-1)
    normalize = Normalize(vmin=0, vmax=corres_1.shape[0])
    cmap = cm.get_cmap("jet")
    colors_picked = cmap(normalize(np.arange(P, dtype=np.float32)))[:, :3]
    colors_source = np.ones((source_face.shape[1], 3), dtype=np.float32)
    colors_source[corres_1, :] = colors_picked
    save_ply_with_face(source_shape[0].cpu().detach().numpy(), source_face[0].cpu().detach().numpy(),
                       os.path.join(opt.log_dir, opt.subdir, "source_corr.ply"), colors_source)
    colors_target = np.ones((target_face.shape[1], 3), dtype=np.float32)
    colors_target[corres_2, :] = colors_picked
    save_ply_with_face(target_shape[0].cpu().detach().numpy(), target_face[0].cpu().detach().numpy(),
                       os.path.join(opt.log_dir, opt.subdir, "target_corr.ply"), colors_target)


def optimize(opt):
    """
    weights are the same with the original source mesh
    target=net(old_source)
    """
    # load new target
    if opt.is_poly:
        target_mesh = om.read_polymesh(opt.model)
    else:
        target_mesh = om.read_trimesh(opt.model)
    target_shape_arr = target_mesh.points()
    target_shape = target_shape_arr.copy()
    target_shape = torch.from_numpy(
        target_shape[:, :3].astype(np.float32)).cuda()
    target_shape.unsqueeze_(0)

    states = torch.load(opt.ckpt)
    if "states" in states:
        states = states["states"]
    cage_v = states["template_vertices"].transpose(1, 2).cuda()
    cage_f = states["template_faces"].cuda()
    shape_v = states["source_vertices"].transpose(1, 2).cuda()
    shape_f = states["source_faces"].cuda()

    if os.path.isfile(opt.model.replace(os.path.splitext(opt.model)[1], ".picked")) and os.path.isfile(opt.source_model.replace(os.path.splitext(opt.source_model)[1], ".picked")):
        new_label_path = opt.model.replace(os.path.splitext(opt.model)[1], ".picked")
        orig_label_path = opt.source_model.replace(os.path.splitext(opt.source_model)[1], ".picked")
        logger.info("Loading picked labels {} and {}".format(orig_label_path, new_label_path))
        import pandas as pd
        new_label = pd.read_csv(new_label_path, delimiter=" ",skiprows=1, header=None)
        orig_label = pd.read_csv(orig_label_path, delimiter=" ",skiprows=1, header=None)
        orig_label_name = orig_label.iloc[:,5]
        new_label_name = new_label.iloc[:,5].tolist()
        new_to_orig_idx = []
        for i, name in enumerate(new_label_name):
            matched_idx = orig_label_name[orig_label_name==name].index
            if matched_idx.size == 1:
                new_to_orig_idx.append((i, matched_idx[0]))
        new_to_orig_idx = np.array(new_to_orig_idx)
        if new_label.shape[1] == 10:
            new_vidx = new_label.iloc[:,9].to_numpy()[new_to_orig_idx[:,0]]
            target_points = target_shape[:, new_vidx, :]
        else:
            new_label_points = torch.from_numpy(new_label.iloc[:,6:9].to_numpy().astype(np.float32))
            target_points = new_label_points.unsqueeze(0).cuda()
            target_points, new_vidx, _ = faiss_knn(1, target_points, target_shape, NCHW=False)
            target_points = target_points.squeeze(2) # B,N,3
            new_label[9] = new_vidx.squeeze(0).squeeze(-1).cpu().numpy()
            new_label.to_csv(new_label_path, sep=" ", header=[str(new_label.shape[0])]+[""]*(new_label.shape[1]-1), index=False)
            target_points = target_points[:, new_to_orig_idx[:,0], :]

        target_points = target_points.cuda()
        source_shape, _ = read_trimesh(opt.source_model)
        source_shape = torch.from_numpy(source_shape[None, :,:3]).float()
        if orig_label.shape[1] == 10:
            orig_vidx = orig_label.iloc[:,9].to_numpy()[new_to_orig_idx[:,1]]
            source_points = source_shape[:, orig_vidx, :]
        else:
            orig_label_points = torch.from_numpy(orig_label.iloc[:,6:9].to_numpy().astype(np.float32))
            source_points = orig_label_points.unsqueeze(0)
            # find the closest point on the original meshes
            source_points, new_vidx, _ = faiss_knn(1, source_points, source_shape, NCHW=False)
            source_points = source_points.squeeze(2) # B,N,3
            orig_label[9] = new_vidx.squeeze(0).squeeze(-1).cpu().numpy()
            orig_label.to_csv(orig_label_path, sep=" ", header=[str(orig_label.shape[0])]+[""]*(orig_label.shape[1]-1), index=False)
            source_points = source_points[:,new_to_orig_idx[:,1],:]

        _, source_center, _ = center_bounding_box(source_shape[0])
        source_points -= source_center
        source_points = source_points.cuda()
        # # shift target so that the belly match
        # try:
        #     orig_bellyUp_idx = orig_label_name[orig_label_name=="bellUp"].index[0]
        #     orig_bellyUp = orig_label_points[orig_bellyUp_idx, :]
        #     new_bellyUp_idx = [i for i, i2 in new_to_orig_idx if i2==orig_bellyUp_idx][0]
        #     new_bellyUp = new_label_points[new_bellyUp_idx,:]
        #     target_points += (orig_bellyUp - new_bellyUp)
        # except Exception as e:
        #     logger.warn("Couldn\'t match belly to belly")
        #     traceback.print_exc(file=sys.stdout)

        # source_points[0] = center_bounding_box(source_points[0])[0]
    elif not os.path.isfile(opt.model.replace(os.path.splitext(opt.model)[1], ".picked")) and os.path.isfile(opt.source_model.replace(os.path.splitext(opt.source_model)[1], ".picked")):
        logger.info("Assuming Faust model")
        orig_label_path = opt.source_model.replace(os.path.splitext(opt.source_model)[1], ".picked")
        logger.info("Loading picked labels {}".format(orig_label_path))
        import pandas as pd
        orig_label = pd.read_csv(orig_label_path, delimiter=" ",skiprows=1, header=None)
        orig_label_name = orig_label.iloc[:,5]
        source_shape, _ = read_trimesh(opt.source_model)
        source_shape = torch.from_numpy(source_shape[None, :,:3]).cuda().float()
        if orig_label.shape[1] == 10:
            idx = torch.from_numpy(orig_label.iloc[:,9].to_numpy()).long()
            source_points = source_shape[:,idx,:]
            target_points = target_shape[:,idx,:]
        else:
            source_points = torch.from_numpy(orig_label.iloc[:,6:9].to_numpy().astype(np.float32))
            source_points = source_points.unsqueeze(0).cuda()
            # find the closest point on the original meshes
            source_points, idx, _ = faiss_knn(1, source_points, source_shape, NCHW=False)
            source_points = source_points.squeeze(2) # B,N,3
            idx = idx.squeeze(-1)
            target_points = target_shape[:,idx,:]

        _, source_center, _ = center_bounding_box(source_shape[0])
        source_points -= source_center
    elif opt.corres_idx is None and target_shape.shape[1] == shape_v.shape[1]:
        logger.info("No correspondence provided, assuming registered Faust models")
        # corresp_idx = torch.randint(0, shape_f.shape[1], (100,)).cuda()
        corresp_v = torch.unique(torch.randint(0, shape_v.shape[1], (4800,))).cuda()
        target_points = torch.index_select(target_shape, 1, corresp_v)
        source_points = torch.index_select(shape_v, 1, corresp_v)

    target_shape[0], target_center, target_scale = center_bounding_box(target_shape[0])
    _, _, source_scale = center_bounding_box(shape_v[0])
    target_scale_factor = (source_scale/target_scale)[1]
    target_shape *= target_scale_factor
    target_points -= target_center
    target_points = (target_points*target_scale_factor).detach()
    # make sure test use the normalized
    target_shape_arr[:] = target_shape[0].cpu().numpy()
    om.write_mesh(os.path.join(opt.log_dir, opt.subdir, os.path.splitext(
        os.path.basename(opt.model))[0]+"_normalized.obj"), target_mesh)
    opt.model = os.path.join(opt.log_dir, opt.subdir, os.path.splitext(
        os.path.basename(opt.model))[0]+"_normalized.obj")
    pymesh.save_mesh_raw(os.path.join(opt.log_dir, opt.subdir, "template-initial.obj"),
                         shape_v[0].cpu().numpy(), shape_f[0].cpu().numpy())
    pymesh.save_mesh_raw(os.path.join(opt.log_dir, opt.subdir, "cage-initial.obj"),
                         cage_v[0].cpu().numpy(), cage_f[0].cpu().numpy())
    save_ply(target_points[0].cpu().numpy(), os.path.join(
        opt.log_dir, opt.subdir, "target_points.ply"))
    save_ply(source_points[0].cpu().numpy(), os.path.join(
        opt.log_dir, opt.subdir, "source_points.ply"))
    logger.info("Optimizing for {} corresponding vertices".format(
        target_points.shape[1]))

    cage_init = cage_v.clone().detach()
    lap_loss = MeshLaplacianLoss(torch.nn.MSELoss(reduction="none"), use_cot=True,
                                 use_norm=True, consistent_topology=True, precompute_L=True)
    mvc_reg_loss = MVCRegularizer(threshold=50, beta=1.0, alpha=0.0)
    cage_v.requires_grad_(True)
    optimizer = torch.optim.Adam([cage_v], lr=opt.lr, betas=(0.5, 0.9))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, int(opt.nepochs*0.4), gamma=0.5, last_epoch=-1)

    if opt.dim == 3:
        weights_ref = mean_value_coordinates_3D(
            source_points, cage_init, cage_f, verbose=False)
    else:
        raise NotImplementedError

    for t in range(opt.nepochs):
        optimizer.zero_grad()
        weights = mean_value_coordinates_3D(
            target_points, cage_v, cage_f, verbose=False)
        loss_mvc = torch.mean((weights-weights_ref)**2)
        # reg = torch.sum((cage_init-cage_v)**2, dim=-1)*1e-4
        reg = 0
        if opt.clap_weight > 0:
            reg = lap_loss(cage_init, cage_v, face=cage_f)*opt.clap_weight
            reg = reg.mean()
        if opt.mvc_weight > 0:
            reg += mvc_reg_loss(weights)*opt.mvc_weight

        # weight regularizer with the shape difference
        # dist = torch.sum((source_points - target_points)**2, dim=-1)
        # weights = torch.exp(-dist)
        # reg = reg*weights*0.1

        loss = loss_mvc + reg
        if (t+1) % 50 == 0:
            print("t {}/{} mvc_loss: {} reg: {}".format(t,
                                                        opt.nepochs, loss_mvc.item(), reg.item()))

        if loss_mvc.item() < 5e-6:
            break
        loss.backward()
        optimizer.step()
        scheduler.step()

    return cage_v, cage_f


def test_one(opt, cage_shape, new_source, new_source_face, new_target, new_target_face):
    states = torch.load(opt.ckpt)
    if "states" in states:
        states = states["states"]

    pymesh.save_mesh_raw(os.path.join(opt.log_dir, opt.subdir, "template-initial.ply"),
                         states["source_vertices"][0].transpose(
                             0, 1).detach().cpu(),
                         states["source_faces"][0].detach().cpu())

    # states["template_vertices"] = cage_shape.transpose(1, 2)
    # states["source_vertices"] = new_source.transpose(1, 2)
    # states["source_faces"] = new_source_face

    pymesh.save_mesh_raw(os.path.join(opt.log_dir, opt.subdir, "template-Sa.ply"),
                         new_source[0].detach().cpu(), new_source_face[0].detach().cpu())
    pymesh.save_mesh_raw(os.path.join(opt.log_dir, opt.subdir, "template-Sb.ply"),
                         new_target[0].detach().cpu(), new_target_face[0].detach().cpu())

    net = networks.FixedSourceDeformer(opt, 3, opt.num_point, bottleneck_size=512,
                                       template_vertices=cage_shape.transpose(1, 2), template_faces=states["template_faces"].cuda(),
                                       source_vertices=new_source.transpose(1, 2), source_faces=new_source_face).cuda()

    net.eval()
    load_network(net, states)

    outputs = net(new_target.transpose(1, 2).contiguous())
    deformed = outputs["deformed"]

    pymesh.save_mesh_raw(os.path.join(opt.log_dir, opt.subdir, "template-Sab.ply"),
                         deformed[0].detach().cpu(), new_target_face[0].detach().cpu())


def test_all(opt, new_cage_shape):
    opt.phase = "test"
    opt.target_model = None
    print(opt.model)

    if opt.is_poly:
        source_mesh = om.read_polymesh(opt.model)
    else:
        source_mesh = om.read_trimesh(opt.model)

    dataset = build_dataset(opt)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False,
                                             collate_fn=tolerating_collate,
                                             num_workers=0, worker_init_fn=lambda id: np.random.seed(np.random.get_state()[1][0] + id))

    states = torch.load(opt.ckpt)
    if "states" in states:
        states = states["states"]
    # states["template_vertices"] = new_cage_shape.transpose(1, 2)
    # states["source_vertices"] = new_source.transpose(1,2)
    # states["source_faces"] = new_source_face
    # new_source_face = states["source_faces"]

    om.write_mesh(os.path.join(opt.log_dir, opt.subdir,
                               "template-Sa.ply"), source_mesh)

    net = networks.FixedSourceDeformer(opt, 3, opt.num_point, bottleneck_size=opt.bottleneck_size,
                                       template_vertices=states["template_vertices"], template_faces=states["template_faces"].cuda(),
                                       source_vertices=states["source_vertices"], source_faces=states["source_faces"]).cuda()
    load_network(net, states)

    source_points = torch.from_numpy(
        source_mesh.points().copy()).float().cuda().unsqueeze(0)
    with torch.no_grad():
        # source_face = net.source_faces.detach()
        for i, data in enumerate(dataloader):
            data = dataset.uncollate(data)

            target_shape, target_filename = data["target_shape"], data["target_file"]
            logger.info("", data["target_file"][0])

            sample_idx = None
            if "sample_idx" in data:
                sample_idx = data["sample_idx"]

            outputs = net(target_shape.transpose(1, 2), cage_only=True)
            if opt.d_residual:
                cage_offset = outputs["new_cage"]-outputs["cage"]
                outputs["cage"] = new_cage_shape
                outputs["new_cage"] = new_cage_shape+cage_offset

            deformed = deform_with_MVC(outputs["cage"], outputs["new_cage"], outputs["cage_face"].expand(
                outputs["cage"].shape[0], -1, -1), source_points)

            for b in range(deformed.shape[0]):
                t_filename = os.path.splitext(target_filename[b])[0]
                source_mesh_arr = source_mesh.points()
                source_mesh_arr[:] = deformed[0].cpu().detach().numpy()
                om.write_mesh(os.path.join(
                    opt.log_dir, opt.subdir, "template-{}-Sab.obj".format(t_filename)), source_mesh)
                # if data["target_face"] is not None and data["target_mesh"] is not None:
                # pymesh.save_mesh_raw(os.path.join(opt.log_dir, opt.subdir, "template-{}-Sa.ply".format(t_filename)),
                #             source_mesh[0].detach().cpu(), source_face[b].detach().cpu())
                pymesh.save_mesh_raw(os.path.join(opt.log_dir, opt.subdir, "template-{}-Sb.ply".format(t_filename)),
                                     data["target_mesh"][b].detach().cpu(), data["target_face"][b].detach().cpu())
                # pymesh.save_mesh_raw(os.path.join(opt.log_dir, opt.subdir, "template-{}-Sab.ply".format(t_filename)),
                #             deformed[b].detach().cpu(), source_face[b].detach().cpu())

                # else:
                #     save_ply(source_mesh[0].detach().cpu(), os.path.join(opt.log_dir, opt.subdir,"template-{}-Sa.ply".format(t_filename)))
                #     save_ply(target_shape[b].detach().cpu(), os.path.join(opt.log_dir, opt.subdir,"template-{}-Sb.ply".format(t_filename)),
                #                 normals=data["target_normals"][b].detach().cpu())
                #     save_ply(deformed[b].detach().cpu(), os.path.join(opt.log_dir, opt.subdir,"template-{}-Sab.ply".format(t_filename)),
                #                 normals=data["target_normals"][b].detach().cpu())

                pymesh.save_mesh_raw(
                    os.path.join(opt.log_dir, opt.subdir, "template-{}-cage1.ply".format(t_filename)),
                    outputs["cage"][b].detach().cpu(), outputs["cage_face"][b].detach().cpu(),
                                   )
                pymesh.save_mesh_raw(
                    os.path.join(opt.log_dir, opt.subdir, "template-{}-cage2.ply".format(t_filename)),
                    outputs["new_cage"][b].detach().cpu(), outputs["cage_face"][b].detach().cpu(),
                                   )

            # if opt.opt_lap and deformed.shape[1] == source_mesh.shape[1]:
            #     deformed = optimize_lap(opt, source_mesh, deformed, source_face)
            #     for b in range(deformed.shape[0]):
            #         pymesh.save_mesh_raw(os.path.join(opt.log_dir, opt.subdir, "template-{}-Sab-optlap.ply".format(t_filename)),
            #                                 deformed[b].detach().cpu(), source_face[b].detach().cpu())

            if i % 20 == 0:
                logger.success("[{}/{}] Done".format(i, len(dataloader)))

    dataset.render_result(os.path.join(opt.log_dir, opt.subdir))


def optimize_lap(opt, source_shape, deformed_shape, face):
    """
    source_shape (B,N,3)
    deformed_shape (B,N,3)
    face         (B,F,3)
    """
    B = deformed_shape.shape[0]
    if opt.corres_idx is None:
        n_selected = int(source_shape.shape[1] * 0.6)
        corresp_v = torch.unique(torch.randint(
            0, source_shape.shape[1], (n_selected, ))).view(1, -1, 1).cuda()
    else:
        corresp_idx = torch.from_numpy(np.loadtxt(
            opt.corres_idx, delimiter=",", dtype=np.int64)).cuda()
        _, corresp_idx_2 = torch.unbind(corresp_idx, dim=1)
        corresp_v = torch.unique(torch.gather(
            face, 1, corresp_idx_2.view(1, -1, 1).expand(-1, -1, 3))).view(1, -1, 1)

    fixed_points = torch.gather(
        deformed_shape, 1, corresp_v.expand(-1, -1, 3)).detach()

    deformed_shape = deformed_shape.detach()
    deformed_shape.requires_grad_(True)
    source_shape = source_shape.detach()
    optimizer = torch.optim.Adam([deformed_shape], lr=0.0005, betas=(0.1, 0.1))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, min_lr=1e-6, verbose=True, patience=25)
    lap_loss = MeshLaplacianLoss(torch.nn.MSELoss(reduction="none"), use_cot=True,
                                 use_norm=True, consistent_topology=True, precompute_L=True)

    for t in range(2000):
        lap_loss_value = torch.mean(
            lap_loss(source_shape, deformed_shape, face=face).view(B, -1))
        fixed_points_new = torch.gather(
            deformed_shape, 1, corresp_v.expand(-1, -1, 3))
        reg_value = torch.mean(
            torch.sum((fixed_points - fixed_points_new)**2, dim=-1))
        loss = lap_loss_value + reg_value
        loss.backward()
        if (t+1) % 50 == 0:
            print("t {}/{} lap: {} reg: {}".format(t, 2000,
                                                   lap_loss_value.item(), reg_value.item()))
        if loss < 1e-8:
            logger.success("Optimization converged!")
            break
        optimizer.step()
        scheduler.step(loss.item())

    return deformed_shape


if __name__ == "__main__":
    parser = MyOptions()
    opt = parser.parse()

    opt.log_dir = os.path.dirname(opt.ckpt)

    os.makedirs(os.path.join(opt.log_dir, opt.subdir), exist_ok=True)
    if opt.use_cage is None:
        # optimize initial cage for the new target
        cage_v, cage_f = optimize(opt)
        pymesh.save_mesh_raw(os.path.join(opt.log_dir, opt.subdir, "optimized_template_cage.ply"),
                             cage_v[0].detach().cpu(), cage_f[0].detach().cpu())
    else:
        cage_v, cage_f = read_trimesh(opt.use_cage)
        cage_v = torch.from_numpy(cage_v[:, :3].astype(np.float32)).cuda()
        cage_f = torch.from_numpy(cage_f[:, :3].astype(np.int64)).cuda()
        cage_v.unsqueeze_(0)
        cage_f.unsqueeze_(0)

    # # test using the new source and initial cage
    # target_shape_pose, target_face_pose, _ = read_trimesh("/home/mnt/points/data/MPI-FAUST/training/registrations/tr_reg_002.ply")
    # target_shape_pose = torch.from_numpy(target_shape_pose[:,:3].astype(np.float32)).cuda()
    # target_face_pose = torch.from_numpy(target_face_pose[:,:3].astype(np.int64)).cuda()
    # target_shape_pose, _, _ = center_bounding_box(target_shape_pose)
    # target_shape_pose.unsqueeze_(0)
    # target_face_pose.unsqueeze_(0)
    # test_one(opt, cage_v, target_shape, target_face, target_shape_pose, target_face_pose)
    test_all(opt, cage_v)
