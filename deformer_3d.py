from __future__ import print_function
from pprint import pprint
import shutil
import datetime
from glob import glob
import torch
import torch.nn as nn
import torch.utils.data
import numpy as np
import pymesh
from pytorch_points.network.geo_operations import mean_value_coordinates_3D, edge_vertex_indices, furthest_point_sample
from pytorch_points.misc import logger
from pytorch_points.utils.geometry_utils import build_gemm, Mesh, get_edge_points
from pytorch_points.utils.pc_utils import load, save_ply, center_bounding_box, read_ply, normalize_to_box
from pytorch_points.utils.pytorch_utils import weights_init, check_values, save_network, load_network, save_grad, saved_variables, \
                                               clamp_gradient_norm, linear_loss_weight, tolerating_collate, clamp_gradient
import os
import losses
from datasets import FaustDataset, PairedSurreal
from common import build_dataset, log_outputs, deform_with_MVC, read_trimesh
import network2 as networks

def test(net=None, subdir="test"):
    opt.phase = "test"
    if isinstance(opt.target_model, str):
        opt.target_model = [opt.target_model]

    if net is None:
        states = torch.load(opt.ckpt)
        if "states" in states:
            states = states["states"]
        if opt.template:
            cage_shape, cage_face = read_trimesh(opt.template)
            cage_shape = torch.from_numpy(cage_shape[:,:3]).unsqueeze(0).float()
            cage_face = torch.from_numpy(cage_face).unsqueeze(0).long()
            states["template_vertices"] = cage_shape.transpose(1,2)
            states["template_faces"] = cage_face

        if opt.source_model:
            source_shape, source_face = read_trimesh(opt.source_model)
            source_shape = torch.from_numpy(source_shape[:,:3]).unsqueeze(0).float()
            source_face = torch.from_numpy(source_face).unsqueeze(0).long()
            states["source_vertices"] = source_shape.transpose(1,2)
            states["source_faces"] = source_shape

        net = networks.FixedSourceDeformer(opt, 3, opt.num_point, bottleneck_size=opt.bottleneck_size,
                                           template_vertices=states["template_vertices"], template_faces=states["template_faces"],
                                           source_vertices=states["source_vertices"], source_faces=states["source_faces"]).cuda()

        load_network(net, states)
        net = net.cuda()
        net.eval()
    else:
        net.eval()

    print(net)

    test_output_dir = os.path.join(opt.log_dir, subdir)
    os.makedirs(test_output_dir, exist_ok=True)
    with torch.no_grad():
        for target_model in opt.target_model:
            assert(os.path.isfile(target_model))
            target_face = None
            target_shape, target_face = read_trimesh(target_model)
            # target_shape = read_ply(target_model)[:,:3]
            # target_shape, _, scale = normalize_to_box(target_shape)
            # normalize acording to height y axis
            # target_shape = target_shape/2*1.7
            target_shape = torch.from_numpy(target_shape[:,:3]).cuda().float().unsqueeze(0)
            if target_face is None:
                target_face = net.source_faces
            else:
                target_face = torch.from_numpy(target_face).cuda().long().unsqueeze(0)
            t_filename = os.path.splitext(os.path.basename(target_model))[0]

            source_mesh = net.source_vertices.transpose(1,2).detach()
            source_face = net.source_faces.detach()

            # furthest sampling
            target_shape_sampled = furthest_point_sample(target_shape, net.source_vertices.shape[2], NCHW=False)[1]
            # target_shape_sampled = (target_shape[:, np.random.permutation(target_shape.shape[1]), :]).contiguous()
            outputs = net(target_shape_sampled.transpose(1,2), None, cage_only=True)
            # deformed = outputs["deformed"]

            deformed = deform_with_MVC(outputs["cage"], outputs["new_cage"],
                                        outputs["cage_face"].expand(outputs["cage"].shape[0], -1, -1),
                                        source_mesh)

            b = 0

            save_ply(target_shape_sampled[b].cpu().numpy(), os.path.join(opt.log_dir, subdir, "template-{}-Sb.pts".format(t_filename)))
            pymesh.save_mesh_raw(os.path.join(opt.log_dir, subdir, "template-{}-Sa.ply".format(t_filename)),
                        source_mesh[0].detach().cpu(), source_face[0].detach().cpu())
            pymesh.save_mesh_raw(os.path.join(opt.log_dir, subdir, "template-{}-Sb.ply".format(t_filename)),
                        target_shape[b].detach().cpu(), target_face[b].detach().cpu())
            pymesh.save_mesh_raw(os.path.join(opt.log_dir, subdir, "template-{}-Sab.ply".format(t_filename)),
                        deformed[b].detach().cpu(), source_face[b].detach().cpu())

            pymesh.save_mesh_raw(os.path.join(opt.log_dir, subdir, "template-{}-cage1.ply".format(t_filename)),
                outputs["cage"][b].detach().cpu(), outputs["cage_face"][b].detach().cpu())
            pymesh.save_mesh_raw(os.path.join(opt.log_dir, subdir, "template-{}-cage2.ply".format(t_filename)),
                outputs["new_cage"][b].detach().cpu(), outputs["cage_face"][b].detach().cpu())


    PairedSurreal.render_result(test_output_dir)


def test_all(net=None, subdir="test"):
    opt.phase = "test"
    dataset = build_dataset(opt)

    if net is None:
        source_shape = dataset.mesh_vertex.unsqueeze(0).to(dtype=torch.float)
        source_face = dataset.mesh_face.unsqueeze(0)
        cage_shape = dataset.cage_vertex.unsqueeze(0).to(dtype=torch.float)
        cage_face = dataset.cage_face.unsqueeze(0)
        net = networks.FixedSourceDeformer(opt, 3, opt.num_point, bottleneck_size=opt.bottleneck_size,
                                           template_vertices=cage_shape.transpose(1,2), template_faces=cage_face,
                                           source_vertices=source_shape.transpose(1,2), source_faces=source_face).cuda()

        load_network(net, opt.ckpt)
        net.eval()
    else:
        net.eval()

    print(net)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False,
                                            num_workers=3, worker_init_fn=lambda id: np.random.seed(np.random.get_state()[1][0] + id))

    chamfer_distance = losses.LabeledChamferDistance(beta=0, gamma=1)
    mse_distance = torch.nn.MSELoss()
    avg_CD = 0
    avg_EMD = 0
    test_output_dir = os.path.join(opt.log_dir, subdir)
    os.makedirs(test_output_dir, exist_ok=True)
    with open(os.path.join(test_output_dir, "eval.txt"), "w") as f:
        with torch.no_grad():
            source_mesh = net.source_vertices.transpose(1,2).detach()
            source_face = net.source_faces.detach()
            for i, data in enumerate(dataloader):
                data = dataset.uncollate(data)

                target_shape, target_filename = data["target_shape"], data["target_file"]

                sample_idx = None
                if "sample_idx" in data:
                    sample_idx = data["sample_idx"]
                outputs = net(target_shape.transpose(1,2), sample_idx)
                deformed = outputs["deformed"]

                deformed = deform_with_MVC(outputs["cage"], outputs["new_cage"],
                                            outputs["cage_face"].expand(outputs["cage"].shape[0], -1, -1),
                                            source_mesh)

                for b in range(outputs["deformed"].shape[0]):
                    t_filename = os.path.splitext(target_filename[b])[0]
                    target_shape_np = target_shape.detach().cpu()[b].numpy()
                    if data["target_face"] is not None and data["target_mesh"] is not None:
                        pymesh.save_mesh_raw(os.path.join(opt.log_dir, subdir, "template-{}-Sa.ply".format(t_filename)),
                                    source_mesh[0].detach().cpu(), source_face[0].detach().cpu())
                        pymesh.save_mesh_raw(os.path.join(opt.log_dir, subdir, "template-{}-Sb.ply".format(t_filename)),
                                    data["target_mesh"][b].detach().cpu(), data["target_face"][b].detach().cpu())
                        pymesh.save_mesh_raw(os.path.join(opt.log_dir, subdir, "template-{}-Sab.ply".format(t_filename)),
                                    deformed[b].detach().cpu(), source_face[b].detach().cpu())
                    else:
                        save_ply(source_mesh[0].detach().cpu(), os.path.join(opt.log_dir, subdir,"template-{}-Sa.ply".format(t_filename)))
                        save_ply(target_shape[b].detach().cpu(), os.path.join(opt.log_dir, subdir,"template-{}-Sb.ply".format(t_filename)),
                                normals=data["target_normals"][b].detach().cpu())
                        save_ply(deformed[b].detach().cpu(), os.path.join(opt.log_dir, subdir,"template-{}-Sab.ply".format(t_filename)),
                                normals=data["target_normals"][b].detach().cpu())

                    pymesh.save_mesh_raw(os.path.join(opt.log_dir, subdir, "template-{}-cage1.ply".format(t_filename)),
                        outputs["cage"][b].detach().cpu(), outputs["cage_face"][b].detach().cpu())
                    pymesh.save_mesh_raw(os.path.join(opt.log_dir, subdir, "template-{}-cage2.ply".format(t_filename)),
                        outputs["new_cage"][b].detach().cpu(), outputs["cage_face"][b].detach().cpu())


                    log_str = "{}/{} {}".format(i, len(dataloader), t_filename)
                    print(log_str)
                    f.write(log_str+"\n")

    dataset.render_result(test_output_dir)


def train():
    dataset = build_dataset(opt)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, drop_last=True,
                                             num_workers=0, worker_init_fn=lambda id: np.random.seed(np.random.get_state()[1][0] + id))
    source_shape = dataset.mesh_vertex.unsqueeze(0).to(dtype=torch.float)
    source_face = dataset.mesh_face.unsqueeze(0)
    cage_shape = dataset.cage_vertex.unsqueeze(0).to(dtype=torch.float)
    cage_face = dataset.cage_face.unsqueeze(0)
    mesh = Mesh(vertices=cage_shape[0], faces=cage_face[0])
    build_gemm(mesh, cage_face[0])
    cage_edge_points = torch.from_numpy(get_edge_points(mesh)).cuda()
    cage_edges = edge_vertex_indices(cage_face[0])

    # network
    net = networks.FixedSourceDeformer(opt, 3, opt.num_point, bottleneck_size=opt.bottleneck_size,
                                       template_vertices=cage_shape.transpose(1,2), template_faces=cage_face,
                                       source_vertices=source_shape.transpose(1,2), source_faces=source_face).cuda()
    print(net)
    net.apply(weights_init)
    if opt.ckpt:
        load_network(net, opt.ckpt)
    net.train()

    all_losses = losses.AllLosses(opt)

    # optimizer
    optimizer = torch.optim.Adam([
        {'params': net.nd_decoder.parameters()},
        {"params": net.encoder.parameters()}],lr=opt.lr)


    # train
    os.makedirs(opt.log_dir, exist_ok=True)
    shutil.copy2(__file__, opt.log_dir)
    shutil.copy2(os.path.join(os.path.dirname(__file__), "network2.py"), opt.log_dir)
    shutil.copy2(os.path.join(os.path.dirname(__file__), "common.py"), opt.log_dir)
    shutil.copy2(os.path.join(os.path.dirname(__file__), "losses.py"), opt.log_dir)
    shutil.copy2(os.path.join(os.path.dirname(__file__), "datasets.py"), opt.log_dir)
    pymesh.save_mesh_raw(os.path.join(opt.log_dir,"t{:06d}_Sa.ply".format(0)),
                  net.source_vertices[0].transpose(0,1).detach().cpu().numpy(),
                  net.source_faces[0].detach().cpu())
    pymesh.save_mesh_raw(os.path.join(opt.log_dir,"t{:06d}_template.ply".format(0)),
                  net.template_vertices[0].transpose(0,1).detach().cpu().numpy(),
                  net.template_faces[0].detach().cpu())

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, max(int(opt.nepochs*0.75),1), gamma=0.5, last_epoch=-1)

    # train
    net.train()
    t = 0
    start_epoch = 0
    warmed_up = False
    mvc_weight = opt.mvc_weight
    opt.mvc_weight = 0

    os.makedirs(opt.log_dir, exist_ok=True)
    running_avg_loss = -1
    log_file = open(os.path.join(opt.log_dir, "loss_log.txt"), "a")
    log_interval = min(max(len(dataloader)//5, 50), 200)
    save_interval = max(opt.nepochs//10, 1)

    with torch.autograd.detect_anomaly():
        if opt.epoch:
            start_epoch = opt.epoch % opt.nepochs
            t += start_epoch*len(dataloader)

        for epoch in range(start_epoch, opt.nepochs):
            for epoch_t, data in enumerate(dataloader):
                progress = epoch_t/len(dataloader)+epoch
                warming_up = progress < opt.warmup_epochs
                if (opt.deform_template or opt.optimize_template) and (progress >= opt.warmup_epochs) and (not warmed_up):
                    if opt.deform_template:
                        optimizer.add_param_group({'params': net.nc_decoder.parameters(), 'lr': 0.1*opt.lr})
                    if opt.optimize_template:
                        optimizer.add_param_group({'params': net.template_vertices, 'lr': 0.1*opt.lr})
                    warmed_up = True
                    # start to compute mvc weight
                    opt.mvc_weight = mvc_weight
                    save_network(net, opt.log_dir, network_label="net", epoch_label="warmed_up")

                ############# get data ###########
                data = dataset.uncollate(data)
                data["cage_edge_points"] = cage_edge_points
                data["cage_edges"] = cage_edges
                data["source_shape"] = net.source_vertices.detach()
                data["source_face"] = net.source_faces.detach()

                ############# run network ###########
                optimizer.zero_grad()
                target_shape_t = data["target_shape"].transpose(1,2)
                sample_idx = None

                if "sample_idx" in data:
                    sample_idx = data["sample_idx"]
                    if data["source_normals"] is not None:
                        data["source_normals"] = torch.gather(data["source_normals"], 1, sample_idx.unsqueeze(-1).expand(-1,-1,3))

                outputs = net(target_shape_t, sample_idx)
                if opt.sfnormal_weight > 0 and ("source_mesh" in data and "source_mesh" is not None):
                    if outputs["deformed"].shape[1] == data["source_mesh"].shape[1]:
                        outputs["deformed_hr"] = outputs["deformed"]
                    else:
                        outputs["deformed_hr"] = deform_with_MVC(
                            outputs["cage"].expand(data["source_mesh"].shape[0], -1, -1).detach(),
                            outputs["new_cage"],
                            outputs["cage_face"].expand(data["source_mesh"].shape[0], -1, -1), data["source_mesh"])
                data["source_shape"] = outputs["source_shape"]

                ############# get losses ###########
                current_loss = all_losses(data, outputs, progress)
                loss_sum = torch.sum(torch.stack([v for v in current_loss.values()], dim=0))
                if running_avg_loss < 0:
                    running_avg_loss = loss_sum
                else:
                    running_avg_loss = running_avg_loss + (loss_sum.item() - running_avg_loss)/(t+1)

                if (t % log_interval == 0) or (loss_sum > 10*running_avg_loss):
                    log_str = "warming up {} e {:03d} t {:05d}: {}".format(not warmed_up, epoch, t,
                                                                           ", ".join(["{} {:.3g}".format(k, v.mean().item()) for k, v in current_loss.items()]))
                    print(log_str)
                    log_file.write(log_str+"\n")
                    log_outputs(opt, t, outputs, data)
                    # save_ply(data["target_shape"][0].detach().cpu().numpy(), os.path.join(opt.log_dir,"step-{:06d}-Sb.ply".format(t)))
                    # save_ply(outputs["deformed"][0].detach().cpu().numpy(), os.path.join(opt.log_dir,"step-{:06d}-Sab.ply".format(t)))
                    # write_trimesh(os.path.join(opt.log_dir, "step-{:06d}-cage1.ply".format(t)),
                    #               outputs["cage"][0].detach().cpu(), outputs["cage_face"][0].detach().cpu(), binary=True)
                    # write_trimesh(os.path.join(opt.log_dir, "step-{:06d}-cage2.ply".format(t)),
                    #               outputs["new_cage"][0].detach().cpu(), outputs["cage_face"][0].detach().cpu(), binary=True)

                if loss_sum > 100*running_avg_loss:
                    logger.info("loss ({}) > 10*running_average_loss ({}). Skip without update.".format(loss_sum, 5*running_avg_loss))
                    torch.cuda.empty_cache()
                    continue

                loss_sum.backward()

                if opt.alternate_cd:
                    optimize_C = (progress > opt.warmup_epochs) and (t % (opt.c_step+opt.d_step)) > opt.d_step
                    if optimize_C:
                        net.nd_decoder.zero_grad()
                        net.encoder.zero_grad()
                    else:
                        try:
                            net.nc_decoder.zero_grad()
                        except AttributeError:
                            net.template_vertices.grad.zero_()

                # clamp_gradient_norm(net, 1)
                optimizer.step()
                if (t + 1) % 500 == 0:
                    save_network(net, opt.log_dir, network_label="net", epoch_label="latest")

                t += 1


            if (epoch + 1) % save_interval == 0:
                save_network(net, opt.log_dir, network_label="net", epoch_label=epoch)

            scheduler.step()

    log_file.close()
    save_network(net, opt.log_dir, network_label="net", epoch_label="final")
    test_all(net=net)


if __name__ == "__main__":
    from option import DeformationOptions
    parser = DeformationOptions()
    opt = parser.parse()
    # reproducability
    torch.backends.cudnn.benchmark = False  # type: ignore
    torch.manual_seed(24)
    torch.backends.cudnn.deterministic = True  # type: ignore
    np.random.seed(24)

    torch.cuda.set_device("cuda:0")  # type: ignore
    if opt.phase == "train":
        if opt.ckpt is not None:
            opt.log_dir = os.path.dirname(opt.ckpt)
        else:
            opt.log_dir = os.path.join(opt.log_dir, "-".join(filter(None, [os.path.basename(__file__)[:-3],
                                                                        datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S"),
                                                                        opt.name])))
    else:
        opt.log_dir = os.path.dirname(opt.ckpt)

    if opt.phase == "test":
        if opt.target_model is None:
            test_all(subdir=opt.subdir)
        else:
            test(subdir=opt.subdir)
    else:
        os.makedirs(opt.log_dir, exist_ok=True)
        log_file = open(os.path.join(opt.log_dir, "loss_log.txt"), "a")
        parser.print_options(opt, log_file)
        log_file.close()
        train()
