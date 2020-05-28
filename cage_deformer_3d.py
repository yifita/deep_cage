from __future__ import print_function
from pprint import pprint
import traceback
import sys
import shutil
import openmesh as om
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import numpy as np
import pymesh
from pytorch_points.misc import logger
from pytorch_points.network.geo_operations import mean_value_coordinates_3D, edge_vertex_indices
from pytorch_points.utils.pc_utils import load, save_ply, save_pts, center_bounding_box
from pytorch_points.utils.geometry_utils import read_trimesh, write_trimesh, build_gemm, Mesh, get_edge_points, generatePolygon
from pytorch_points.utils.pytorch_utils import weights_init, check_values, save_network, load_network, save_grad, saved_variables, \
    clamp_gradient_norm, linear_loss_weight, tolerating_collate, clamp_gradient, fix_network_parameters
import losses
import networks
from common import loadInitCage, build_dataset, crisscross_input, log_outputs, deform_with_MVC


def test(net=None, save_subdir="test"):
    opt.phase = "test"
    dataset = build_dataset(opt)

    if opt.dim == 3:
        init_cage_V, init_cage_Fs = loadInitCage([opt.template])
        cage_V_t = init_cage_V.transpose(1,2).detach().cuda()
    else:
        init_cage_V = generatePolygon(0, 0, 1.5, 0, 0, 0, opt.cage_deg)
        init_cage_V = torch.tensor([(x, y) for x, y in init_cage_V], dtype=torch.float).unsqueeze(0)
        cage_V_t = init_cage_V.transpose(1,2).detach().cuda()
        init_cage_Fs = [torch.arange(opt.cage_deg, dtype=torch.int64).view(1,1,-1).cuda()]

    if net is None:
        # network
        net = networks.NetworkFull(opt, dim=opt.dim, bottleneck_size=opt.bottleneck_size,
                               template_vertices=cage_V_t, template_faces=init_cage_Fs[-1],
                               ).cuda()
        net.eval()
        load_network(net, opt.ckpt)
    else:
        net.eval()

    print(net)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False,
                                             collate_fn=tolerating_collate,
                                             num_workers=0,
                                             worker_init_fn=lambda id: np.random.seed(np.random.get_state()[1][0] + id))

    test_output_dir = os.path.join(opt.log_dir, save_subdir)
    os.makedirs(test_output_dir, exist_ok=True)
    with open(os.path.join(test_output_dir, "eval.txt"), "w") as f:
        with torch.no_grad():
            for i, data in enumerate(dataloader):
                data = dataset.uncollate(data)

                ############# blending ############
                # sample 4 different alpha
                if opt.blend_style:
                    num_alpha = 4
                    blend_alpha = torch.linspace(0, 1, steps=num_alpha, dtype=torch.float32).cuda().reshape(num_alpha, 1)
                    data["source_shape"] = data["source_shape"].expand(num_alpha, -1, -1).contiguous()
                    data["target_shape"] = data["target_shape"].expand(num_alpha, -1, -1).contiguous()
                else:
                    blend_alpha = 1.0

                data["alpha"] = blend_alpha

                ###################################
                source_shape_t = data["source_shape"].transpose(1,2).contiguous().detach()
                target_shape_t = data["target_shape"].transpose(1,2).contiguous().detach()

                outputs = net(source_shape_t, target_shape_t, blend_alpha)
                deformed = outputs["deformed"]


                ####################### evaluation ########################
                s_filename = os.path.splitext(data["source_file"][0])[0]
                t_filename = os.path.splitext(data["target_file"][0])[0]

                log_str = "{}/{} {}-{} ".format(i, len(dataloader), s_filename, t_filename)
                print(log_str)
                f.write(log_str+"\n")

                ###################### outputs ############################
                for b in range(deformed.shape[0]):
                    if "source_mesh" in data and data["source_mesh"] is not None:
                        if isinstance(data["source_mesh"][0], str):
                            source_mesh = om.read_polymesh(data["source_mesh"][0]).points().copy()
                            source_mesh = dataset.normalize(source_mesh, opt.isV2)
                            source_mesh = torch.from_numpy(source_mesh.astype(np.float32)).unsqueeze(0).cuda()
                            deformed = deform_with_MVC(outputs["cage"][b:b+1], outputs["new_cage"][b:b+1],
                                outputs["cage_face"], source_mesh)
                        else:
                            deformed = deform_with_MVC(outputs["cage"][b:b+1], outputs["new_cage"][b:b+1],
                                outputs["cage_face"], data["source_mesh"])

                    deformed[b] = center_bounding_box(deformed[b])[0]
                    if data["source_face"] is not None and data["source_mesh"] is not None:
                        source_mesh = data["source_mesh"][0].detach().cpu()
                        source_mesh = center_bounding_box(source_mesh)[0]
                        source_face = data["source_face"][0].detach().cpu()
                        tosave = pymesh.form_mesh(vertices=source_mesh, faces=source_face)
                        pymesh.save_mesh(os.path.join(opt.log_dir, save_subdir, "{}-{}-Sa.obj".format(s_filename, t_filename)),
                                         tosave, use_float=True
                                      )
                        tosave = pymesh.form_mesh(vertices=deformed[0].detach().cpu(), faces=source_face)
                        pymesh.save_mesh(os.path.join(opt.log_dir, save_subdir, "{}-{}-Sab-{}.obj".format(s_filename, t_filename, b)),
                                         tosave, use_float=True,
                                         )
                    elif data["source_face"] is None and isinstance(data["source_mesh"][0], str):
                        orig_file_path = data["source_mesh"][0]
                        mesh = om.read_polymesh(orig_file_path)
                        points_arr = mesh.points()
                        points_arr[:] = source_mesh[0].detach().cpu().numpy()
                        om.write_mesh(os.path.join(opt.log_dir, save_subdir, "{}-{}-Sa.obj".format(s_filename, t_filename)), mesh)
                        points_arr[:] = deformed[0].detach().cpu().numpy()
                        om.write_mesh(os.path.join(opt.log_dir, save_subdir, "{}-{}-Sab-{}.obj".format(s_filename, t_filename, b)), mesh)
                    else:
                        # save to "pts" for rendering
                        save_pts(os.path.join(opt.log_dir, save_subdir,"{}-{}-Sa.pts".format(s_filename,t_filename)), data["source_shape"][b].detach().cpu())
                        save_pts(os.path.join(opt.log_dir, save_subdir,"{}-{}-Sab-{}.pts".format(s_filename,t_filename, b)), deformed[0].detach().cpu())

                    if data["target_face"] is not None and data["target_mesh"] is not None:
                        data["target_mesh"][0] = center_bounding_box(data["target_mesh"][0])[0]
                        tosave = pymesh.form_mesh(vertices=data["target_mesh"][0].detach().cpu(), faces=data["target_face"][0].detach().cpu())
                        pymesh.save_mesh(os.path.join(opt.log_dir, save_subdir, "{}-{}-Sb.obj".format(s_filename, t_filename)),
                                    tosave, use_float=True,
                                    )
                    elif data["target_face"] is None and isinstance(data["target_mesh"][0], str):
                        orig_file_path = data["target_mesh"][0]
                        mesh = om.read_polymesh(orig_file_path)
                        points_arr = mesh.points()
                        points_arr[:] = dataset.normalize(points_arr.copy(), opt.isV2)
                        om.write_mesh(os.path.join(opt.log_dir, save_subdir, "{}-{}-Sb.obj".format(s_filename, t_filename)), mesh)
                    else:
                        save_pts(os.path.join(opt.log_dir, save_subdir,"{}-{}-Sb.pts".format(s_filename,t_filename)), data["target_shape"][0].detach().cpu())

                    outputs["cage"][b] = center_bounding_box(outputs["cage"][b])[0]
                    outputs["new_cage"][b] = center_bounding_box(outputs["new_cage"][b])[0]
                    pymesh.save_mesh_raw(
                        os.path.join(opt.log_dir, save_subdir, "{}-{}-cage1-{}.ply".format(s_filename, t_filename, b)),
                        outputs["cage"][b].detach().cpu(), outputs["cage_face"][0].detach().cpu(), binary=True)
                    pymesh.save_mesh_raw(
                        os.path.join(opt.log_dir, save_subdir, "{}-{}-cage2-{}.ply".format(s_filename, t_filename, b)),
                        outputs["new_cage"][b].detach().cpu(), outputs["cage_face"][0].detach().cpu(), binary=True)

    dataset.render_result(test_output_dir)


def train():
    dataset = build_dataset(opt)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, drop_last=True,
                                             collate_fn=tolerating_collate,
                                             num_workers=2, worker_init_fn=lambda id: np.random.seed(np.random.get_state()[1][0] + id))

    if opt.dim == 3:
        # cage (1,N,3)
        init_cage_V, init_cage_Fs = loadInitCage([opt.template])
        cage_V_t = init_cage_V.transpose(1,2).detach().cuda()
        cage_edge_points_list = []
        cage_edges_list = []
        for F in init_cage_Fs:
            mesh = Mesh(vertices=init_cage_V[0], faces=F[0])
            build_gemm(mesh, F[0])
            cage_edge_points = torch.from_numpy(get_edge_points(mesh)).cuda()
            cage_edge_points_list.append(cage_edge_points)
            cage_edges_list = [edge_vertex_indices(F[0])]
    else:
        init_cage_V = generatePolygon(0, 0, 1.5, 0, 0, 0, opt.cage_deg)
        init_cage_V = torch.tensor([(x, y) for x, y in init_cage_V], dtype=torch.float).unsqueeze(0)
        cage_V_t = init_cage_V.transpose(1,2).detach().cuda()
        init_cage_Fs = [torch.arange(opt.cage_deg, dtype=torch.int64).view(1,1,-1).cuda()]

    # network
    net = networks.NetworkFull(opt, dim=opt.dim, bottleneck_size=opt.bottleneck_size,
                               template_vertices=cage_V_t, template_faces=init_cage_Fs[-1],
                               ).cuda()

    net.apply(weights_init)
    if opt.ckpt:
        load_network(net, opt.ckpt)

    all_losses = losses.AllLosses(opt)
    # optimizer
    optimizer = torch.optim.Adam([
        {"params": net.encoder.parameters()},
        {"params": net.nd_decoder.parameters()},
        {"params": net.merger.parameters()}], lr=opt.lr)

    if opt.full_net:
        optimizer.add_param_group({'params': net.nc_decoder.parameters(), 'lr': 0.1*opt.lr})
    if opt.optimize_template:
        optimizer.add_param_group({'params': net.template_vertices, 'lr': opt.lr})

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, int(opt.nepochs*0.4), gamma=0.1, last_epoch=-1)

    # train
    net.train()
    start_epoch = 0
    t = 0

    steps_C = 20
    steps_D = 20

    # train
    os.makedirs(opt.log_dir, exist_ok=True)
    shutil.copy2(__file__, opt.log_dir)
    shutil.copy2(os.path.join(os.path.dirname(__file__), "networks.py"), opt.log_dir)
    shutil.copy2(os.path.join(os.path.dirname(__file__), "losses.py"), opt.log_dir)
    shutil.copy2(os.path.join(os.path.dirname(__file__), "datasets.py"), opt.log_dir)
    shutil.copy2(os.path.join(os.path.dirname(__file__), "common.py"), opt.log_dir)
    shutil.copy2(os.path.join(os.path.dirname(__file__), "option.py"), opt.log_dir)
    print(net)

    log_file = open(os.path.join(opt.log_dir, "training_log.txt"), "a")
    log_file.write(str(net)+"\n")

    log_interval = max(len(dataloader)//5, 50)
    save_interval = max(opt.nepochs//10, 1)
    running_avg_loss = -1

    with torch.autograd.detect_anomaly():
        if opt.epoch:
            start_epoch = opt.epoch % opt.nepochs
            t += start_epoch*len(dataloader)

        for epoch in range(start_epoch, opt.nepochs):
            for t_epoch, data in enumerate(dataloader):
                warming_up = epoch < opt.warmup_epochs
                progress = t_epoch/len(dataloader)+epoch
                optimize_C = (t % (steps_C+steps_D)) > steps_D

                ############# get data ###########
                data = dataset.uncollate(data)
                data = crisscross_input(data)
                if opt.dim == 3:
                    data["cage_edge_points"] = cage_edge_points_list[-1]
                    data["cage_edges"] = cage_edges_list[-1]
                source_shape, target_shape = data["source_shape"], data["target_shape"]

                ############# blending ############
                if opt.blend_style:
                    blend_alpha = torch.rand((source_shape.shape[0], 1), dtype=torch.float32).to(device=source_shape.device)
                else:
                    blend_alpha = 1.0
                data["alpha"] = blend_alpha

                ############# run network ###########
                optimizer.zero_grad()
                # optimizer_C.zero_grad()
                # optimizer_D.zero_grad()
                source_shape_t = source_shape.transpose(1,2)
                target_shape_t = target_shape.transpose(1,2)
                outputs = net(source_shape_t, target_shape_t, data["alpha"])

                ############# get losses ###########
                current_loss = all_losses(data, outputs, progress)
                loss_sum = torch.sum(torch.stack([v for v in current_loss.values()], dim=0))
                if running_avg_loss < 0:
                    running_avg_loss = loss_sum
                else:
                    running_avg_loss = running_avg_loss + (loss_sum.item() - running_avg_loss)/(t+1)

                if (t % log_interval == 0) or (loss_sum > 5*running_avg_loss):
                    log_str = "warming up {} e {:03d} t {:05d}: {}".format(warming_up, epoch, t,
                                                                           ", ".join(["{} {:.3g}".format(k, v.mean().item()) for k, v in current_loss.items()]))
                    print(log_str)
                    log_file.write(log_str+"\n")
                    log_outputs(opt, t, outputs, data)

                if loss_sum > 100*running_avg_loss:
                    logger.info("loss ({}) > 5*running_average_loss ({}). Skip without update.".format(loss_sum, 5*running_avg_loss))
                    torch.cuda.empty_cache()
                    continue

                loss_sum.backward()
                if epoch < opt.warmup_epochs:
                    try:
                        net.nc_decoder.zero_grad()
                        net.encoder.zero_grad()
                    except AttributeError:
                        net.template_vertices.grad.zero_()

                if opt.alternate_cd:
                    optimize_C = (epoch > opt.warmup_epochs) and (epoch % (opt.c_epoch+opt.d_epoch)) > opt.d_epoch
                    if optimize_C:
                        net.nd_decoder.zero_grad()
                    else:
                        try:
                            net.encoder.zero_grad()
                            net.nc_decoder.zero_grad()
                        except AttributeError:
                            net.template_vertices.grad.zero_()

                clamp_gradient(net, 0.1)
                optimizer.step()

                if (t + 1) % 500 == 0:
                    save_network(net, opt.log_dir, network_label="net", epoch_label="latest")

                t += 1

            if (epoch + 1) % save_interval == 0:
                save_network(net, opt.log_dir, network_label="net", epoch_label=epoch)

            scheduler.step()
            if opt.eval:
                try:
                    test(net=net, save_subdir="epoch_{}".format(epoch))
                except Exception as e:
                    traceback.print_exc(file=sys.stdout)
                    logger.warn("Failed to run test", str(e))

    log_file.close()
    save_network(net, opt.log_dir, network_label="net", epoch_label="final")
    test(net=net)


if __name__ == "__main__":
    from option import BaseOptions
    import datetime
    import os
    parser = BaseOptions()
    opt = parser.parse()

    # reproducability
    torch.manual_seed(24)
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = False  # type: ignore
    np.random.seed(24)

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
        test(save_subdir=opt.subdir)
    else:
        os.makedirs(opt.log_dir, exist_ok=True)
        log_file = open(os.path.join(opt.log_dir, "training_log.txt"), "a")
        parser.print_options(opt, log_file)
        log_file.close()
        train()