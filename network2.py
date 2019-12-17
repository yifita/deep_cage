from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
from pytorch_points.misc import logger
from pytorch_points.network.layers import Conv1d, Linear
from pytorch_points.network.pointnet2_modules import PointnetSAModuleMSG
from pytorch_points.network.operations import scatter_add, faiss_knn
from pytorch_points.network.geo_operations import (mean_value_coordinates,
                                                   mean_value_coordinates_3D,
                                                   compute_face_normals_and_areas)

from pytorch_points.utils.geometry_utils import get_edge_points
from pytorch_points.utils.pc_utils import random_sphere
from pytorch_points.utils.pytorch_utils import linear_loss_weight, check_values, save_grad
from networks import PointNetfeat, MultiFoldPointGen, PointNet2feat
from common import deform_with_MVC

def xyz_to_polar(sphere_points):
    """
    (B,3,N) -> theta, phi (B,2,N), r (B)
    x = r*cos(theta)*sin(phi)
    y = r*sin(theta)*sin(phi)
    z = r*cos(phi)
    """
    r = torch.sqrt(torch.sum(sphere_points*sphere_points, dim=1))
    theta = torch.atan2(sphere_points[:,1,:], sphere_points[:,0,:])
    z = sphere_points[:,2,:]/r
    z.clamp_(-1.0+1e-5, 1.0-1e-5)
    phi = torch.acos(z)
    phi = phi.masked_fill(z==1, 0.0)
    r = torch.mean(r, dim=-1)
    assert(check_values(phi))
    assert(check_values(theta))
    return torch.stack([theta,phi], dim=1), r

def angle_to_xyz(sphere_angles, r):
    """
    (B,3,N) (B,) -> (B,2,N)
    x = r*cos(theta)*sin(phi)
    y = r*sin(theta)*sin(phi)
    z = r*cos(phi)
    """
    r = r.view(-1,1)
    theta = sphere_angles[:,0,:]
    phi = sphere_angles[:,1,:]
    x = torch.sin(phi)*torch.cos(theta)*r
    y = torch.sin(phi)*torch.sin(theta)*r
    z = torch.cos(phi)*r
    return torch.stack([x,y,z], dim=1)


class NetworkFull(nn.Module):
    def __init__(self, opt, dim, num_points, bottleneck_size,
                 template_vertices, template_faces,
                 **kwargs):
        super().__init__()
        self.opt = opt
        ###### shared encoder ########
        self.encoder = nn.Sequential(
            PointNetfeat(dim=dim, num_points=num_points,bottleneck_size=bottleneck_size),
            Linear(bottleneck_size, bottleneck_size, activation="lrelu", normalization=opt.normalization)
            )

        ###### module to optimize template ########
        self.prepare_sphere_template(template_vertices, template_faces)

        ###### cage prediction and cage deformation ########
        self.nc_decoder = MultiFoldPointGen(bottleneck_size, dim, n_fold=opt.n_fold, normalization=opt.normalization,
                                            concat_prim=opt.concat_prim, residual=opt.c_residual, return_aux=True)
        self.D_use_C_global_code = opt.c_global
        nc_bottleneck = bottleneck_size if opt.c_global else bottleneck_size//2
        self.merger = nn.Sequential(
                Conv1d(bottleneck_size+nc_bottleneck, bottleneck_size+nc_bottleneck, 1, activation="lrelu", normalization=opt.normalization),
            )
        self.nd_decoder = MultiFoldPointGen(bottleneck_size+nc_bottleneck, dim, dim, n_fold=opt.n_fold, normalization=opt.normalization,
                                            concat_prim=opt.concat_prim, return_aux=False, residual=opt.d_residual)

    def prepare_sphere_template(self, template_vertices, template_faces):
        assert(template_vertices.ndim==3 and template_vertices.shape[1]==3) # (1,3,V)
        angle, self.r = xyz_to_polar(template_vertices)

        self.template = nn.Parameter(angle, requires_grad=self.opt.optimize_template)
        assert(template_faces.ndim==3 and template_faces.shape[2]==3) # (1,F,3)
        self.register_buffer("template_faces", template_faces)
        self.register_buffer("template_vertices", template_vertices)
        if self.template.requires_grad:
            logger.info("Enabled vertex optimization")

    def forward(self, source_shape, target_shape, alpha=1.0):
        """
        source_shape (B,3,N)
        target_shape (B,3,M)
        return:
            deformed (B,3,N)
            cage     (B,3,P)
            new_cage (B,3,P)
            weights  (B,N,P)
        """
        B, _, N = source_shape.shape
        _, M, N = target_shape.shape
        _, _, P = self.template.shape

        ############ Encoder #############
        input_shapes = torch.cat([source_shape, target_shape], dim=0)
        shape_code = self.encoder(input_shapes)
        shape_code.unsqueeze_(-1)
        s_code, t_code = torch.split(shape_code, B, dim=0)

        ############ sample on the spherical template  ##############
        sphere_sample = torch.from_numpy(random_sphere(B,N-P)).to(device=source_shape.device, dtype=source_shape.dtype)
        sphere_sample = sphere_sample.transpose(1,2).contiguous()
        template = self.template.view(1,2,-1).expand(B,-1,-1)
        template_v = angle_to_xyz(template, self.r)
        template_v += torch.randn_like(template_v)*0.01
        sphere_sample = torch.cat([template_v, sphere_sample], dim=-1)

        ########### Cage ############
        cage_surface, cage_feat = self.nc_decoder(s_code, sphere_sample)
        cage = cage_surface[:,:,:P]

        ########### Deform ##########
        if not self.D_use_C_global_code:
            s_code = cage_feat
        alpha.unsqueeze_(-1)
        t_code = alpha*t_code + (1-alpha)*s_code
        st_code = torch.cat([s_code, t_code], dim=1)
        # ts_code = torch.cat([t_code, s_code], dim=1)
        # target_code = torch.cat([st_code, ts_code], dim=0)
        target_code = self.merger(st_code)

        deformed_surface = self.nd_decoder(target_code, cage_surface)
        new_cage = deformed_surface[:,:,:P]

        ########### MVC ##############
        deformed_shapes, weights, weights_unnormed = deform_with_MVC(cage.transpose(1,2).contiguous(), new_cage.transpose(1,2).contiguous(),
                self.template_faces.expand(B,-1,-1), source_shape.transpose(1,2).contiguous(), verbose=True)

        ########### prepare for output ##############
        cage = cage.transpose(1,2)
        new_cage = new_cage.transpose(1,2)
        cage_surface = cage_surface.transpose(1,2)
        deformed_surface = deformed_surface.transpose(1,2)
        # cage, t_cage = torch.split(cage, B, dim=0)
        # s_new_cage, t_new_cage = torch.split(new_cage, B, dim=0)
        # cage_surface, t_surface = torch.split(cage_surface, B, dim=0)
        # s_deformed_surface, t_deformed_surface = torch.split(deformed_surface, B, dim=0)
        # deformed, t_deformed = torch.split(deformed_shapes, B, dim=0)

        return {"cage": cage,
                "cage_surface": cage_surface,
                # "t_surface": t_surface,
                "new_cage": new_cage,
                # "t_new_cage": t_new_cage,
                "new_cage_surface": deformed_surface,
                # "t_deformed_surface": t_deformed_surface,
                "deformed": deformed_shapes,
                # "t_deformed": t_deformed,
                "cage_face": self.template_faces,
                "weight": weights,
                "weight_unnormed": weights_unnormed
                }

class DeformationSharedMLP(nn.Module):
    """deformation of a 2D patch into a 3D surface"""

    def __init__(self,dim=3, residual=True, normalization="none"):

        super().__init__()
        layer_size = 128
        self.residual = residual
        self.conv1 = Conv1d(dim, layer_size, 1, activation="lrelu", normalization=normalization)
        self.conv2 = Conv1d(layer_size, layer_size, 1, activation="lrelu", normalization=normalization)
        self.conv3 = Conv1d(layer_size, dim, 1, activation=None)

    def forward(self, x):
        orig = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        if self.residual:
            x += orig
        return x

class MLPDeformer(nn.Module):
    def __init__(self, dim, bottleneck_size, npoint, residual=True, normalization=None):
        super().__init__()
        self.npoint = npoint
        self.dim = dim
        self.residual = residual
        self.layers = nn.Sequential(
                # Linear(bottleneck_size, bottleneck_size, activation="lrelu", normalization=normalization),
                Linear(bottleneck_size, 512, activation="lrelu", normalization=normalization),
                Linear(512,1024, activation="lrelu", normalization=normalization),
                Linear(1024, npoint*dim)
            )
    def forward(self, code, template):
        B, C, N = template.shape
        assert(self.npoint == N)
        assert(self.dim == C)
        if code.ndim > 2:
            code = code.view(B, -1)
        x = self.layers(code)
        x = x.reshape(B,C,N)
        if self.residual:
            x += template
        return x

class FixedSourceDeformer(torch.nn.Module):
    def __init__(self, opt, dim, num_points, bottleneck_size,
                 template_vertices=None, template_faces=None, source_vertices=None, source_faces=None,
                 **kwargs):
        super().__init__()
        self.opt = opt
        self.initialized = False
        self.dim = dim
        ###### shared encoder ########
        if opt.pointnet2:
            self.encoder = PointNet2feat(dim=dim, num_points=opt.num_point, bottleneck_size=bottleneck_size, normalization=opt.normalization)
            bottleneck_size = self.encoder.bottleneck_size
        else:
            self.encoder = nn.Sequential(
                PointNetfeat(dim=dim, num_points=opt.num_point, bottleneck_size=bottleneck_size, normalization=opt.normalization),
                Linear(bottleneck_size, bottleneck_size, activation="tanh", normalization=opt.normalization)
                )

        ###### save template and source to buffer ########
        self.initialize_buffers(template_vertices, template_faces, source_vertices, source_faces)
        self.prob = None
        # print("!!!code_scale", self.code_scale)

        ###### cage refinement and cage deformation ########
        if opt.optimize_template:
            self.template_vertices = nn.Parameter(self.template_vertices)
            logger.info("optimize template cage as parameters")
        if opt.deform_template:
            logger.info("optimize template cage with point fold")
            self.nc_decoder = DeformationSharedMLP(dim, normalization=opt.normalization, residual=opt.c_residual)

        if opt.atlas:
            self.nd_decoder = MultiFoldPointGen((bottleneck_size+dim if opt.use_correspondence else bottleneck_size), dim,
                                                n_fold=opt.n_fold, normalization=opt.normalization,
                                                concat_prim=opt.concat_prim, return_aux=False, residual=opt.d_residual)
        else:
            self.nd_decoder = MLPDeformer(dim=dim, bottleneck_size=bottleneck_size, npoint=self.template_vertices.shape[-1],
                                          residual=opt.d_residual, normalization=opt.normalization)

    def initialize_buffers(self, template_vertices=None, template_faces=None, source_vertices=None, source_faces=None):
        if template_vertices is not None:
            assert(template_vertices.ndim==3 and template_vertices.shape[1]==3) # (1,3,V)
        else:
            template_vertics = torch.zeros((1,self.dim,1), dtype=torch.float32)
        if template_faces is not None:
            assert(template_faces.ndim==3 and template_faces.shape[2]==3) # (1,F,3)
        else:
            template_faces = torch.zeros((1,1,3), dtype=torch.int64)

        if source_vertices is not None:
            assert(source_vertices.ndim==3 and source_vertices.shape[1]==3) # (1,3,V)
        else:
            template_vertics = torch.zeros((1,self.dim,1), dtype=torch.float32)
        if source_faces is not None:
            assert(source_faces.ndim==3 and source_faces.shape[2]==3) # (1,F,3)
        else:
            source_faces = torch.zeros((1,1,3), dtype=torch.int64)
        if not self.initialized:
            self.register_buffer("template_faces", template_faces)
            self.register_buffer("template_vertices", template_vertices)
            self.register_buffer("source_faces", source_faces)
            self.register_buffer("source_vertices", source_vertices)
        else:
            self.template_faces.resize_as_(template_faces).copy_(template_faces)
            self.template_vertices.resize_as_(template_vertices).copy_(template_vertices)
            self.source_faces.resize_as_(source_faces).copy_(source_faces)
            self.source_vertices.resize_as_(source_vertices).copy_(source_vertices)

        self.initialized = True

    def forward(self, target_shape, sample_idx=None, alpha=1.0, cage_only=False):
        """
        source_shape (1,3,M)
        target_shape (B,3,M)
        return:
            deformed (B,3,N)
            cage     (B,3,P)
            new_cage (B,3,P)
            weights  (B,N,P)
        """
        assert(self.initialized)
        B, _, N = target_shape.shape
        _, _, P = self.template_vertices.shape
        ############ Resample source shape #########
        # sample
        if sample_idx is not None:
            source_shape = self.source_vertices.expand(B,-1,-1)
            source_shape = torch.gather(source_shape, -1, sample_idx.unsqueeze(1).expand(-1,3,-1))
        elif self.training and self.source_vertices.shape[-1] != N:
            if self.opt.regular_sampling:
                if self.prob is None:
                    _, farea = compute_face_normals_and_areas(self.source_vertices.transpose(1,2), self.source_faces)
                    v_area = scatter_add(farea.view(-1,1).expand(-1,3).contiguous().view(-1), self.source_faces.view(-1), 0)
                    self.prob = (v_area / torch.sum(v_area)).cpu().numpy()
                random_sample = torch.from_numpy(np.random.choice(self.source_vertices.shape[-1], size=N, p=self.prob).astype(np.int64)).cuda()
            else:
                random_sample = torch.from_numpy(np.random.choice(self.source_vertices.shape[-1], size=N, replace=False).astype(np.int64)).cuda()
            source_shape = torch.index_select(self.source_vertices, -1, random_sample)
            source_shape = source_shape.expand(B,-1,-1)
        else:
            source_shape = self.source_vertices.detach()
            source_shape = source_shape.expand(B,-1,-1)

        ############ Encoder #############
        t_code = self.encoder(target_shape)#*self.code_scale
        t_code = t_code.unsqueeze(-1)

        ############ Cage template ##############
        template_v = self.template_vertices.view(1,3,-1)
        if self.opt.deform_template:
            template_v = self.nc_decoder(template_v)

        ############ Use correspondence #########
        if self.opt.use_correspondence and self.opt.atlas:
            # find the closest cage-to-point vertex
            closest, idx, dist = faiss_knn(3, template_v.expand(B,-1,-1), source_shape, NCHW=True)
            # get the corresponding target xyz idx shape is (B,P,K), target shape is (B,D,N), returns (B,D,P,K)
            target_xyz = torch.gather(target_shape.unsqueeze(2).expand(-1, -1, P, -1), 3, idx.unsqueeze(1).expand(-1,self.opt.dim,-1,-1))
            # (B,D,P)
            target_xyz = torch.median(target_xyz, dim=-1)[0]
            # concatenate to code
            t_code = torch.cat([t_code.expand(-1,-1,P), target_xyz], dim=1).contiguous()

        ########### Deform ##########
        new_cage = self.nd_decoder(t_code, template_v.view(1,3,-1).expand(B,-1,-1))

        ########### MVC ##############
        if not cage_only:
            weights, weights_unnormed = mean_value_coordinates_3D(source_shape.transpose(1,2),
                                                                template_v.expand(B,-1,-1).transpose(1,2),
                                                                self.template_faces.expand(B,-1,-1),
                                                                verbose=True)
            mvc = weights.expand(B,-1,-1)
            deformed_shapes = torch.sum(mvc.unsqueeze(-1)*new_cage.transpose(1,2).unsqueeze(1), dim=2)
        else:
            weights = None
            weights_unnormed = None
            deformed_shapes = None

        ########### prepare for output ##############
        cage = template_v.transpose(1,2)
        new_cage = new_cage.transpose(1,2)

        return {"source_shape": source_shape.transpose(1,2),
                "cage": cage,
                "new_cage": new_cage,
                "deformed": deformed_shapes,
                "cage_face": self.template_faces,
                "weight": weights,
                "weight_unnormed": weights_unnormed,
                }
