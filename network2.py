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
