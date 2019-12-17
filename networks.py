from __future__ import print_function
import warnings
from pprint import pprint
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data
from pytorch_points.misc import logger
from pytorch_points.network.layers import Conv1d, Linear
from pytorch_points.network.pointnet2_modules import PointnetSAModuleMSG
from pytorch_points.network.model_loss import (ChamferLoss, MeshLaplacianLoss,
                                               PointEdgeLengthLoss,
                                               PointLaplacianLoss, PointStretchLoss,
                                               nndistance)
from pytorch_points.network.geo_operations import (mean_value_coordinates,
                                                   mean_value_coordinates_3D,
                                                   normalize_point_batch_to_sphere)
from pytorch_points.network.operations import faiss_knn
from pytorch_points.utils.geometry_utils import generatePolygon, get_edge_points
from pytorch_points.utils.pc_utils import load, save_ply
from pytorch_points.utils.pytorch_utils import (check_values, load_network,
                                                save_grad, save_network,
                                                saved_variables, weights_init)
from common import deform_with_MVC

class STN(nn.Module):
    def __init__(self, num_points = 2500, dim=3):
        super(STN, self).__init__()
        self.num_points = num_points
        self.conv1 = torch.nn.Conv1d(dim, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        #self.mp1 = torch.nn.MaxPool1d(num_points)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, dim*dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        #x = self.mp1(x)
        #print(x.size())
        x,_ = torch.max(x, 2)
        #print(x.size())
        x = x.view(-1, 1024)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        iden = torch.eye(3, dtype=x.dtype, device=x.device).view(1,9).expand(batchsize,1)
        x = x + iden
        x = x.view(-1, 3, 3)
        return x

class PointNet2feat(nn.Module):
    """
    pointcloud (B,3,N)
    return (B,bottleneck_size)
    """
    def __init__(self, dim=3, num_points=2048, num_levels=3, bottleneck_size=512, normalization=None):
        super().__init__()
        assert(dim==3)
        self.SA_modules = nn.ModuleList()
        self.postSA_mlp = nn.ModuleList()
        NPOINTS = []
        RADIUS = []
        MLPS = []
        start_radius = 0.2
        start_mlp = 24
        self.l_output = []
        for i in range(num_levels):
            NPOINTS += [num_points//4]
            num_points = num_points//4
            RADIUS += [[start_radius, ]]
            start_radius *= 2
            final_mlp = min(256, start_mlp*4)
            MLPS += [[[start_mlp, start_mlp*2, final_mlp], ]]
            start_mlp *= 2
            self.l_output.append(start_mlp)

        bottleneck_size_per_SA = bottleneck_size // len(MLPS)
        self.bottleneck_size = bottleneck_size_per_SA*len(MLPS)

        in_channels = 0
        for k in range(len(MLPS)):
            mlps = [[in_channels]+mlp for mlp in MLPS[k]]
            in_channels = 0
            for idx in range(len(MLPS[k])):
                in_channels += MLPS[k][idx][-1]
            self.SA_modules.append(
                PointnetSAModuleMSG(npoint=NPOINTS[k], radii=RADIUS[k], nsamples=[32,], mlps=mlps, normalization=normalization)
                )
            self.postSA_mlp.append(Conv1d(in_channels, bottleneck_size_per_SA, 1, normalization=normalization, activation="tanh"))

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = (
            pc[..., 3:].transpose(1, 2).contiguous()
            if pc.size(-1) > 3 else None
        )

        return xyz, features

    def forward(self, pointcloud: torch.cuda.FloatTensor, return_all=False):
        pointcloud = pointcloud.transpose(1,2).contiguous()
        li_xyz, li_features = self._break_up_pc(pointcloud)

        # B,C,N
        # l_xyz, l_features = [xyz], [li_features]
        l_xyz, l_features = [], []
        for i in range(len(self.SA_modules)):
            # Pointnetmodule + MLP + maxpool
            li_xyz, li_features = self.SA_modules[i](li_xyz, li_features)
            li_features_post = self.postSA_mlp[i](li_features)
            l_xyz.append(li_xyz)
            l_features.append(li_features_post)

        # max pool (B,4*#SA,1) all SAmodules
        # exclude the first None features
        global_code = torch.cat([torch.max(l_feat, dim=-1)[0] for l_feat in l_features], dim=1)

        l_features.append(global_code)
        l_xyz.append(None)
        if return_all:
            return l_features, l_xyz
        else:
            return global_code

class PointNetfeat3DCoded(nn.Module):
    def __init__(self, npoint=2500, nlatent=1024):
        """Encoder"""

        super(PointNetfeat, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, nlatent, 1)
        self.lin1 = nn.Linear(nlatent, nlatent)
        self.lin2 = nn.Linear(nlatent, nlatent)

        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(nlatent)
        self.bn4 = torch.nn.BatchNorm1d(nlatent)
        self.bn5 = torch.nn.BatchNorm1d(nlatent)

        self.npoint = npoint
        self.nlatent = nlatent

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        point_feat = self.bn3(self.conv3(x))
        x, _ = torch.max(point_feat, 2)
        x = x.view(-1, self.nlatent)
        x = F.relu(self.bn4(self.lin1(x).unsqueeze(-1)))
        x = F.relu(self.bn5(self.lin2(x.squeeze(2)).unsqueeze(-1)))
        return x.squeeze(2), point_feat


class UnetCageGen(nn.Module):
    """
    Receive sampled feature and location from encoder, for each point in the template,
    find k NN points in l_xyz, concatenate their features to get the code
    Params:
        template   Tensor (B,D,N)
        l_xyz      List(Tensor) of shape (B,N_l,D)
        l_features List(Tensor) of shape (B,C_l,N_l)
    Return:
        xyz        Tensor (B,D,N)
        feat_all   Tensor (B,sum_l(C_l), N) if return_aux
    """
    def __init__(self, bottleneck_size, dim=3, knn_k=3,
                 normalization=None, concat_prim=True, n_fold=2, feat_NN=False, **kwargs):
        super().__init__()
        self.decoder = MultiFoldPointGen(bottleneck_size, dim, n_fold=n_fold, normalization=normalization, concat_prim=concat_prim)
        self.feat_NN = feat_NN
        self.knn_k = knn_k


    def interpolate_features(self, query, points, feats, q_normals=None, p_normals=None):
        """
        compute knn point distance and interpolation weight
        :param
            query           (B,M,D)
            points          (B,N,D)
            normals         (B,N,D)
            feats           (B,C,N)
        :return
            distance    Bx1xNxK
            weight      Bx1xNxK
        """
        B, M, D = query.shape
        feats_t = feats.transpose(1,2).contiguous()

        # compute weights based on exponential
        grouped_points, grouped_idx, grouped_dist = faiss_knn(self.knn_k, query, points, NCHW=False)

        # dynamic variance mean_P(min_K distance)
        h = torch.mean(torch.min(grouped_dist, dim=2, keepdim=True)[0], dim=1, keepdim=True) + 1e-8
        # (B,M,K) TODO try linear correlation like in KPconv

        weight = torch.exp(-grouped_dist / (h / 2)).detach()
        sumW = torch.sum(weight, dim=2, keepdim=True)
        weight = weight / sumW

        import pdb; pdb.set_trace()
        # (B, M, K, C)
        grouped_feats_ref = torch.gather(feats_t.unsqueeze(1).expand(-1, M, -1, -1), 2, grouped_idx.unsqueeze(-1).expand(-1,-1,-1, feats_t.shape[-1]))
        # (B, C, M, K)
        grouped_feats = torch.gather(feats.unsqueeze(2).expand(-1, -1, M, -1), 3, grouped_idx.unsqueeze(1).expand(-1, feats.shape[1], -1, -1))
        print(torch.all(torch.eq(grouped_feats, grouped_feats_ref.permute([0, 3, 1, 2]))))
        # (B,C,M,K)
        weighted_feats = torch.sum(grouped_feats*weight.unsqueeze(1), dim=-1)

        return weighted_feats


    def forward(self, template, l_xyz, l_features, return_aux=False):
        B, D, N = template.shape
        template_t = template.transpose(1,2).contiguous()
        interpolated = []
        for i, xyz_feat in enumerate(zip(l_xyz, l_features)):
            # (B,N,3) and (B,C,N)
            xyz, feat = xyz_feat
            # expand global features
            if xyz is None:
                feat = feat.unsqueeze(-1).expand(-1, -1, N)
                interpolated += [feat]
                continue
            # merge neighbors with point distance+normal similarity
            feat = self.interpolate_features(template_t, xyz, feat, q_normals=None, p_normals=None)
            interpolated += [feat]

        # (B,sum(feat_l.shape[1]), M)
        feat_all = torch.cat(interpolated, dim=1)

        xyz = self.decoder(feat_all, template)
        if return_aux:
            return xyz, feat_all
        return xyz

class UnetDeformGen(UnetCageGen):
    """
    Params:
        template            Tensor (B,D,N)
        template_features   Tensor (B, sum_l(C_l), N_l) from UnetCageGen
        l_xyz               List(Tensor) of shape (B,N_l,D)
        l_features          List(Tensor) of shape (B,C_l,N_l)
    Return:
        xyz        Tensor (B,D,N)
        feat_all   Tensor (B,sum_l(C_l), N) if return_aux
    """
    def interpolate_features(self, query_feats, feats, points):
        """
        find the kNN in feature space, interpolate these feature with exponential weights
        :param
            query_feats (B,C,M)
            feats       (B,C,N)
            points      (B,N,D)
        :return
            weighted_feats (B,C,M)
            weighted_xyz   (B,M,dim)
        """
        B, C, M = query_feats.shape
        query_feats_t = query_feats.transpose(1,2).contiguous()
        feats_t = feats.transpose(1,2).contiguous()
        # compute weights based on exponential
        grouped_feats_t, grouped_idx, grouped_dist = faiss_knn(self.knn_k, query_feats_t, feats_t, NCHW=False)
        grouped_feats = grouped_feats_t.permute((0, 3, 1, 2))

        # dynamic variance mean_P(min_K distance)
        h = torch.mean(torch.min(grouped_dist, dim=2, keepdim=True)[0], dim=1, keepdim=True) + 1e-8
        # (B,M,K) TODO try linear correlation like in KPconv
        weight = torch.exp(-grouped_dist / (h / 2)).detach()

        sumW = torch.sum(weight, dim=2, keepdim=True)
        weight = weight / sumW
        weighted_feats = torch.sum(grouped_feats*weight.unsqueeze(1), dim=-1)

        grouped_xyz = torch.gather(points.unsqueeze(1).expand(-1, M, -1, -1), 2, grouped_idx.unsqueeze(-1).expand(-1, -1, -1, points.shape[-1]))
        weighted_xyz = torch.sum(grouped_xyz*weight.unsqueeze(-1), dim=2)

        return weighted_feats, weighted_xyz

    def forward(self, template, template_features, l_xyz, l_features, return_aux=False):
        B, D, N = template.shape
        template_t = template.transpose(1,2).contiguous()
        interpolated = []
        for i, feat_xyz_feat in enumerate(zip(template_features, l_xyz, l_features)):
            # (B,N,3) and (B,C,N)
            query_feat, xyz, feat = feat_xyz_feat
            # expand global features
            if xyz is None:
                feat = feat.unsqueeze(-1).expand(-1, -1, N)
                interpolated += [feat]
                continue
            # merge neighbors with feature distance
            feat, matched_xyz = self.interpolate_features(query_feat, feat, xyz, q_normals=None, p_normals=None)
            interpolated += [feat]

        # (B,sum(feat_l.shape[1])+dim, M)
        feat_all = torch.cat(interpolated+[matched_xyz.transpose(1,2)], dim=1)

        xyz = self.decoder(feat_all, template)
        if return_aux:
            return xyz, feat_all
        return xyz

# encoding input points
class PointNetfeat(nn.Module):
    def __init__(self, dim=3, num_points=2500, global_feat=True, trans=False, bottleneck_size=512, activation="relu", normalization=None):
        super().__init__()
        self.conv1 = Conv1d(dim, 64, 1, activation=activation, normalization=normalization)
        # self.stn_embedding = STN(num_points = num_points, K=64)
        self.conv2 = Conv1d(64, 128, 1, activation=activation, normalization=normalization)
        self.conv3 = Conv1d(128, bottleneck_size, 1, activation=None, normalization=normalization)
        #self.mp1 = torch.nn.MaxPool1d(num_points)

        self.trans = trans
        #self.mp1 = torch.nn.MaxPool1d(num_points)
        self.num_points = num_points
        self.global_feat = global_feat

    def forward(self, x):
        batchsize = x.size()[0]
        if self.trans:
            trans = self.stn(x)
            x = x.transpose(2,1)
            x = torch.bmm(x, trans)
            x = x.transpose(2,1)
        x = self.conv1(x)
        pointfeat = x
        x = self.conv2(x)
        x = self.conv3(x)
        x,_ = torch.max(x, dim=2)
        if self.trans:
            if self.global_feat:
                return x, trans
            else:
                x = x.view(batchsize, -1, 1).repeat(1, 1, self.num_points)
                return torch.cat([x, pointfeat], 1), trans
        else:
            return x

class PointGenCon(nn.Module):
    def __init__(self, bottleneck_size, out_dim, prim_dim, normalization=None, concat_prim=False):
        """
        param:
            cat_prim: keep concatenate atlas coordinate to the features
        """
        super(PointGenCon, self).__init__()
        self.concat_prim = concat_prim
        if concat_prim:
            self.layers = nn.ModuleList([
                Conv1d(bottleneck_size+prim_dim, bottleneck_size//2, 1, activation="lrelu", normalization=normalization),
                Conv1d(bottleneck_size//2+prim_dim, bottleneck_size//4, 1, activation="lrelu", normalization=normalization),
                Conv1d(bottleneck_size//4+prim_dim, out_dim, 1, activation=None, normalization=None),
            ])
        else:
            self.layers = nn.ModuleList([
                Conv1d(bottleneck_size+prim_dim, bottleneck_size//2, 1, activation="lrelu", normalization=normalization),
                Conv1d(bottleneck_size//2, bottleneck_size//4, 1, activation="lrelu", normalization=normalization),
                Conv1d(bottleneck_size//4, out_dim, 1, activation=None, normalization=None),
            ])

    def forward(self, x, primative):
        if x.ndimension() != primative.ndimension():
            x = x.unsqueeze(-1).expand(-1, -1, primative.shape[-1])

        for i, layer in enumerate(self.layers):
            if self.concat_prim or i==0:
                x = torch.cat([x, primative], dim=1)
            if (i+1) == len(self.layers):
                xyz = layer(x)
            else:
                x = layer(x)
        return xyz, x

class MultiFoldPointGen(nn.Module):
    """
    :params:
        code (B,C,1) or (B,C)
        primative (B,dim,P)

    :return:
        primative (B,dim,P)
        [point_feat (B,C,P)] decoder's last feature layer before getting the primiative coordinates
    """
    def __init__(self, bottleneck_size, out_dim=3, prim_dim=3,
                n_fold=3, normalization=None, concat_prim=True, residual=True, return_aux=True):
        super().__init__()
        folds = []
        self.prim_dim = prim_dim
        for i in range(n_fold):
            cur_out_dim = min(bottleneck_size, 64 * (n_fold-i)) if (i+1) < n_fold else 3
            folds += [PointGenCon(bottleneck_size, cur_out_dim, prim_dim, normalization=normalization, concat_prim=concat_prim)]
        self.folds = nn.ModuleList(folds)
        self.return_aux = return_aux
        self.residual = residual
        if self.residual:
            assert(prim_dim==out_dim)

    def forward(self, code, primative):
        for i, fold in enumerate(self.folds):
            if code.ndimension() != primative.ndimension():
                code_exp = code.unsqueeze(-1).expand(-1, -1, primative.shape[-1])
            else:
                code_exp = code.expand(-1, -1, primative.shape[-1])
            assert(primative.shape[1] == self.prim_dim)
            xyz, point_feat = fold(code_exp, primative)

        if self.residual:
            xyz = primative+xyz
        if self.return_aux:
            return xyz, point_feat
        return xyz


class MLPDeformer(nn.Module):
    def __init__(self, dim, bottleneck_size, npoint, residual=True, normalization=None):
        super().__init__()
        self.npoint = npoint
        self.dim = dim
        self.residual = residual
        self.layers = nn.Sequential(
                Linear(bottleneck_size, 512, activation="lrelu", normalization=normalization),
                Linear(512, 256, activation="lrelu", normalization=normalization),
                Linear(256, npoint*dim)
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


class NetworkFull(nn.Module):
    def __init__(self, opt, dim, bottleneck_size,
                 template_vertices, template_faces,
                 **kargs
                 ):

        super().__init__()
        self.opt = opt
        self.dim = dim
        self.set_up_template(template_vertices, template_faces)

        ###### source and target encoder ########
        # we don't share encoder because if we want to update deformer only, we don't want to change cage
        if opt.pointnet2:
            self.encoder = PointNet2feat(dim=dim, num_points=opt.num_point, bottleneck_size=bottleneck_size)
        else:
            self.encoder = nn.Sequential(
                PointNetfeat(dim=dim, num_points=opt.num_point,bottleneck_size=bottleneck_size),
                Linear(bottleneck_size, bottleneck_size, activation="lrelu", normalization=opt.normalization)
                )

        ###### cage prediction and cage deformation ########
        if opt.full_net:
            if not opt.atlas:
                self.nc_decoder = MLPDeformer(dim=dim, bottleneck_size=bottleneck_size, npoint=self.template_vertices.shape[-1],
                                            residual=opt.c_residual, normalization=opt.normalization)
            else:
                self.nc_decoder = MultiFoldPointGen(bottleneck_size, dim, dim, n_fold=opt.n_fold, normalization=opt.normalization,
                                                    concat_prim=opt.concat_prim, return_aux=False, residual=opt.c_residual)


        self.D_use_C_global_code = opt.c_global
        self.merger = nn.Sequential(
                Conv1d(bottleneck_size*2, bottleneck_size*2, 1, activation="lrelu", normalization=opt.normalization),
            )
        if not opt.atlas:
            self.nd_decoder = MLPDeformer(dim=dim, bottleneck_size=bottleneck_size*2, npoint=self.template_vertices.shape[-1],
                                        residual=opt.d_residual, normalization=opt.normalization)
        else:
            # self.nd_decoder = torch.nn.ModuleList(
            #     [
            #         MultiFoldPointGen(bottleneck_size*2, 64, dim, n_fold=opt.n_fold, normalization=opt.normalization,
            #                        concat_prim=opt.concat_prim, return_aux=False, residual=False),
            #         MultiFoldPointGen(bottleneck_size, dim, 64, n_fold=opt.n_fold, normalization=opt.normalization,
            #                        concat_prim=opt.concat_prim, return_aux=False, residual=False),
            #     ]
            self.nd_decoder = MultiFoldPointGen(bottleneck_size*2, dim, dim, n_fold=opt.n_fold, normalization=opt.normalization,
                                   concat_prim=opt.concat_prim, return_aux=False, residual=opt.d_residual)


    def set_up_template(self, template_vertices, template_faces):
        # save template as buffer
        assert(template_vertices.ndim==3 and template_vertices.shape[1]==self.dim) # (1,3,V)
        if self.dim == 3:
            assert(template_faces.ndim==3 and template_faces.shape[2]==3) # (1,F,3)

        self.register_buffer("template_faces", template_faces)
        self.register_buffer("template_vertices", template_vertices)
        self.template_vertices = nn.Parameter(self.template_vertices, requires_grad=(self.opt.optimize_template))
        if self.template_vertices.requires_grad:
            logger.info("Enabled vertex optimization")

    def forward(self, source_shape, target_shape, alpha=1.0):
        """
        source_shape (B,3,N)
        target_shape (B,3,M)
        init_cage    (B,3,P)
        return:
            deformed (B,3,N)
            cage     (B,3,P)
            new_cage (B,3,P)
            weights  (B,N,P)
        """
        B, _, N = source_shape.shape
        _, M, N = target_shape.shape
        _, _, P = self.template_vertices.shape
        ############ Encoder #############
        input_shapes = torch.cat([source_shape, target_shape], dim=0)
        shape_code = self.encoder(input_shapes)
        shape_code.unsqueeze_(-1)
        s_code, t_code = torch.split(shape_code, B, dim=0)

        # ############ Cage ################
        cage = self.template_vertices.view(1,self.dim,-1).expand(B,-1,-1)
        if self.opt.full_net:
            cage = self.nc_decoder(s_code, cage)
            # cage.register_hook(save_grad("d_cage"))

        ########### Deform ##########
        # first fold use global feature
        target_code = torch.cat([s_code, t_code], dim=1)
        target_code = self.merger(target_code)
        new_cage = self.nd_decoder(target_code, cage)

        ########### MVC ##############
        if self.dim == 3:
            cage = cage.transpose(1,2).contiguous()
            new_cage = new_cage.transpose(1,2).contiguous()
            deformed_shapes, weights, weights_unnormed = deform_with_MVC(cage,
                                                                         new_cage,
                                                                         self.template_faces.expand(B,-1,-1),
                                                                         source_shape.transpose(1,2).contiguous(),
                                                                         verbose=True)
        elif self.dim == 2:
            weights, weights_unnormed = mean_value_coordinates(source_shape, cage, verbose=True)
            deformed_shapes = torch.sum(weights.unsqueeze(1)*new_cage.unsqueeze(-1), dim=2).transpose(1,2).contiguous()
            cage = cage.transpose(1,2)
            new_cage = new_cage.transpose(1,2)

        return {
            "cage": cage,
            "new_cage": new_cage,
            "deformed": deformed_shapes,
            "cage_face": self.template_faces,
            "weight": weights,
            "weight_unnormed": weights_unnormed
        }



# class UNetwork(nn.Module):
#     def __init__(self, dim, num_points, bottleneck_size, polygon_degree=None, template=None,
#                  C_residual=True, D_residual=True, D_use_enc_code=True, normalization=None,
#                  knn_k=3, concat_prim=True, multi_fold=False, **kwargs):
#         super().__init__()
#         self.num_points = num_points
#         self.dim = dim
#         self.C_residual = C_residual
#         self.D_residual = D_residual
#         self.encoder = PointNet2feat(dim=dim, num_points=num_points, bottleneck_size=bottleneck_size)
#         self.use_enc_code = D_use_enc_code

#         # num_points, bottleneck_size, dim=3,
#                 #  residual=True, normalization=None, concat_prim=True, multi_fold=True, **kwargs):
#         code_len = sum(self.encoder.l_output)
#         self.nc_decoder = UnetCageGen(code_len, dim, knn_k=knn_k, n_fold=2, normalization=normalization, concat_prim=concat_prim)
#         # find the nn in feature space, concatenate feature values
#         self.nd_decoder = UnetDeformGen(code_len+dim, dim, knn_k=knn_k, n_fold=2, normalization=normalization, concat_prim=concat_prim, feat_NN=True)
#         self.template = template


#     def forward(self, source_shape, target_shape, template=None, cage_F=None):
#         """
#         source_shape (B,3,N)
#         target_shape (B,3,M)
#         init_cage    (B,3,P)
#         cage_F       (B,F,3)  for dim=3
#         return:
#             deformed (B,3,N)
#             cage     (B,3,P)
#             new_cage (B,3,P)
#             weights  (B,N,P)
#         """
#         assert(source_shape.shape[-1]==self.num_points)
#         batchsize = source_shape.shape[0]

#         nc_code, nc_xyz = self.encoder(source_shape, return_all=True)
#         if template is None:
#             if self.template is None and dim==2:
#                 template = generatePolygon(0, 0, 1.5, 0, 0, 1, self.polygon_degree)
#                 template = torch.tensor([(x, y) for x, y in template], dtype=torch.float).unsqueeze(0).transpose(1, 2).cuda()
#             elif self.template is not None:
#                 template = self.template.detach()
#             else:
#                 raise ValueError("must provide template at class initialization or forward pass for dim == 3")

#         # (B, dim, primitiveSize)
#         template = template.expand(batchsize, -1, -1)
#         # fit cage
#         cage, nc_code_interpolated = self.nc_decoder(template, nc_xyz, nc_code, return_aux=True)
#         if self.C_residual:
#             cage = cage+template
#         # encode target using the shared encoder
#         nd_code, nd_xyz = self.encoder(target_shape, return_all=True)
#         # if self.use_enc_code:
#         #     target_code = torch.cat([nc_code, target_code], dim=1)
#         # else:
#         #     target_code = target_code

#         l_feat_len = [c.shape[1] for c in nc_code]
#         l_nc_code_interpolated = torch.split(nc_code_interpolated, l_feat_len, dim=1)
#         nd_feat = torch.split(nd_code)
#         new_cage, _ = self.nd_decoder(cage, l_nc_code_interpolated, nc_xyz, nc_code, return_aux=True)

#         if self.D_residual:
#             new_cage = new_cage + cage

#         if self.dim == 3:
#             source_shape = source_shape.transpose(1,2)
#             cage_t = cage.transpose(1,2)
#             weights = mean_value_coordinates_3D(source_shape, cage_t, cage_F)
#             new_cage_t = new_cage.transpose(1,2)
#             # (B,P,N)*(B,N,3)->(B,N,3)
#             deformed = torch.sum(weights.unsqueeze(-1)*new_cage_t.unsqueeze(1), dim=2)
#             deformed = deformed.transpose(1,2)
#             weights = weights.transpose(1,2)
#         elif self.dim == 2:
#             weights = mean_value_coordinates(source_shape, cage)
#             deformed = torch.sum(weights.unsqueeze(1)*new_cage.unsqueeze(-1), dim=2)
#         return deformed, cage, new_cage, weights


if __name__ == "__main__":
    npoint = 5000
    # net = NetworkFull(dim=3, num_points=npoint, nc_bottleneck=512, nd_bottleneck=512,
    #                 C_residual=True, D_residual=True, D_use_enc_code=True, MLP=False,
    #                 concat_prim=True, multi_fold=True, normalization=None).cuda()
    # net = NetworkSharedEnc(dim=3, num_points=npoint, bottleneck_size=512,
    #                        C_residual=True, D_resiual=True, D_use_enc_code=True, template=None, normalization=None,
    #                        concat_prim=True, multi_fold=True)
    net = UNetwork(dim=3, num_points=npoint, bottleneck_size=512,
                   C_residual=True, D_residual=True, D_use_enc_code=True, template=None, normalization=None,
                   concat_prim=False, multi_fold=False)
    print(net)
    # points = torch.rand((4, 3, npoint)).cuda()
    # cage_V = torch.rand((4, 3, 120)).cuda()
