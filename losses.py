from collections import defaultdict
import numpy as np
import torch
from pytorch_points.network.operations import faiss_knn, dot_product, batch_svd, ball_query, group_knn
from pytorch_points.utils.pytorch_utils import save_grad, linear_loss_weight
from pytorch_points.network.model_loss import nndistance, labeled_nndistance
from pytorch_points.network.geo_operations import (compute_face_normals_and_areas, dihedral_angle,
                                                  CotLaplacian, UniformLaplacian, batch_normals)
from pytorch_points.network.model_loss import (MeshLaplacianLoss, PointEdgeLengthLoss, \
                                               MeshStretchLoss, PointStretchLoss, PointLaplacianLoss,
                                               SimpleMeshRepulsionLoss, MeshEdgeLengthLoss,
                                               NormalLoss)
from pytorch_points.misc import logger

class AllLosses(torch.nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.loss = defaultdict(float)
        self.labeled_chamfer_loss = LabeledChamferDistance(beta=opt.beta, gamma=opt.gamma, delta=opt.delta)
        self.cage_shortLength_loss = SimpleMeshRepulsionLoss(0.02, reduction="mean", consistent_topology=True)
        self.cage_faceAngle_loss = MeshDihedralAngleLoss(threshold=np.pi/30)
        self.mvc_reg_loss = MVCRegularizer(threshold=50, beta=1.0, alpha=0.0)
        self.cage_laplacian = MeshLaplacianLoss(torch.nn.L1Loss(reduction="mean"), use_cot=False, use_norm=True,
                                                consistent_topology=True, precompute_L=True)
        self.cage_smooth_loss = MeshSmoothLoss(torch.nn.MSELoss(reduction="mean"), use_cot=False, use_norm=True)
        self.grounding_loss = GroundingLoss(up_dim=(1 if "SHAPENET" in opt.dataset else 2))
        if opt.sym_plane is not None:
            self.symmetry_loss = SymmetryLoss(sym_plane=opt.sym_plane, NCHW=False).cuda()
        # mesh_chamfer_loss = losses.InterpolatedCDTriMesh(interpolate_n=5, beta=1.0, gamma=0.0, delta=1/30)
        # cage_inside_loss = InsideLoss3DTriMesh(reduction="max")
        # cage_inside_loss = ExtPointToNearestFaceDistance(reduction="mean", min_dist=opt.cinside_eps)
        if self.opt.dataset in ("SURREAL", "FAUST"):
            logger.info("Using GTNormal loss")
            self.shape_normal_loss = GTNormalLoss()
        else:
            logger.info("Using KNN for normal loss")
            self.shape_normal_loss = NormalLoss(reduction="none", nn_size=16)
        self.shape_fnormal_loss = FaceNormalLoss(n_faces=300)
        self.stretch_loss = PointStretchLoss((4 if opt.dim==3 else 2), reduction="mean")
        self.edge_loss = PointEdgeLengthLoss((4 if opt.dim==3 else 2), torch.nn.MSELoss(reduction="mean"))
        if self.opt.regular_sampling or (not opt.mesh_data):
            logger.info("Using point laplacian loss")
            self.shape_laplacian = PointLaplacianLoss(16, torch.nn.MSELoss(reduction="none"), use_norm=opt.slap_norm)
        else:
            logger.info("Using mesh laplacian loss")
            self.shape_laplacian = MeshLaplacianLoss(torch.nn.MSELoss(reduction="none"), use_cot=True,
                                                     use_norm=True, consistent_topology=True, precompute_L=True)
        self.p2f_loss = LocalFeatureLoss(16, torch.nn.MSELoss(reduction="none"))

    def forward(self, all_inputs, all_outputs, progress=1.0):
        self.loss.clear()
        B = all_outputs["new_cage"].shape[0]
        # ======== cage deformation back and forth ============= #
        if self.opt.loss == "LCD":
            loss, idx12, idx21 = self.labeled_chamfer_loss(all_outputs["deformed"],
                                                          all_inputs["target_shape"],
                                                          all_inputs["source_label"],
                                                          all_inputs["target_label"])
            self.idx12 = idx12.to(dtype=torch.int64)
            self.idx21 = idx21.to(dtype=torch.int64)
            self.loss["LCD"] += loss*opt.loss_weight
            # S-to-S use MSE
            dist = torch.sum((all_outputs["deformed"][self.opt.batch_size*2:, :, :] - all_inputs["target_shape"][self.opt.batch_size*2:,:,:])**2, dim=-1)
            self.loss["MSE"] += dist.mean()*opt.loss_weight
        elif self.opt.loss == "CD":
            loss, idx12, idx21  = self.labeled_chamfer_loss(all_outputs["deformed"],
                                                            all_inputs["target_shape"])
            self.loss["CD"] = loss
            self.loss["CD"] *= self.opt.loss_weight
            self.idx12 = idx12.to(dtype=torch.int64)
            self.idx21 = idx21.to(dtype=torch.int64)
            # S-to-S use MSE
            dist = torch.sum((all_outputs["deformed"][self.opt.batch_size*2:, :, :] - all_inputs["target_shape"][self.opt.batch_size*2:,:,:])**2, dim=-1)
            self.loss["MSE"] += dist.mean()*self.opt.loss_weight
        elif self.opt.loss == "MSE":
            dist = torch.sum((all_outputs["deformed"] - all_inputs["target_shape"])**2, dim=-1)
            self.loss["MSE"] += dist.mean()
            self.loss["MSE"] += torch.max(dist, dim=1)[0].mean()
            self.loss["MSE"] *= self.opt.loss_weight
            # self.loss["MSE"] += torch.sum((all_outputs["t_deformed"] - all_inputs["source_shape"])**2, dim=-1).mean()

        # ======== cage surface close to the source shape ============= #
        if self.opt.cshape_weight > 0:
            ref_surface = all_inputs["source_shape"]+0.1*all_inputs["source_normals"]
            loss, _, _ = self.labeled_chamfer_loss(all_outputs["cage"], ref_surface)
            self.loss["CSHAPE"] += loss
            self.loss["CSHAPE"] *= linear_loss_weight(self.opt.nepochs, progress, self.opt.cshape_weight, 0)

        # ======== cage center must be close to shape center ========== #
        if self.opt.gravity_weight > 0:
            cage_shift = torch.mean(all_outputs["cage"], dim=1) - torch.mean(all_inputs["source_shape"], dim=1)
            self.loss["GRAV"] += torch.mean(torch.nn.functional.softshrink(torch.sum(cage_shift**2, dim=-1), lambd=0.1))
            # cage_shift = torch.mean(all_outputs["new_cage"], dim=1) - torch.mean(all_inputs["target_shape"], dim=1)
            # self.loss["GRAV"] += torch.mean(torch.nn.functional.softshrink(torch.sum(cage_shift**2, dim=-1), lambd=0.1))
            self.loss["GRAV"] *= self.opt.gravity_weight

        # ======== penalize large unnormalized weight and/or negative weights ========== #
        if self.opt.mvc_weight > 0:
            self.loss["WREG"] += self.mvc_reg_loss(all_outputs["weight"]) * self.opt.mvc_weight

        # ======== feature preservation via point to surface ======== #
        if self.opt.p2f_weight > 0:
            self.loss["P2F"] = torch.mean(self.p2f_loss(all_inputs["source_shape"], all_outputs["deformed"]))
            self.loss["P2F"] *= linear_loss_weight(self.opt.nepochs, progress, self.opt.p2f_weight, self.opt.p2f_weight/10)

        # ======== feature preservation via laplacian ========== #
        if self.opt.slap_weight > 0:
            # reduction none (B,P)
            slap1 = torch.mean(
                self.shape_laplacian(all_inputs["source_shape"], all_outputs["deformed"], face=all_inputs["source_face"]).view(B,-1),
                dim=-1, keepdim=True)
            # use idx12 to get the closest points on the target, laplacian of these points compute
            if self.opt.blend_style and hasattr(self, "idx21"):
                slap1 *= (1-all_inputs["alpha"])
                # slap2 = 0.5*torch.mean(self.shape_laplacian(all_outputs["deformed"], all_inputs["target_shape"], idx12=self.idx12), dim=-1)
                slap2 = torch.mean(
                    self.shape_laplacian(all_outputs["deformed"], all_inputs["target_shape"], idx12=self.idx12).view(B,-1),
                    dim=-1, keepdim=True)
                slap2 *= all_inputs["alpha"]
                self.loss["SLAP"] += slap2.mean()
            self.loss["SLAP"] += slap1.mean()
            self.loss["SLAP"] *= linear_loss_weight(self.opt.nepochs, progress, self.opt.slap_weight, self.opt.slap_weight/10)

        # ======== feature preservation via normal ================= #
        if self.opt.snormal_weight > 0:
            snormal1 = torch.mean(
                self.shape_normal_loss(all_inputs["source_shape"], all_outputs["deformed"]), dim=-1, keepdim=True)
            if self.opt.blend_style and hasattr(self, "idx21"):
                snormal1 *= (1-all_inputs["alpha"])
                # snormal2 = 0.5*torch.mean(self.shape_normal_loss(all_inputs["deformed"], all_inputs["target_shape"], idx=self.idx12), dim=-1)
                snormal2 = torch.mean(
                    self.shape_normal_loss(all_outputs["deformed"], all_inputs["target_shape"], idx12=self.idx12),
                    dim=-1, keepdim=True)
                snormal2 *= all_inputs["alpha"]
                self.loss["SNORMAL"] += snormal2.mean()
            self.loss["SNORMAL"] += snormal1.mean()
            self.loss["SNORMAL"] *= linear_loss_weight(self.opt.nepochs, progress, self.opt.snormal_weight, self.opt.snormal_weight/10)

        # ======== enforce symmetry on cage ========== #
        if self.opt.sym_weight > 0:
            self.loss["SYM"] += self.symmetry_loss(all_outputs["deformed"])
            self.loss["SYM"] += self.symmetry_loss(all_outputs["cage"])
            self.loss["SYM"] *= self.opt.sym_weight

        # ======== enforce to stay on the ground ========== #
        if self.opt.ground_weight > 0:
            self.loss["GROUND"] += self.grounding_loss(all_inputs["source_shape"], all_outputs["deformed"])
            self.loss["GROUND"] *= self.opt.ground_weight

        # ======== cage face angle should be larger than pi/6 ========== #
        if self.opt.cfangle_weight > 0:
            # self.loss["CFANGLE"] += self.cage_faceAngle_loss(all_outputs["cage"], edge_points=all_inputs["cage_edge_points"])
            self.loss["CFANGLE"] += self.cage_faceAngle_loss(all_outputs["new_cage"], edge_points=all_inputs["cage_edge_points"])
            self.loss["CFANGLE"] *= self.opt.cfangle_weight

        # ======== cage face angle should be larger than pi/6 ========== #
        if self.opt.csmooth_weight > 0:
            # self.loss["CSMOOTH"] += self.cage_smooth_loss(all_outputs["cage"], face=all_outputs["cage_face"])
            self.loss["CSMOOTH"] += self.cage_smooth_loss(all_outputs["new_cage"], face=all_outputs["cage_face"])
            self.loss["CSMOOTH"] *= self.opt.csmooth_weight

        # ======== penalize cage with very short edges ================= #
        if self.opt.cshort_weight > 0:
            # TODO add cage_edges to all_inputs
            self.loss["CEDGE"] = self.cage_shortLength_loss(all_outputs["cage"], edges=all_inputs["cage_edges"])
            # self.loss["CEDGE"] = self.cage_shortLength_loss(all_outputs["t_cage"], edges=all_inputs["cage_edges"])
            self.loss["CEDGE"] *= self.opt.cshort_weight

        # ======== require new cage similar to cage ================= #
        if self.opt.clap_weight > 0:
            self.loss["CLAP"] += self.cage_laplacian(all_outputs["cage"].expand(B,-1,-1).contiguous().detach(),
                                                     all_outputs["new_cage"].contiguous(), face=all_outputs["cage_face"])
            self.loss["CLAP"] *= self.opt.clap_weight

        # ======== penalize increasing point distance ================= #
        if self.opt.sstretch_weight > 0:
            self.loss["SSTRETCH"] += self.stretch_loss(all_outputs["source_shape"], all_outputs["deformed"])*self.opt.sstretch_weight

        # ======== penalize knn distance change ================= #
        if self.opt.sedge_weight > 0:
            self.loss["SEDGE"] += self.edge_loss(all_outputs["source_shape"], all_outputs["deformed"])
            self.loss["SEDGE"] *= linear_loss_weight(self.opt.nepochs, progress, self.opt.sedge_weight, self.opt.sedge_weight/10)

        if self.opt.sfnormal_weight > 0:
            # randomly compare a subset of face normals
            self.loss["SFNORMAL"] += self.shape_fnormal_loss(all_inputs["target_mesh"], all_outputs["deformed_hr"],
                                                             all_inputs["source_face"].expand(B,-1,-1))
            self.loss["SFNORMAL"] *= linear_loss_weight(self.opt.nepochs, progress, self.opt.sfnormal_weight, self.opt.sfnormal_weight/10)

        return self.loss

class FaceNormalLoss(torch.nn.Module):
    def __init__(self, n_faces=100):
        super().__init__()
        self.n_faces= n_faces
        self.cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-08)

    def forward(self, ref_mesh_V, mesh_V, mesh_F):
        B, F, _ = mesh_F.shape
        face_sample_idx = torch.randint(min(self.n_faces, F), (B, self.n_faces, 1), dtype=torch.int64).to(device=mesh_F.device)
        sampled_F = torch.gather(mesh_F, 1, face_sample_idx.expand(-1,-1,3))
        ref_normals,_ = compute_face_normals_and_areas(ref_mesh_V, mesh_F)
        normals,_ = compute_face_normals_and_areas(mesh_V, mesh_F)
        cos = self.cos(ref_normals, normals)
        return torch.mean(1-cos)

class GroundingLoss(torch.nn.Module):
    def __init__(self, up_dim=1):
        super().__init__()
        self.up_dim = up_dim  # if z is the up direction, the up_dim = 2
    # previous ground stays on the ground
    def forward(self, source, deformed):
        """
        source:   (B,N,3)
        deformed: (B,N,3)
        """
        eps = 1e-2
        ground_level = torch.min(source[:,:,self.up_dim], dim=1)[0]
        ground_point_mask = (source[:,:,self.up_dim] - ground_level.unsqueeze(-1)).abs() < eps
        source_ground_level = torch.masked_select(source[:,:,self.up_dim], ground_point_mask)
        deformed_ground_level = torch.masked_select(deformed[:,:,self.up_dim], ground_point_mask)
        return torch.mean(torch.abs(source_ground_level - deformed_ground_level))

# class SymmetricPointFaceDistance(nn.Module):
#     def forward(self, input, input_normals, target, target_normals):


class ExtPointToNearestFaceDistance(torch.nn.Module):
    """
    for every exteror points return the squared distance to the closest face
    """
    def __init__(self, min_dist=0.1, reduction="mean"):
        super().__init__()
        self.min_dist = min_dist
        self.reduction = reduction

    def forward(self, mesh_V, mesh_F, points, exterior_flag, mesh_FN=None):
        """
        mesh_V        (B,N,3)
        mesh_F        (B,F,3)
        mesh_FN       (B,F,3)
        points        (B,P,3)
        exterior_flat (B,P,1)
        """
        if mesh_FN is None:
            mesh_FN, _ = compute_face_normals_and_areas(mesh_V, mesh_F)
            mesh_FN = mesh_FN.detach()
        else:
            mesh_FN = mesh_FN.detach()

        B, F, _ = mesh_F.shape
        _, N, D = mesh_V.shape
        _, P, D = points.shape
        # (B,N,D) (B,F,3) -> (B,F,3,3) face points
        face_points = torch.gather(mesh_V.unsqueeze(1).expand(-1,F,-1,-1), 2, mesh_F.unsqueeze(-1).expand(-1,-1,-1,3))
        # (B,F,3)
        face_center = torch.mean(face_points, dim=-2)
        # (B,P,F,3)
        point_to_face_center = points.unsqueeze(2) - face_center.unsqueeze(1)
        # point to face distance (B,P,F,3)
        point_to_face_signed_dist = (dot_product(point_to_face_center, mesh_FN.unsqueeze(1), dim=-1, keepdim=True)+self.min_dist)
        point_to_face_v = point_to_face_signed_dist * mesh_FN.unsqueeze(1)
        # (B,P,F)
        point_to_face_sqdist = torch.sum(point_to_face_v*point_to_face_v, dim=-1)
        # ignore faces outside the points
        point_to_face_sqdist.masked_fill_(point_to_face_signed_dist.squeeze(-1)<0, 1e10)
        # (B,P)
        point_to_face_sqdist, _ = torch.min(point_to_face_sqdist, dim=-1)
        # ignore interior points
        inside_flag = (~exterior_flag.view(B,P))| torch.all(point_to_face_signed_dist.view(B,P,F)<0, dim=-1)
        point_to_face_sqdist.masked_fill_(inside_flag, 0)
        if self.reduction == "mean":
            point_to_face_sqdist = torch.mean(point_to_face_sqdist.view(B,-1), dim=1)
        elif self.reduction == "max":
            point_to_face_sqdist = torch.max(point_to_face_sqdist.view(B,-1), dim=1)[0]
        elif self.reduction == "sum":
            point_to_face_sqdist = torch.sum(point_to_face_sqdist.view(B,-1), dim=1)
        elif self.reduction == "none":
            pass
        else:
            raise NotImplementedError

        point_to_face_sqdist = torch.mean(point_to_face_sqdist, dim=0)
        return point_to_face_sqdist


class MVCRegularizer(torch.nn.Module):
    """
    penalize MVC with large absolute value and negative values
    alpha * large_weight^2 + beta * (negative_weight)^2
    """
    def __init__(self, alpha=1.0, beta=1.0, threshold=5.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.threshold = threshold

    def forward(self, weights):
        # ignore all weights <= 5
        # B, N, F, _ = loss.shape
        loss = 0
        if self.alpha > 0:
            large_loss = torch.log(torch.nn.functional.relu(weights.abs()-self.threshold)+1)
            # large_loss = large_loss ** 2
            loss += (torch.mean(large_loss)) * self.alpha
        if self.beta > 0:
            neg_loss = torch.nn.functional.relu(-weights)
            neg_loss = neg_loss ** 2
            loss += (torch.mean(neg_loss)) * self.beta

        return loss

class LabeledChamferDistance(torch.nn.Module):
    """
    Learning to Sample Dovrat et.al
    mean_{xyz1}(nd_{1to2})+\beta*max_{xyz1}(nd_{1to2})+(\gamma+\delta|xyz1|)mean_{xyz2}(nd_{2to1})
    ===
    :param:
        xyz1: generated points
        xyz2: reference points
    """
    def __init__(self, beta=1.0, gamma=1, delta=0):
        super().__init__()
        self.beta = beta
        self.gamma = gamma
        self.delta = delta

    def forward(self, xyz1, xyz2, label1=None, label2=None):
        P = xyz1.shape[1]
        if label1 is not None and label2 is not None:
            dist12, dist21, idx12, idx21 = labeled_nndistance(xyz1, xyz2, label1, label2)
        else:
            dist12, dist21, idx12, idx21 = nndistance(xyz1, xyz2)
        # pred2gt is for each element in gt, the closest distance to this element
        loss = torch.mean(dist12, dim=-1) + torch.max(dist12, dim=-1)[0]*self.beta + (self.gamma+self.delta*P)*(torch.mean(dist21, dim=-1))
        loss = torch.mean(loss)
        return loss, idx12, idx21

class SymmetryLoss(torch.nn.Module):
    """
    symmetry loss
    chamfer(mirrored(xyz), xyz)
    ===
    :params:
        sym_plane ("yz"): list of "xy", "yz", "zx"
        NCHW      bool  : point dimension
        xyz             : (B,3,N) or (B,N,3)
    """
    def __init__(self, sym_plane=("yz",), NCHW=True):
        super().__init__()
        self.sym_plane = sym_plane
        assert(isinstance(self.sym_plane, tuple) or isinstance(self.sym_plane, list)), "sym_plane must be a list or tuple"
        self.metric = LabeledChamferDistance(beta=0.0, gamma=1.0, delta=0)
        self.register_buffer("base_ones", torch.ones((3,), dtype=torch.float))
        self.NCHW = NCHW

        self.mirror_ops = []
        for p in self.sym_plane:
            if 'x' not in p:
                self.mirror_ops += [lambda xyz: xyz*self.get_mirror_multiplier(0)]
            elif 'y' not in p:
                self.mirror_ops += [lambda xyz: xyz*self.get_mirror_multiplier(1)]
            elif 'z' not in p:
                self.mirror_ops += [lambda xyz: xyz*self.get_mirror_multiplier(2)]
            else:
                raise ValueError

    def get_mirror_multiplier(self, dim_id):
        base_ones = self.base_ones.clone()
        base_ones[dim_id] = -1
        if self.NCHW:
            return base_ones.view((1,3,1))
        else:
            return base_ones.view((1,1,3))

    def forward(self, xyz):
        loss = 0
        for op in self.mirror_ops:
            m_xyz = op(xyz)
            loss += self.metric(m_xyz.detach(), xyz)[0]
        return loss


class ConditionNumberLoss(torch.nn.Module):
    """
    compare ratio of the largest and smallest principal component values
    ===
    params:
        ref_points: (B,N,dim)
        points:     (B,N,dim)
    """
    def __init__(self, ball_size, metric, reduction="mean"):
        super().__init__()
        self.reduction = reduction
        self.ball_size2 = ball_size * 2
        self.metric = metric
        self.nn_size = 16

    def forward(self, ref_points, points, *args, **kwargs):
        B,N,C = ref_points.shape
        # TODO replace with ball query
        # (B,P,K,3), (B,P,K), (B,P,K)
        ref_grouped_points, ref_group_idx, ref_group_dist = faiss_knn(self.nn_size, ref_points, ref_points, NCHW=False)
        mask = (ref_group_dist < self.ball_size2)
        ref_grouped_points.masked_fill_(~mask.unsqueeze(-1), 0.0)
        # number of points inside the ball (B,P,1)
        nball = torch.sum(mask.to(torch.float), dim=-1, keepdim=True)
        ref_group_center = torch.sum(ref_grouped_points, dim=2, keepdim=True)/nball.unsqueeze(-1)
        # B,P,K,3
        ref_points = ref_grouped_points - ref_group_center
        ref_allpoints = ref_points.view(-1, self.nn_size, C).contiguous()
        U_ref, S_ref, V_ref = batch_svd(ref_allpoints)
        ref_cond = S_ref[:,0]/(S_ref[:,-1]+S_ref[:,0])
        ref_cond = ref_cond.view(B, N).contiguous()

        # grouped_points, group_idx, _ = faiss_knn(self.nn_size, points, points, NCHW=False)
        grouped_points = torch.gather(points.unsqueeze(1).expand(-1,N,-1,-1), 2, ref_group_idx.unsqueeze(-1).expand(-1,-1,-1,C))
        grouped_points.masked_fill(~mask.unsqueeze(-1), 0.0)
        group_center = torch.sum(grouped_points, dim=2, keepdim=True)/nball.unsqueeze(-1)
        points = grouped_points - group_center
        allpoints = points.view(-1, self.nn_size, C).contiguous()
        # S (BN, k)
        U, S, V = batch_svd(allpoints)
        cond = S[:,0]/(S[:,-1]+S[:,0])
        cond = cond.view(B, N).contiguous()


        return self.metric(cond,ref_cond)


class InsideLoss2D(torch.nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, cage, shape, shape_normals, epsilon=0.01, interpolate=True):
        """ Penalize polygon cage that is inside the given shape
        Args:
            cage: (B,M,3)
            shape: (B,N,3)
            shape_normals: (B,N,3)
        return:

        """
        B,M,D = cage.shape
        interpolate_n = 10
        # find the closest point on the shape
        cage_p = cage[:,[i for i in range(1, M)]+[0], :]
        t = torch.linspace(0, 1, interpolate_n).to(device=cage_p.device)
        # B,M,K,3
        cage_itp = t.reshape([1, 1, interpolate_n, 1])*cage_p.unsqueeze(2).expand(-1, -1, interpolate_n, -1) + \
            (1-t.reshape([1, 1, interpolate_n, 1]))*cage.unsqueeze(2).expand(-1, -1, interpolate_n, -1)
        cage_itp = cage_itp.reshape(B, -1, D)
        nn_point, nn_index, _ = faiss_knn(1, cage_itp, shape, NCHW=False)
        nn_point = nn_point.squeeze(2)
        nn_normal = torch.gather(
            shape_normals.unsqueeze(1).expand(-1, nn_index.shape[1], -1, -1), 2,
            nn_index.unsqueeze(-1).expand(-1,-1,-1,shape_normals.shape[-1]))
        nn_normal = nn_normal.squeeze(2)

        # if <(q-p), n> is negative, then this point is inside the shape, gradient is along the normal direction
        dot = dot_product(cage_itp - nn_point - epsilon*nn_normal, nn_normal, dim=-1)
        loss = torch.where(dot < 0, -dot, torch.zeros_like(dot))

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "max":
            return torch.mean(torch.max(loss, dim=-1)[0])
        elif self.reduction == "sum":
            return loss.mean(torch.sum(loss, dim=-1))
        elif self.reduction == "none":
            return loss
        else:
            raise NotImplementedError
        return loss

class InterpolatedCDTriMesh(torch.nn.Module):
    """
    Reconstruction between cage and shape
    mean(shape2cage) + beta*max(shape2cage) + (gamma+delta*|CAGE|*mean(cage2shape))
    """
    def __init__(self, interpolate_n=4, beta=1.0, gamma=1, delta=0):
        super().__init__()
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.interpolate_n = 4

        interpolate_n = interpolate_n
        t = torch.linspace(0, 1, interpolate_n)
        # [(T,T),(T,T)]
        sample_weights = torch.meshgrid(t, t)
        # (T*T,3)
        sample_weights = torch.stack(sample_weights+((1-sample_weights[0]-sample_weights[1]),), dim=-1).view(-1,3)
        mask = (sample_weights[:,2]>=0).unsqueeze(-1).expand_as(sample_weights)
        # (S,3)
        self.sample_weights = torch.masked_select(sample_weights, mask).view(-1, 3)
        self.threshold = torch.nn.Hardshrink(0.05)

    def forward(self, cage_v, cage_f, shape, interpolate=True):
        B,M,D = cage_v.shape
        B,F,_ = cage_f.shape
        B,N,_ = shape.shape
        self.sample_weights = self.sample_weights.to(device=shape.device)

        # sample points using interpolated barycentric weights on cage triangles (B,F,1,3,3)
        cage_face_vertices = torch.gather(cage_v, 1, cage_f.reshape(B,F*3,1).expand(-1,-1,cage_v.shape[-1])).reshape(B,F,1,3,3)
        sample_weights = self.sample_weights.unsqueeze(0).unsqueeze(0).unsqueeze(-1).to(device=cage_v.device)  # (1,1,S,3,1)
        # (B,F,S,3)
        cage_sampled_points = torch.sum(sample_weights*cage_face_vertices, dim=-2).reshape(B,-1,3)
        cage2shape, shape2cage, _, _ = nndistance(cage_sampled_points, shape)
        shape2cage = self.threshold(shape2cage)
        cage2shape = self.threshold(cage2shape)
        loss = torch.mean(shape2cage, dim=1)*(self.gamma+self.delta*M) + torch.mean(cage2shape, dim=1) + self.beta*torch.max(cage2shape, dim=1)[0]
        loss = torch.mean(loss)
        return loss


class InsideLoss3DTriMesh(torch.nn.Module):
    """Penalize cage inside a triangle mesh
    Args:
        cage_v: (B,M,3)
        cage_f: (B,F,3)
        shape:  (B,N,3)
        shape_f: (B,FF,3)
        shape_fn: (B,FF,3)
    """
    def __init__(self, reduction="mean", interpolate_n=4):
        super().__init__()
        self.reduction = reduction
        interpolate_n = interpolate_n
        t = torch.linspace(0, 1, interpolate_n)
        # [(T,T),(T,T)]
        sample_weights = torch.meshgrid(t, t)
        # (T*T,3)
        sample_weights = torch.stack(sample_weights+((1-sample_weights[0]-sample_weights[1]),), dim=-1).view(-1,3)
        mask = (sample_weights[:,2]>=0).unsqueeze(-1).expand_as(sample_weights)
        # (S,3)
        self.sample_weights = torch.masked_select(sample_weights, mask).view(-1, 3)

    def forward(self, cage_v, cage_f, shape, shape_vn, epsilon=0.01, interpolate=True):
        B,M,D = cage_v.shape
        B,F,_ = cage_f.shape
        B,N,_ = shape.shape
        self.sample_weights = self.sample_weights.to(device=shape.device)
        # B,FF,_ = shape_f.shape
        # sample points using interpolated barycentric weights on cage triangles (B,F,1,3,3)
        cage_face_vertices = torch.gather(cage_v, 1, cage_f.reshape(B,F*3,1).expand(-1,-1,cage_v.shape[-1])).reshape(B,F,1,3,3)
        sample_weights = self.sample_weights.unsqueeze(0).unsqueeze(0).unsqueeze(-1).to(device=cage_v.device)  # (1,1,S,3,1)
        # (B,F,S,3)
        cage_sampled_points = torch.sum(sample_weights*cage_face_vertices, dim=-2).reshape(B,-1,3)

        # shape_face_vertices = torch.gather(shape, 1, shape_f.view(B,F*3,1)).view(B,F,3,3)

        # find the closest point on the shape
        nn_point, nn_index, _ = faiss_knn(1, cage_sampled_points, shape, NCHW=False)
        nn_point = nn_point.squeeze(2)
        # (B,FS,1)
        nn_normal = torch.gather(
            shape_vn.unsqueeze(1).expand(-1, nn_index.shape[1], -1, -1), 2,
            nn_index.unsqueeze(-1).expand(-1,-1,-1,shape_vn.shape[-1]))
        nn_normal = nn_normal.squeeze(2)

        # if <(q-p), n> is negative, then this point is inside the shape, gradient is along the normal direction
        dot = dot_product(cage_sampled_points - nn_point - epsilon*nn_normal, nn_normal, dim=-1)
        loss = torch.where(dot < 0, -dot, torch.zeros_like(dot))

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "max":
            return torch.mean(torch.max(loss, dim=-1)[0])
        elif self.reduction == "sum":
            return loss.mean(torch.sum(loss, dim=-1))
        elif self.reduction == "none":
            return loss
        else:
            raise NotImplementedError
        return loss



class MeshDihedralAngleLoss(torch.nn.Module):
    """
    if vert1 and vert both given, penalize difference of the dihedral angle between vert1 and vert2
    otherwise penalize if dehedral angle < pi/4
    vert1           (B,N,3)
    vert2           (B,N,3)
    edge_points     List(torch.Tensor(E, 4))
    """
    def __init__(self, threshold=np.pi/6, edge_points=None, reduction="mean"):
        super().__init__()
        self.edge_points = edge_points
        self.reduction = reduction
        self.threshold = threshold

    def forward(self, vert1, vert2=None, edge_points=None):
        if edge_points is None:
            edge_points = self.edge_points
        assert(edge_points is not None)
        B = vert1.shape[0]
        loss = []
        for b in range(B):
            angles1 = dihedral_angle(vert1[b], edge_points)
            if vert2 is not None:
                angles2 = dihedral_angle(vert2[b], edge_points)
                tmp = self.metric(angles1, angles2)
            else:
                tmp = torch.nn.functional.relu(np.pi/4 - angles1)
                tmp = tmp*tmp
                tmp = torch.mean(tmp)

            loss.append(tmp)

        loss = torch.stack(loss, dim=0)
        if self.reduction != "none":
            loss = loss.mean()

        return loss

class GTNormalLoss(torch.nn.Module):
    """
    compare the PCA normals of two point clouds
    ===
    params:
        NCHW: order of dimensions, default True
        pred: (B,3,N) if NCHW, (B,N,3) otherwise
    """
    def __init__(self, nn_size=10, NCHW=True):
        super().__init__()
        self.nn_size = nn_size
        self.NCHW = NCHW
        self.cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-08)

    def forward(self, pred, gt_normals):
        pred_normals = batch_normals(pred, nn_size=10, NCHW=self.NCHW)
        cos = self.cos(pred_normals, gt_normals)
        return torch.mean(1-cos)


class MeshSmoothLoss(torch.nn.Module):
    """
    compare laplacian of two meshes with the same connectivity assuming known correspondence
    metric: an instance of a module e.g. L1Loss
    use_cot: cot laplacian is used instead of uniformlaplacian
    consistent_topology: assume face matrix is the same during the entire use
    precompute_L: assume vert1 is always the same
    """
    def __init__(self, metric, use_cot=False, use_norm=False):
        super().__init__()
        if use_cot:
            self.laplacian = CotLaplacian()
        else:
            self.laplacian = UniformLaplacian()
        self.metric = metric

    def forward(self, vert1, face=None):
        lap1 = self.laplacian(vert1, face)
        lap1 = torch.norm(lap1, dim=-1, p=2)
        return lap1.mean()

class LocalFeatureLoss(torch.nn.Module):
    """
    penalize point to surface loss
    Given points (B,N,3)
    1. find KNN and the center
    2. fit PCA, get normal
    3. project p-center to normal
    """
    def __init__(self, nn_size=10, metric=torch.nn.MSELoss("mean"), **kwargs):
        super().__init__()
        self.nn_size = nn_size
        self.metric = metric

    def forward(self, xyz1, xyz2, **kwargs):
        xyz1 = xyz1.contiguous()
        xyz2 = xyz2.contiguous()
        B,N,C = xyz1.shape
        grouped_points, idx, _ = group_knn(self.nn_size, xyz1, xyz1, unique=True, NCHW=False)
        group_center = torch.mean(grouped_points, dim=2, keepdim=True)
        grouped_points = grouped_points - group_center
        # fit pca
        allpoints = grouped_points.view(-1, self.nn_size, C).contiguous()
        # BN,C,k
        U, S, V = batch_svd(allpoints)
        # V is BNxCxC, last_u BNxC
        normals = V[:, :, -1].view(B, N, C).detach()
        # FIXME what about the sign of normal
        ptof1 = dot_product((xyz1 - group_center.squeeze(2)), normals, dim=-1)

        # for xyz2 use the same neighborhood
        grouped_points = torch.gather(xyz2.unsqueeze(1).expand(-1,N,-1,-1), 2, idx.unsqueeze(-1).expand(-1,-1,-1,C))
        group_center = torch.mean(grouped_points, dim=2, keepdim=True)
        grouped_points = grouped_points - group_center
        allpoints = grouped_points.view(-1, self.nn_size, C).contiguous()
        # MB,C,k
        U, S, V = batch_svd(allpoints)
        # V is MBxCxC, last_u MBxC
        normals = V[:, :, -1].view(B, N, C).detach()
        ptof2 = dot_product((xyz2 - group_center.squeeze(2)), normals, dim=-1)
        # compare ptof1 and ptof2 absolute value (absolute value can only determine bent, not direction of bent)
        loss = self.metric(ptof1.abs(), ptof2.abs())
        # # penalize flat->curve
        bent = ptof2-ptof1
        bent.masked_fill_(bent<0, 0.0)
        bent = self.metric(bent, torch.zeros_like(bent))
        # bent.masked_fill_(bent<=1.0, 0.0)
        loss += 5*bent
        return loss
