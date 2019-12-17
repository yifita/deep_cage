from glob import glob
import os
import torch
import numpy as np
from pytorch_points.network.geo_operations import mean_value_coordinates, normalize
from pytorch_points.utils.pc_utils import load, save_ply, save_ply_with_face

def perturb(polygon, RADIUS_PERTURB, ANGLE_PERTURB):
    """
    vary polygon vertices
    Args:
        polygon (B,2,M)
        RADIUS_PERTURB perturb range in terms of ratio of the initial length
        ANGLE_PERTURB perturb range in rad
    """
    polygon_normalized = normalize(polygon, dim=1)
    angles = torch.atan2(polygon_normalized[:,:1,:],polygon_normalized[:,1:,:])
    angles_offset = torch.clamp(torch.randn_like(angles)*ANGLE_PERTURB/3, -ANGLE_PERTURB, ANGLE_PERTURB)
    angles = angles_offset+angles
    polygon_rad = torch.norm(polygon, dim=1, p=2, keepdim=True)
    polygon_rad = polygon_rad*(1+torch.clamp(torch.randn_like(polygon_rad)*RADIUS_PERTURB/3, -RADIUS_PERTURB, RADIUS_PERTURB))
    new_polygon = polygon_rad*torch.cat([torch.cos(angles), torch.sin(angles)], dim=1)
    return new_polygon

if __name__ == "__main__":
    torch.manual_seed(24)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(24)
    # pick source, deform it
    DATA_DIR = "/home/yifan/Data/mnist_contour/WLOP_100"
    DIGIT = 1
    DATA_ALL = sorted(glob(os.path.join(DATA_DIR, "{:d}".format(DIGIT), "*.ply")))
    ANGLE_PERTURB = np.pi/8
    RADIUS_PERTURB = 0.4
    PERTURB_ITER = 2
    PERTURB_EPOCH = 3000
    # SHAPE_IDX = np.random.randint(0, len(DATA_ALL), dtype=int)
    SOURCE_NAME = "gingerbreadman"
    # SOURCE_PATH = os.path.join(DATA_DIR, "{:d}".format(DIGIT), SOURCE_NAME+".ply")
    SOURCE_PATH = "/home/yifan/Documents/Cage/scripts/wlop/build/gingerbreadman.ply"
    CAGE_PATH = "/home/yifan/Documents/Cage/scripts/wlop/build/gingerbreadman_cage.ply"

    # polygon_list = [
    #     (-0.523185483870968,	0.553246753246753),
    #     (-0.644153225806452,	-0.101298701298701),
    #     (-0.166330645161290,	-0.218181818181818),
    #     (0.190524193548387,	-0.381818181818182),
    #     (0.450604838709678,	-0.553246753246754),
    #     (0.656250000000000,	0),
    #     (0.335685483870968,	0.225974025974026),
    #     (-0.154233870967742,	0.444155844155844),
    # ]

    # polygon = torch.tensor([(x, y) for x, y in polygon_list], dtype=torch.float).unsqueeze(0).transpose(1, 2)

    source = torch.tensor(load(SOURCE_PATH)[:,:2], dtype=torch.float).unsqueeze(0).transpose(1,2)
    save_ply(source[0].transpose(0,1).numpy(), "../vanilla_data/{}/{}.ply".format(SOURCE_NAME, SOURCE_NAME))
    polygon = torch.tensor(load(CAGE_PATH))[:,:2].unsqueeze(0).transpose(1,2)
    save_ply(polygon[0].transpose(0,1).numpy(), "../vanilla_data/{}/{}-cage.ply".format(SOURCE_NAME, SOURCE_NAME), binary=False)
    weights = mean_value_coordinates(source, polygon)
    # perturb
    for i in range(PERTURB_EPOCH):
        new_polygon = polygon
        for k in range(PERTURB_ITER):
            new_polygon = perturb(new_polygon, RADIUS_PERTURB, ANGLE_PERTURB)
            # (B,2,M,N) * (B,2,M,1) -> (B,2,N)
            deformed = torch.sum(weights.unsqueeze(1)*new_polygon.unsqueeze(-1), dim=2)
            save_ply(deformed[0].transpose(0,1), "../vanilla_data/{}/{}-{}.ply".format(SOURCE_NAME, SOURCE_NAME, i*PERTURB_ITER+k))
            save_ply(new_polygon[0].transpose(0,1).numpy(), "../vanilla_data/{}/{}-{}-cage.ply".format(SOURCE_NAME, SOURCE_NAME, i*PERTURB_ITER+k), binary=False)