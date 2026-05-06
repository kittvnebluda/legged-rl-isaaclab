from isaaclab.terrains import HfTerrainBaseCfg
from isaaclab.utils import configclass

from . import hf_terrains


@configclass
class HfRandomSquareHolesTerrainCfg(HfTerrainBaseCfg):
    # Function pointer to the generator above
    function = hf_terrains.random_square_holes_terrain

    # Square dimensions (in meters)
    hole_width: float = 0.3

    # Number of holes to scatter
    num_holes: int = 20

    # Depth range [min, max] (in meters)
    # At difficulty 0, depth is 0.05m; at difficulty 1, depth is 0.25m
    hole_depth_range: tuple[float, float] = (0.1, 0.3)
