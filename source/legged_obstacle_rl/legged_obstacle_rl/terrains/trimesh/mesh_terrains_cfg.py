from isaaclab.terrains import SubTerrainBaseCfg
from isaaclab.utils import configclass

from . import mesh_terrains


@configclass
class MeshDiamondWalkwayTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a diamond walkway mesh terrain."""

    function = mesh_terrains.diamond_walkway_terrain

    beam_width_range: tuple[float, float] = (0.4, 0.05)

    # Height of the beams above the ground (in meters)
    beam_height: float = 0.2

    # Size of the subterrain tile
    size: tuple[float, float] = (4.0, 4.0)
