from dataclasses import MISSING

from isaaclab.managers import CommandTermCfg
from isaaclab.markers import SPHERE_MARKER_CFG
from isaaclab.utils import configclass

from .commands import UniformBodyHeightCommand


@configclass
class UniformBodyHeightCommandCfg(CommandTermCfg):
    class_type: type = UniformBodyHeightCommand

    asset_name: str = "robot"
    resampling_time_range: tuple[float, float] = (8.0, 12.0)
    debug_vis: bool = True

    marker_cfg = SPHERE_MARKER_CFG.replace(prim_path="/Visuals/Command/height_goal")

    @configclass
    class Ranges:
        height: tuple[float, float] = MISSING
        """Range for the height (in m)."""

    ranges: Ranges = MISSING
    """Distribution ranges for the height command."""
