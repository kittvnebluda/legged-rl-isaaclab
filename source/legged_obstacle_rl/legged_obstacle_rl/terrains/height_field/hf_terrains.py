from typing import TYPE_CHECKING

import numpy as np
from isaaclab.terrains.height_field.utils import height_field_to_mesh

if TYPE_CHECKING:
    from . import hf_terrains_cfg


@height_field_to_mesh
def random_square_holes_terrain(difficulty: float, cfg) -> np.ndarray:
    """Generate a flat terrain with randomly placed square holes.

    The depth of the holes is determined by the difficulty level, scaling between the
    provided min and max depth range.

    Args:
        difficulty: The difficulty of the terrain (0.0 to 1.0).
        cfg: The configuration for the terrain.

    Returns:
        The height field of the terrain as a 2D numpy array (int16).
    """
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)

    hole_depth = cfg.hole_depth_range[0] + difficulty * (cfg.hole_depth_range[1] - cfg.hole_depth_range[0])

    hole_width_pixels = int(cfg.hole_width / cfg.horizontal_scale)
    hole_depth_pixels = int(hole_depth / cfg.vertical_scale)

    hf_raw = np.zeros((width_pixels, length_pixels), dtype=np.float32)

    max_x = width_pixels - hole_width_pixels
    max_y = length_pixels - hole_width_pixels

    if max_x > 0 and max_y > 0:
        for _ in range(cfg.num_holes):
            start_x = np.random.randint(0, max_x)
            start_y = np.random.randint(0, max_y)
            hf_raw[start_x : start_x + hole_width_pixels, start_y : start_y + hole_width_pixels] = -hole_depth_pixels

    return np.rint(hf_raw).astype(np.int16)
