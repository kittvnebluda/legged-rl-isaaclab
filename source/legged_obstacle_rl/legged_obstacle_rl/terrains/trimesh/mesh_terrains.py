import numpy as np
import trimesh
from isaaclab.terrains.trimesh.utils import make_plane


def diamond_walkway_terrain(difficulty: float, cfg) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a diamond-shaped walkway with a center horizontal beam."""

    beam_width = cfg.beam_width_range[1] + (1.0 - difficulty) * (cfg.beam_width_range[0] - cfg.beam_width_range[1])
    beam_height = cfg.beam_height

    meshes_list = [make_plane(cfg.size, 0.0, center_zero=False)]
    terrain_center = np.array([cfg.size[0] / 2.0, cfg.size[1] / 2.0, 0.0])

    half_x = cfg.size[0] / 2.0
    half_y = cfg.size[1] / 2.0
    diag_length = np.sqrt(half_x**2 + half_y**2)
    angle = np.arctan2(half_y, half_x)

    h_beam_dims = (cfg.size[0], beam_width, beam_height)
    h_beam_pos = (terrain_center[0], terrain_center[1], terrain_center[2] + beam_height / 2.0)
    h_beam = trimesh.creation.box(h_beam_dims, trimesh.transformations.translation_matrix(h_beam_pos))
    meshes_list.append(h_beam)

    diag_dims = (diag_length, beam_width, beam_height)

    orientations = [
        ([half_x / 2.0, 3 * half_y / 2.0], angle),  # Top-Left
        ([3 * half_x / 2.0, 3 * half_y / 2.0], -angle),  # Top-Right
        ([half_x / 2.0, half_y / 2.0], -angle),  # Bottom-Left
        ([3 * half_x / 2.0, half_y / 2.0], angle),  # Bottom-Right
    ]

    for pos_xy, rot_z in orientations:
        matrix = trimesh.transformations.translation_matrix([pos_xy[0], pos_xy[1], beam_height / 2.0])
        matrix = trimesh.transformations.concatenate_matrices(
            matrix, trimesh.transformations.rotation_matrix(rot_z, [0, 0, 1])
        )
        diag_mesh = trimesh.creation.box(diag_dims, matrix)
        meshes_list.append(diag_mesh)

    origin = np.array([terrain_center[0], terrain_center[1] + 1, beam_height])

    return meshes_list, origin
