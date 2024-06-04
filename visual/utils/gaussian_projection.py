from typing import Tuple
import torch
import struct


def project_gaussians(
        means_3d: torch.Tensor,  # [n, 3]
        scales: torch.Tensor,  # [n, 3]
        scale_modifier: float,
        quaternions: torch.Tensor,  # [n, 4]
        world_to_camera: torch.Tensor,  # [4, 4]
        fx: torch.Tensor,
        fy: torch.Tensor,
        cx: torch.Tensor,
        cy: torch.Tensor,
        img_height: torch.Tensor,
        img_width: torch.Tensor,
        block_width: int,
        min_depth: float = 0.01,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Args
       means_3d (Tensor): xyzs of gaussians.
       scales (Tensor): scales of the gaussians.
       scale_modifier (float): A global scaling factor applied to the scene.
       quaternions (Tensor): rotations in quaternion [w,x,y,z] format.
       world_to_camera (Tensor): world-to-camera transform matrix (transposed, the translation vector located at the last row)
       fx (float): focal length x.
       fy (float): focal length y.
       cx (float): principal point x.
       cy (float): principal point y.
       img_height (int): height of the rendered image.
       img_width (int): width of the rendered image.
       block_width (int): side length of tiles inside projection/rasterization in pixels (always square). 16 is a good default value, must be between 2 and 16 inclusive.
       min_depth (float): minimum z depth threshold.
    """

    block_height = block_width

    # transform from world space to camera space
    means_3d_in_camera_space = torch.matmul(means_3d, world_to_camera[:3, :3]) + world_to_camera[3, :3]
    with torch.no_grad():
        is_min_depth_satisfied = means_3d_in_camera_space[:, 2] >= min_depth
    # means_3d_in_camera_space_min_depth_satisfied = means_3d_in_camera_space[is_min_depth_satisfied]

    # calculate 3D covariance matrix
    cov_3d = compute_cov_3d(scales, scale_modifier, quaternions=quaternions)

    # calculate 2D covariance matrix
    max_x_on_normalized_plane = (0.5 * img_width) / fx
    max_y_on_normalized_plane = (0.5 * img_height) / fy
    cov_2d = compute_cov_2d(
        means_3d_in_camera_space,
        tan_fovx=max_x_on_normalized_plane,
        tan_fovy=max_y_on_normalized_plane,
        focal_x=fx,
        focal_y=fy,
        cov_3d=cov_3d,
        world_to_camera=world_to_camera,
    )
    # compute 2D covariance determinant
    cov_2d_det_orig = torch.linalg.det(cov_2d)
    # Apply low-pass filter: every Gaussian should be at least
    # one pixel wide/high. Discard 3rd row and column.
    cov_2d_00 = cov_2d[:, 0, 0] + 0.3
    cov_2d_11 = cov_2d[:, 1, 1] + 0.3
    cov_2d = torch.stack([cov_2d_00, cov_2d[:, 0, 1], cov_2d[:, 1, 0], cov_2d_11], dim=-1).reshape((cov_2d.shape[0], 2, 2))
    # compute new determinant
    cov_2d_det = torch.linalg.det(cov_2d)
    if torch.any(cov_2d_det == 0):
        raise RuntimeError("zero determinant cov_2d found")
    # compute compensation factor
    compensation = torch.sqrt(torch.clamp_min(cov_2d_det_orig / cov_2d_det, 0.))

    # invert 2D covariance matrix (conic)
    inv_det = 1. / cov_2d_det
    conic = torch.concat([
        (cov_2d[:, 1, 1] * inv_det).unsqueeze(-1),
        (-cov_2d[:, 0, 1] * inv_det).unsqueeze(-1),
        (cov_2d[:, 0, 0] * inv_det).unsqueeze(-1),
    ], dim=-1)

    # transform means 3D to image plane
    ## project through camera intrinsics
    means_3d_on_normalized_plane = means_3d_in_camera_space / (means_3d_in_camera_space[:, 2:] + 1e-6)
    ## build intrinsics matrix
    ## 0.5 offset should be added here to reach same quality as NDC projection,
    ## but the rasterizer will do it, so just simply use original cx and cy
    intrinsics_matrix = torch.tensor([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1],
    ], dtype=means_3d_on_normalized_plane.dtype, device=means_3d_on_normalized_plane.device)
    means_2d_on_image_plane = torch.matmul(means_3d_on_normalized_plane, intrinsics_matrix.T)

    # compute gaussian extent in screen space
    ## calculate two eigenvalues of the 2D covariance matrix
    mid = 0.5 * (cov_2d[:, 0, 0] + cov_2d[:, 1, 1])
    mid_squared = mid * mid
    mid_squared_and_det_diff = mid_squared[:, None] - cov_2d_det[:, None]
    clamped_mid_squared_and_det_diff = torch.clamp_min(mid_squared_and_det_diff, 0.1)
    sqrt_diff = torch.sqrt(clamped_mid_squared_and_det_diff)
    ## two eigenvalues, [n, 1]
    lambda1 = mid[:, None] + sqrt_diff
    lambda2 = mid[:, None] - sqrt_diff
    ## use the maximum one as the radius
    radius = torch.ceil(3. * torch.sqrt(torch.maximum(lambda1, lambda2))).int()  # [n, 1]

    # calculate touched tiles
    tile_grid = torch.tensor([
        (img_width + block_width - 1) // block_width,
        (img_height + block_height - 1) // block_height,
        1,
    ], device=radius.device).long()
    block = torch.tensor([block_width, block_height, 1], dtype=torch.int, device=radius.device)
    block_xy = block[0:2].unsqueeze(0)  # [1, 2[xy]]
    ## get rect, inclusive min, exclusive max (differ to the vanilla version)
    rect_min = ((means_2d_on_image_plane[:, 0:2] - radius) / block_xy).int()
    # rect_max = ((means_2d_on_image_plane[:, 0:2] + radius + block_xy - 1) / block_xy).int()  # this will report CUDA memory access error
    rect_max = ((means_2d_on_image_plane[:, 0:2] + radius) / block_xy).int() + 1
    for rect in [rect_min, rect_max]:
        for i in range(2):
            rect[:, i] = torch.clamp(rect[:, i], min=0, max=tile_grid[i])
    rect_diff = rect_max - rect_min
    touched_tile_count = rect_diff[:, 0] * rect_diff[:, 1]

    mask = torch.logical_and(is_min_depth_satisfied, touched_tile_count > 0)
    invert_mask = ~mask
    radii = torch.where(invert_mask, 0, radius.squeeze(-1))
    conic = torch.where(invert_mask[..., None], 0, conic)
    xys = torch.where(invert_mask[..., None], 0, means_2d_on_image_plane[:, 0:2])
    cov3d = torch.where(invert_mask[..., None, None], 0, cov_3d)
    # cov2d = torch.where(invert_mask[..., None, None], 0, cov_2d)
    compensation = torch.where(invert_mask, 0, compensation)
    num_tiles_hit = torch.where(invert_mask, 0, touched_tile_count)
    depths = torch.where(invert_mask, 0, means_3d_in_camera_space[:, 2])

    return xys, depths, radii, conic, compensation, num_tiles_hit, cov3d, mask, rect_min, rect_max

    # # build masks
    # with torch.no_grad():
    #     is_touched_any_tiles_after_min_depth_filter = touched_tile_count > 0
    #     min_depth_satisfied_indices = torch.nonzero(is_min_depth_satisfied).squeeze(-1)
    #     touched_tile_indices = min_depth_satisfied_indices[is_touched_any_tiles_after_min_depth_filter]
    #     is_touched_any_tiles = torch.zeros((means_3d.shape[0],), dtype=torch.bool, device=means_3d.device)
    #     is_touched_any_tiles[touched_tile_indices] = True
    #
    # # xys[n, 2], depths[n], radii[n], conics[n, 3], comp[n], num_tiles_hit[n], cov3d[n, 6], mask[n]
    # return means_2d_on_image_plane[is_touched_any_tiles_after_min_depth_filter][:, :2], \
    #     means_3d_in_camera_space_min_depth_satisfied[is_touched_any_tiles_after_min_depth_filter, 2], \
    #     radius[is_touched_any_tiles_after_min_depth_filter].squeeze(-1), \
    #     conic[is_touched_any_tiles_after_min_depth_filter], \
    #     compensation[is_touched_any_tiles_after_min_depth_filter], \
    #     touched_tile_count[is_touched_any_tiles_after_min_depth_filter], \
    #     cov_3d[is_touched_any_tiles_after_min_depth_filter], \
    #     is_touched_any_tiles


def build_tile_bounds(
        img_height: torch.Tensor,
        img_width: torch.Tensor,
        block_width: int,
        device,
):
    block_height = block_width
    return torch.tensor([
        (img_width + block_width - 1) // block_width,
        (img_height + block_height - 1) // block_height,
        1,
    ], device=device).int()


def build_gaussian_sort_key(
        depths: torch.Tensor,  # [n]
        rect_min: torch.Tensor,  # [n, 2-xy]
        rect_max: torch.Tensor,  # [n, 2-xy]
        tile_bounds: torch.Tensor,  # [3]
        cumsum_tiles_hit: torch.Tensor,  # [n]
):
    total_tiles_hit = cumsum_tiles_hit[-1].item()
    sort_key = torch.zeros((total_tiles_hit,), dtype=torch.int64, device=depths.device)
    gaussian_ids = torch.zeros((total_tiles_hit,), dtype=torch.int32, device=depths.device)

    # the cumsum of the previous gaussian is the base index
    base_index_list = torch.concat([
        torch.tensor([0], dtype=cumsum_tiles_hit.dtype, device=cumsum_tiles_hit.device),
        cumsum_tiles_hit,
    ], dim=0)

    for gaussian_idx in range(depths.shape[0]):
        # Get raw byte representation of the float value at the given index
        raw_bytes = struct.pack("f", depths[gaussian_idx])

        # Interpret those bytes as an int32_t
        depth_id_n = struct.unpack("i", raw_bytes)[0]

        index = base_index_list[gaussian_idx].item()

        for i in range(rect_min[gaussian_idx][1], rect_max[gaussian_idx][1]):
            row_tile_id_offset = tile_bounds[0] * i
            for j in range(rect_min[gaussian_idx][0], rect_max[gaussian_idx][0]):
                tile_id = row_tile_id_offset + j
                sort_key[index] = (tile_id << 32) | depth_id_n
                gaussian_ids[index] = gaussian_idx

                index += 1

    return sort_key, gaussian_ids


def build_rotation_matrix(quaternions):
    # normalize quaternion
    quaternion_norm = torch.norm(quaternions, dim=-1)
    normalized_quaternion = quaternions / quaternion_norm[:, None]
    # build rotation matrix
    r = normalized_quaternion[:, 0]
    x = normalized_quaternion[:, 1]
    y = normalized_quaternion[:, 2]
    z = normalized_quaternion[:, 3]
    rotation_matrix = torch.zeros((quaternions.shape[0], 3, 3), dtype=quaternions.dtype, device=quaternions.device)
    rotation_matrix[:, 0, 0] = 1 - 2 * (y * y + z * z)
    rotation_matrix[:, 0, 1] = 2 * (x * y - r * z)
    rotation_matrix[:, 0, 2] = 2 * (x * z + r * y)
    rotation_matrix[:, 1, 0] = 2 * (x * y + r * z)
    rotation_matrix[:, 1, 1] = 1 - 2 * (x * x + z * z)
    rotation_matrix[:, 1, 2] = 2 * (y * z - r * x)
    rotation_matrix[:, 2, 0] = 2 * (x * z - r * y)
    rotation_matrix[:, 2, 1] = 2 * (y * z + r * x)
    rotation_matrix[:, 2, 2] = 1 - 2 * (x * x + y * y)

    return rotation_matrix


def compute_cov_3d(scales, scale_modifier, quaternions):
    """
    :param scales:
    :param scale_modifier:
    :param quaternions: in wxyz
    :return:
    """
    n_gaussians, n_scales = scales.shape
    # build scaling matrix
    scaling_matrix = torch.zeros((n_gaussians, n_scales, n_scales), dtype=scales.dtype, device=scales.device)
    for i in range(n_scales):
        scaling_matrix[:, i, i] = scales[:, i] * scale_modifier

    # build rotation matrix
    rotation_matrix = build_rotation_matrix(quaternions)

    m = torch.bmm(rotation_matrix, scaling_matrix)
    sigma = torch.matmul(m, m.transpose(1, 2))

    return sigma


def compute_cov_2d(t, tan_fovx, tan_fovy, focal_x, focal_y, cov_3d, world_to_camera):
    limx = 1.3 * tan_fovx
    limy = 1.3 * tan_fovy
    txtz = t[:, 0] / t[:, 2]
    tytz = t[:, 1] / t[:, 2]

    clamped_x = torch.clamp(txtz, min=-limx, max=limx) * t[:, 2]
    clamped_y = torch.clamp(tytz, min=-limy, max=limy) * t[:, 2]

    means_in_camera_space = torch.stack([clamped_x, clamped_y, t[:, 2]], dim=-1)

    # build Jacobian matrix J
    J = torch.zeros(
        (means_in_camera_space.shape[0], 3, 3),
        dtype=means_in_camera_space.dtype,
        device=means_in_camera_space.device
    )
    J[:, 0, 0] = focal_x / means_in_camera_space[:, 2]
    J[:, 0, 2] = -(focal_x * means_in_camera_space[:, 0]) / (means_in_camera_space[:, 2] * means_in_camera_space[:, 2])
    J[:, 1, 1] = focal_y / means_in_camera_space[:, 2]
    J[:, 1, 2] = -(focal_y * means_in_camera_space[:, 1]) / (means_in_camera_space[:, 2] * means_in_camera_space[:, 2])
    # the third row of J is ignored

    # build transform matrix W
    W = world_to_camera[:3, :3].T

    T = torch.matmul(J, W[None, :])

    cov_2d = T @ cov_3d @ T.transpose(1, 2)

    return cov_2d[:, :2, :2]
