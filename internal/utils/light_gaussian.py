from typing import List, Tuple
import torch
from internal.renderers.gsplat_hit_pixel_count_renderer import GSplatHitPixelCountRenderer


def get_count_and_score(
        gaussian_model,
        cameras: List,
        anti_aliased: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    device = gaussian_model.get_xyz.device
    num_gaussians = gaussian_model.get_xyz.shape[0]

    # initialize accumulation tensors
    count_total = torch.zeros(
        num_gaussians,
        dtype=torch.int,
        device=device,
    )
    opacity_score_total = torch.zeros(
        num_gaussians,
        dtype=torch.float,
        device=device,
    )
    alpha_score_total = torch.zeros(
        num_gaussians,
        dtype=torch.float,
        device=device,
    )
    visibility_score_total = torch.zeros(
        num_gaussians,
        dtype=torch.float,
        device=device,
    )

    # count for each training camera
    for i in range(len(cameras)):
        camera = cameras[i]
        count, opacity_score, alpha_score, visibility_score = GSplatHitPixelCountRenderer.hit_pixel_count(
            means3D=gaussian_model.get_xyz,
            opacities=gaussian_model.get_opacity,
            scales=gaussian_model.get_scaling,
            rotations=gaussian_model.get_rotation,
            viewpoint_camera=camera.to_device(device),
            anti_aliased=anti_aliased,
        )
        # add to total
        count_total += count
        opacity_score_total += opacity_score
        alpha_score_total += alpha_score
        visibility_score_total += visibility_score

    return count_total, opacity_score_total, alpha_score_total, visibility_score_total


def calculate_v_imp_score(scales, importance_scores, v_pow):
    """
    Copied from LightGaussian: https://github.com/VITA-Group/LightGaussian

    :param scales: The scales of the 3D Gaussians, typically obtain via getter `model.get_scaling`.
    :param importance_scores: The importance scores for each Gaussian component.
    :param v_pow: The power to which the volume ratios are raised.
    :return: A list of adjusted values (v_list) used for pruning.
    """
    # Calculate the volume of each Gaussian component
    volume = torch.prod(scales, dim=1)
    # Determine the kth_percent_largest value
    index = int(len(volume) * 0.9)
    sorted_volume, _ = torch.sort(volume, descending=True)
    kth_percent_largest = sorted_volume[index]
    # Calculate v_list
    v_list = torch.pow(volume / kth_percent_largest, v_pow)
    v_list = v_list * importance_scores
    return v_list


def get_prune_mask(percent, importance_score):
    sorted_tensor, _ = torch.sort(importance_score, dim=0)
    index_nth_percentile = int(percent * (sorted_tensor.shape[0] - 1))
    value_nth_percentile = sorted_tensor[index_nth_percentile]
    prune_mask = (importance_score <= value_nth_percentile).squeeze()
    return prune_mask
