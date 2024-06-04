from typing import Tuple, Literal
from dataclasses import dataclass


@dataclass
class OptimizationParams:
    position_lr_init: float = 0.00016
    position_lr_final: float = 0.0000016
    position_lr_delay_mult: float = 0.01
    position_lr_max_steps: float = 30_000
    feature_lr: float = 0.0025
    feature_rest_lr_init: float = 0.0025 / 20.
    feature_rest_lr_final_factor: float = 0.1
    feature_rest_lr_max_steps: int = -1
    feature_extra_lr_init: float = 1e-3
    feature_extra_lr_final_factor: float = 0.1
    feature_extra_lr_max_steps: int = 30_000
    opacity_lr: float = 0.05
    scaling_lr: float = 0.005
    rotation_lr: float = 0.001
    percent_dense: float = 0.01
    lambda_dssim: float = 0.2
    densification_interval: int = 100
    opacity_reset_interval: int = 3000
    densify_from_iter: int = 500
    densify_until_iter: int = 15_000
    densify_grad_threshold: float = 0.0002

    spatial_lr_scale: float = -1  # auto calculate from camera poses if > 0

    rgb_diff_loss: Literal["l1", "l2"] = "l1"
