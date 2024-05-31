from typing import Tuple, Optional
from dataclasses import dataclass

import numpy as np
import torch

from internal.cameras.cameras import Cameras


@dataclass
class ImageSet:
    image_names: list

    image_paths: list
    """ Full path to the image file """

    cameras: Cameras
    """ Camera intrinscis and extrinsics """

    depth_paths: Optional[list] = None
    """ Full path to the depth file """

    mask_paths: Optional[list] = None
    """ Full path to the mask file """

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        return self.image_names[index], self.image_paths[index], self.mask_paths[index], self.cameras[index]

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __post_init__(self):
        if self.mask_paths is None:
            self.mask_paths = [None for _ in range(len(self.image_paths))]


@dataclass
class PointCloud:
    xyz: np.ndarray

    rgb: np.ndarray


@dataclass
class DataParserOutputs:
    train_set: ImageSet

    val_set: ImageSet

    test_set: ImageSet

    point_cloud: PointCloud

    # ply_path: str

    appearance_group_ids: Optional[dict]

    camera_extent: Optional[float] = None

    def __post_init__(self):
        if self.camera_extent is None:
            camera_centers = self.train_set.cameras.camera_center
            average_camera_center = torch.mean(camera_centers, dim=0)
            camera_distance = torch.linalg.norm(camera_centers - average_camera_center, dim=-1)
            max_distance = torch.max(camera_distance)
            self.camera_extent = float(max_distance * 1.1)


class DataParser:
    def get_outputs(self) -> DataParserOutputs:
        """
        :return: [training set, validation set, point cloud]
        """

        pass
