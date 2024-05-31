import time

import numpy as np
import json
import os.path

import torch

from .dataparser import ImageSet, PointCloud, DataParser, DataParserOutputs
from visualonfigs.dataset import BlenderParams
from visualameras.cameras import Cameras
from visualtils.graphics_utils import fov2focal, getNerfppNorm
from visualtils.sh_utils import SH2RGB


class BlenderDataParser(DataParser):
    def __init__(self, path: str, output_path: str, global_rank: int, params: BlenderParams) -> None:
        super().__init__()
        self.path = path
        self.output_path = output_path
        self.global_rank = global_rank
        self.params = params

    def _parse_transforms_json(self, split: str) -> ImageSet:
        with open(os.path.join(self.path, "transforms_{}.json".format(split)), "r") as f:
            transforms = json.load(f)

        # if in reconstruction mode, merge val and test into train set
        if split == "train" and self.params.split_mode == "reconstruction":
            for i in ["val", "test"]:
                with open(os.path.join(self.path, "transforms_{}.json".format(i)), "r") as f:
                    transforms["frames"] += json.load(f)["frames"]

        # TODO: auto detect image size
        width = 800

        # parse extrinsic
        image_name_list = []
        image_path_list = []
        camera_to_world_list = []
        time_list = []
        for frame in transforms["frames"]:
            image_name_with_extension = "{}.png".format(frame["file_path"])
            image_name_list.append(os.path.basename(image_name_with_extension))
            image_path_list.append(os.path.join(self.path, image_name_with_extension))
            camera_to_world_list.append(frame["transform_matrix"])
            if "time" in frame:
                time_list.append(frame["time"])
            else:
                time_list.append(0.)
        camera_to_world = torch.tensor(camera_to_world_list, dtype=torch.float64)
        # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
        camera_to_world[:, :3, 1:3] *= -1
        world_to_camera = torch.linalg.inv(camera_to_world).to(torch.float)

        R = world_to_camera[:, :3, :3]
        T = world_to_camera[:, :3, 3]

        # parse focal length
        fx = torch.tensor(
            [fov2focal(fov=transforms["camera_angle_x"], pixels=width)],
            dtype=torch.float32,
        ).expand(R.shape[0])
        # TODO: allow different fy
        fy = torch.clone(fx)

        width = torch.tensor([width], dtype=torch.float32).expand(R.shape[0])
        # TODO: allow different height
        height = torch.clone(width)

        return ImageSet(
            image_names=image_name_list,
            image_paths=image_path_list,
            mask_paths=[None for _ in range(len(image_name_list))],
            cameras=Cameras(
                R=R,
                T=T,
                fx=fx,
                fy=fy,
                cx=width / 2,
                cy=height / 2,
                width=width,
                height=height,
                appearance_id=torch.zeros_like(width),
                normalized_appearance_id=torch.zeros_like(width),
                distortion_params=None,
                camera_type=torch.zeros_like(width),
                time=torch.tensor(time_list, dtype=torch.float),
            ),
        )

    def get_outputs(self) -> DataParserOutputs:
        # ply_path = os.path.join(self.path, "points3D.ply")
        # while os.path.exists(ply_path) is False:
        #     if self.global_rank == 0:
        #         # Since this data set has no colmap data, we start with random points
        #         num_pts = 100_000
        #         print(f"Generating random point cloud ({num_pts})...")
        #
        #         # We create random points inside the bounds of the synthetic Blender scenes
        #         xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        #         shs = np.random.random((num_pts, 3)) / 255.0
        #         # pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))
        #
        #         store_ply(ply_path + ".tmp", xyz, SH2RGB(shs) * 255)
        #         os.rename(ply_path + ".tmp", ply_path)
        #         break
        #     else:
        #         # waiting ply
        #         print("#{} waiting for {}".format(os.getpid(), ply_path))
        #         time.sleep(1)
        # pcd = fetch_ply(ply_path)

        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")

        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        # shs = np.random.random((num_pts, 3)) / 255.0
        # rgb = np.asarray(SH2RGB(shs) * 255, dtype=np.uint8)
        if self.params.random_point_color is True:
            rgb = np.asarray(np.random.random((num_pts, 3)) * 255, dtype=np.uint8)  # random rgb color will produce artifacts
        else:
            rgb = np.ones((num_pts, 3), dtype=np.uint8) * 127

        train_set = self._parse_transforms_json("train")

        # R_list = []
        # T_list = []
        # for i in train_set.cameras:
        #     R_list.append(i.R.T.numpy())
        #     T_list.append(i.T.numpy())
        # norm = getNerfppNorm(R_list=R_list, T_list=T_list)

        return DataParserOutputs(
            train_set=train_set,
            val_set=self._parse_transforms_json("val"),
            test_set=self._parse_transforms_json("test"),
            point_cloud=PointCloud(
                xyz=xyz,
                rgb=rgb,
            ),
            # camera_extent=norm["radius"],
            appearance_group_ids=None,
        )
