import os.path
from glob import glob

import numpy as np
import torch

from visualfigs.dataset import NSVFParams
from visualeras.cameras import Cameras
from visualls.sh_utils import SH2RGB
from visualls.graphics_utils import getNerfppNorm
from .dataparser import DataParser, DataParserOutputs, ImageSet, PointCloud


class NSVFDataParser(DataParser):
    def __init__(self, path: str, output_path: str, global_rank: int, params: NSVFParams):
        super().__init__()

        self.path = path
        self.output_path = output_path
        self.global_rank = global_rank
        self.params = params

    def _parse(self, intrinsics_matrix, split: str) -> ImageSet:
        # TODO: support custom file range
        if split == "train":
            prefixs = ["0_"]
            if split == "train" and self.params.split_mode == "reconstruction":
                prefixs += ["1_", "2_"]
        elif split == "val":
            prefixs = ["1_"]
        else:
            prefixs = ["2_"]

        # build filename list
        rgb_file_list = []
        pose_file_list = []
        for i in prefixs:
            rgb_file_list += sorted(list(glob(os.path.join(self.path, "rgb", "{}*.*".format(i)))))
            pose_file_list += sorted(list(glob(os.path.join(self.path, "pose", "{}*.*".format(i)))))
        image_name_list = [os.path.basename(i) for i in rgb_file_list]

        # parse camera extrinsic
        pose_list = []
        for pose_file in pose_file_list:
            pose_list.append(self.parse_extrinsics(self.load_matrix(pose_file), world2camera=False))
        camera_to_world = np.asarray(pose_list)
        world_to_camera = np.linalg.inv(camera_to_world)
        world_to_camera = torch.tensor(world_to_camera, dtype=torch.float)
        R = world_to_camera[:, :3, :3]
        T = world_to_camera[:, :3, 3]

        # build camera intrinsics
        fx = torch.tensor([intrinsics_matrix[0, 0]], dtype=torch.float32).expand(R.shape[0])
        fy = torch.tensor([intrinsics_matrix[1, 1]], dtype=torch.float32).expand(R.shape[0])
        cx = torch.tensor([intrinsics_matrix[0, 2]], dtype=torch.float32).expand(R.shape[0])
        cy = torch.tensor([intrinsics_matrix[1, 2]], dtype=torch.float32).expand(R.shape[0])

        # TODO: allow different image shape
        width = torch.tensor([800], dtype=torch.float32).expand(R.shape[0])
        height = torch.clone(width)

        cameras = Cameras(
            R=R,
            T=T,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            width=width,
            height=height,
            appearance_id=torch.zeros_like(width),
            normalized_appearance_id=torch.zeros_like(width),
            distortion_params=None,
            camera_type=torch.zeros_like(width),
        )

        return ImageSet(
            image_names=image_name_list,
            image_paths=rgb_file_list,
            mask_paths=None,
            cameras=cameras,
        )

    def get_outputs(self) -> DataParserOutputs:
        intrinsics_matrix = self.load_intrinsics()
        bbox = np.asarray(self.load_bbox())
        xyz_min, xyz_max = bbox[:3], bbox[3:6]
        xyz_center = (xyz_min + xyz_max) / 2
        bbox_size = np.max(xyz_max - xyz_min)

        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")

        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = (np.random.random((num_pts, 3)) - 0.5) * bbox_size + xyz_center
        if self.params.random_point_color is True:
            rgb = np.asarray(np.random.random((num_pts, 3)) * 255, dtype=np.uint8)  # random rgb color will produce artifacts
        else:
            rgb = np.ones((num_pts, 3), dtype=np.uint8) * 127

        train_set = self._parse(intrinsics_matrix, "train")

        # R_list = []
        # T_list = []
        # for i in train_set.cameras:
        #     R_list.append(i.R.numpy())
        #     T_list.append(i.T.numpy())
        # norm = getNerfppNorm(R_list=R_list, T_list=T_list)

        return DataParserOutputs(
            train_set=train_set,
            val_set=self._parse(intrinsics_matrix, "val"),
            test_set=self._parse(intrinsics_matrix, "test"),
            point_cloud=PointCloud(
                xyz=xyz,
                rgb=rgb,
            ),
            # camera_extent=norm["radius"],
            appearance_group_ids=None,
        )

    def load_bbox(self):
        with open(os.path.join(self.path, "bbox.txt"), "r") as f:
            return [float(w) for w in f.read().strip().split()]

    @classmethod
    def load_matrix(cls, path):
        with open(path, "r") as f:
            lines = [[float(w) for w in line.strip().split()] for line in f]
        if len(lines[0]) == 2:
            lines = lines[1:]
        if len(lines[-1]) == 2:
            lines = lines[:-1]
        return np.array(lines).astype(np.float32)

    def load_intrinsics(self, resized_width=None, invert_y=False):
        filepath = os.path.join(self.path, "intrinsics.txt")
        try:
            intrinsics = self.load_matrix(filepath)
            if intrinsics.shape[0] == 3 and intrinsics.shape[1] == 3:
                _intrinsics = np.zeros((4, 4), np.float32)
                _intrinsics[:3, :3] = intrinsics
                _intrinsics[3, 3] = 1
                intrinsics = _intrinsics
            if intrinsics.shape[0] == 1 and intrinsics.shape[1] == 16:
                intrinsics = intrinsics.reshape(4, 4)
            return intrinsics
        except ValueError:
            pass

        # Get camera intrinsics
        with open(filepath, 'r') as file:

            f, cx, cy, _ = map(float, file.readline().split())
        fx = f
        if invert_y:
            fy = -f
        else:
            fy = f

        # Build the intrinsic matrices
        full_intrinsic = np.array([[fx, 0., cx, 0.],
                                   [0., fy, cy, 0],
                                   [0., 0, 1, 0],
                                   [0, 0, 0, 1]])
        return full_intrinsic

    @classmethod
    def parse_extrinsics(cls, extrinsics, world2camera=True):
        """ this function is only for numpy for now"""
        if extrinsics.shape[0] == 3 and extrinsics.shape[1] == 4:
            extrinsics = np.vstack([extrinsics, np.array([[0, 0, 0, 1.0]])])
        if extrinsics.shape[0] == 1 and extrinsics.shape[1] == 16:
            extrinsics = extrinsics.reshape(4, 4)
        if world2camera:
            extrinsics = np.linalg.inv(extrinsics).astype(np.float32)
        return extrinsics
