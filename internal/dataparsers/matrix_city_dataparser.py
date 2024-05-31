import os
import math
import json
import numpy as np
import torch

from typing import Tuple
from PIL import Image
from tqdm import tqdm
from internal.configs.dataset import MatrixCityParams
from internal.cameras.cameras import Cameras
from .dataparser import ImageSet, PointCloud, DataParser, DataParserOutputs


class MatrixCityDataParser(DataParser):
    def __init__(self, path: str, output_path: str, global_rank: int, params: MatrixCityParams) -> None:
        super().__init__()
        self.path = path
        self.output_path = output_path
        self.global_rank = global_rank
        self.params = params

    def _parse_json(self, paths, build_point_cloud: bool = False) -> Tuple[ImageSet, PointCloud]:
        image_names = []
        image_paths = []
        depth_paths = []
        c2w_tensor_list = []
        R_tensor_list = []
        T_tensor_list = []
        fx_tensor_list = []
        fy_tensor_list = []
        cx_tensor_list = []
        cy_tensor_list = []
        width_tensor_list = []
        height_tensor_list = []
        for json_relative_path in paths:
            path = os.path.join(self.path, json_relative_path)
            with open(path, "r") as f:
                transforms = json.load(f)

            fov_x = transforms["camera_angle_x"]

            # get image shape, assert all image use same shape
            base_dir = os.path.dirname(path)
            if "path" in transforms["frames"][0]:
                base_dir = os.path.join(base_dir, transforms["frames"][0]["path"])
            image = Image.open(os.path.join(
                base_dir,
                "rgb",
                "{:04d}.png".format(0),
            ))
            width = image.width
            height = image.height
            image.close()

            # build image name list and camera poses
            c2w_list = []
            for frame in transforms["frames"]:
                # TODO: load fov provided by frame
                frame_id = frame["frame_index"]
                image_names.append("{:04d}".format(frame_id))
                base_dir = os.path.dirname(path)
                if "path" in frame:
                    base_dir = os.path.join(base_dir, frame["path"])
                image_paths.append(os.path.join(
                    base_dir,
                    "rgb",
                    "{:04d}.png".format(frame_id),
                ))
                depth_paths.append(os.path.join(
                    base_dir,
                    "depth",
                    "{:04d}.exr".format(frame_id),
                ))

                c2w = torch.tensor(frame['rot_mat'], dtype=torch.float64)
                c2w_list.append(c2w)

            # convert c2w to w2c
            camera_to_world = torch.stack(c2w_list)
            camera_to_world[:, :3, :3] *= 100
            camera_to_world[:, :3, 3] *= self.params.scale
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            camera_to_world[:, :3, 1:3] *= -1
            c2w_tensor_list.append(camera_to_world)
            world_to_camera = torch.linalg.inv(camera_to_world).to(torch.float)

            # extract R and T from w2c
            R = world_to_camera[:, :3, :3]
            T = world_to_camera[:, :3, 3]

            # calculate camera intrinsics
            fx = torch.tensor([.5 * width / np.tan(.5 * fov_x)]).expand(R.shape[0])
            fy = fx
            cx = torch.tensor([width / 2]).expand(R.shape[0])
            cy = torch.tensor([height / 2]).expand(R.shape[0])
            width = torch.tensor([width], dtype=torch.float).expand(R.shape[0])
            height = torch.tensor([height], dtype=torch.float).expand(R.shape[0])

            # append to list
            R_tensor_list.append(R)
            T_tensor_list.append(T)
            fx_tensor_list.append(fx)
            fy_tensor_list.append(fy)
            cx_tensor_list.append(cx)
            cy_tensor_list.append(cy)
            width_tensor_list.append(width)
            height_tensor_list.append(height)

        width = torch.concat(width_tensor_list, dim=0)
        cameras = Cameras(
            R=torch.concat(R_tensor_list, dim=0),
            T=torch.concat(T_tensor_list, dim=0),
            fx=torch.concat(fx_tensor_list, dim=0),
            fy=torch.concat(fy_tensor_list, dim=0),
            cx=torch.concat(cx_tensor_list, dim=0),
            cy=torch.concat(cy_tensor_list, dim=0),
            width=width,
            height=torch.concat(height_tensor_list, dim=0),
            appearance_id=torch.zeros_like(width),
            normalized_appearance_id=torch.zeros_like(width),
            distortion_params=None,
            camera_type=torch.zeros_like(width),
        )
        if build_point_cloud is True:
            import open3d as o3d
            import hashlib
            import dataclasses

            # check whether need to regenerate point cloud based on params
            params_dict = dataclasses.asdict(self.params)
            del params_dict["test"]  # ignore test set
            params_json = json.dumps(params_dict, indent=4, ensure_ascii=False)
            print(params_json)
            ply_file_path = os.path.join(
                self.path,
                "{}.ply".format(hashlib.sha1(params_json.encode("utf-8")).hexdigest()),
            )
            if os.path.exists(ply_file_path):
                final_pcd = o3d.io.read_point_cloud(ply_file_path)
                point_cloud = PointCloud(
                    np.asarray(final_pcd.points),
                    (np.asarray(final_pcd.colors) * 255).astype(np.uint8),
                )
            else:
                c2w = torch.concat(c2w_tensor_list, dim=0)

                os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
                import cv2

                points_per_image = math.ceil(self.params.max_points / (len(image_paths) // self.params.depth_read_step))

                def read_depth(path: str, scale: float):
                    return cv2.imread(
                        path,
                        cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH,
                    )[..., 0] * scale

                def read_rgb(path: str):
                    image = cv2.imread(path)
                    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                xyz_list = []
                rgb_list = []
                with tqdm(range(len(image_paths) // self.params.depth_read_step), desc="Building point cloud") as t:
                    for frame_idx in t:
                        frame_idx = frame_idx * self.params.depth_read_step
                        t.set_description(image_paths[frame_idx])
                        # build intrinsics matrix
                        fx = cameras.fx[frame_idx]
                        fy = cameras.fy[frame_idx]
                        cx = cameras.cx[frame_idx]
                        cy = cameras.cy[frame_idx]
                        K = np.eye(3)
                        K[0, 2] = cx
                        K[1, 2] = cy
                        K[0, 0] = fx
                        K[1, 1] = fy

                        # build pixel coordination
                        width = int(cameras.width[frame_idx])
                        height = int(cameras.height[frame_idx])
                        image_pixel_count = width * height
                        u_coord = np.tile(np.arange(width), (height, 1)).reshape(image_pixel_count)
                        v_coord = np.tile(np.arange(height), (width, 1)).T.reshape(image_pixel_count)
                        p2d = np.array([u_coord, v_coord, np.ones_like(u_coord)]).T
                        homogenous_coordinate = np.matmul(p2d, np.linalg.inv(K).T)

                        # read rgb and depth
                        rgb = read_rgb(image_paths[frame_idx]).reshape((-1, 3))
                        depth = read_depth(depth_paths[frame_idx], self.params.scale * self.params.depth_scale).reshape((-1,))

                        # discard invalid depth
                        valid_depth_indices = np.where(depth < self.params.max_depth * self.params.scale * self.params.depth_scale)
                        rgb = rgb[valid_depth_indices]
                        depth = depth[valid_depth_indices]
                        homogenous_coordinate = homogenous_coordinate[valid_depth_indices]

                        # random sample
                        valid_pixel_count = rgb.shape[0]
                        if points_per_image < valid_pixel_count:
                            sample_indices = np.random.choice(valid_pixel_count, points_per_image, replace=False)
                            homogenous_coordinate = homogenous_coordinate[sample_indices]
                            rgb = rgb[sample_indices]
                            depth = depth[sample_indices]

                        # convert to world coordination
                        points_3d_in_camera = homogenous_coordinate * depth[:, None]
                        points_3d_in_camera[:, 1] *= -1
                        points_3d_in_camera[:, 2] *= -1
                        image_c2w = c2w[frame_idx].numpy()
                        image_c2w[:3, 1:3] *= -1
                        points_3d_in_world = np.matmul(points_3d_in_camera, image_c2w[:3, :3].T) + image_c2w[:3, 3]

                        xyz_list.append(points_3d_in_world)
                        rgb_list.append(rgb)

                point_cloud = PointCloud(
                    np.concatenate(xyz_list, axis=0),
                    np.concatenate(rgb_list, axis=0),
                )

                final_pcd = o3d.geometry.PointCloud()
                final_pcd.points = o3d.utility.Vector3dVector(point_cloud.xyz)
                final_pcd.colors = o3d.utility.Vector3dVector(point_cloud.rgb / 255.)
                o3d.io.write_point_cloud(ply_file_path, final_pcd)
                with open("{}.config.json".format(ply_file_path), "w") as f:
                    f.write(params_json)
        else:
            point_cloud = None
        return ImageSet(
            image_names=image_names,
            image_paths=image_paths,
            depth_paths=depth_paths,
            mask_paths=None,
            cameras=cameras,
        ), point_cloud

    def get_outputs(self) -> DataParserOutputs:
        train_set, point_cloud = self._parse_json(self.params.train, True)
        test_set, _ = self._parse_json(self.params.test)

        return DataParserOutputs(
            train_set=train_set,
            val_set=test_set,
            test_set=test_set,
            point_cloud=point_cloud,
            appearance_group_ids=None,
        )
