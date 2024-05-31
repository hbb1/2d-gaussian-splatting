import os.path
import math
import json
import time
from typing import Tuple

import torch
import numpy as np

from plyfile import PlyData, PlyElement

import visual.utils.colmap as colmap_utils
from visual.cameras.cameras import Cameras
from visual.dataparsers.dataparser import DataParser, ImageSet, PointCloud, DataParserOutputs
from visual.configs.dataset import ColmapParams
from visual.utils.graphics_utils import getNerfppNorm


class ColmapDataParser(DataParser):
    def __init__(self, path: str, output_path: str, global_rank: int, params: ColmapParams) -> None:
        super().__init__()
        self.path = path
        self.output_path = output_path
        self.global_rank = global_rank
        self.params = params

    def detect_sparse_model_dir(self) -> str:
        if os.path.isdir(os.path.join(self.path, "sparse", "0")):
            return os.path.join(self.path, "sparse", "0")
        return os.path.join(self.path, "sparse")

    def get_image_dir(self) -> str:
        if self.params.image_dir is None:
            image_dir = os.path.join(self.path, "images")
            if self.params.down_sample_factor > 1:
                image_dir = image_dir + "_{}".format(self.params.down_sample_factor)
            return image_dir
        return os.path.join(self.path, self.params.image_dir)

    @staticmethod
    def rotation_matrix(a, b):
        """Compute the rotation matrix that rotates vector a to vector b.

        Args:
            a: The vector to rotate.
            b: The vector to rotate to.
        Returns:
            The rotation matrix.
        """
        a = a / torch.linalg.norm(a)
        b = b / torch.linalg.norm(b)
        v = torch.cross(a, b)
        c = torch.dot(a, b)
        # If vectors are exactly opposite, we add a little noise to one of them
        if c < -1 + 1e-8:
            eps = (torch.rand(3) - 0.5) * 0.01
            return ColmapDataParser.rotation_matrix(a + eps, b)
        s = torch.linalg.norm(v)
        skew_sym_mat = torch.Tensor(
            [
                [0, -v[2], v[1]],
                [v[2], 0, -v[0]],
                [-v[1], v[0], 0],
            ]
        )
        return torch.eye(3) + skew_sym_mat + skew_sym_mat @ skew_sym_mat * ((1 - c) / (s ** 2 + 1e-8))

    @staticmethod
    def read_points3D_binary(path_to_model_file, selected_image_ids: dict = None):
        """
        see: src/base/reconstruction.cc
            void Reconstruction::ReadPoints3DBinary(const std::string& path)
            void Reconstruction::WritePoints3DBinary(const std::string& path)
        """

        with open(path_to_model_file, "rb") as fid:
            num_points = colmap_utils.read_next_bytes(fid, 8, "Q")[0]

            xyzs = []
            rgbs = []
            errors = []

            for p_id in range(num_points):
                binary_point_line_properties = colmap_utils.read_next_bytes(
                    fid, num_bytes=43, format_char_sequence="QdddBBBd")
                xyz = np.array(binary_point_line_properties[1:4])
                rgb = np.array(binary_point_line_properties[4:7])
                error = np.array(binary_point_line_properties[7])
                track_length = colmap_utils.read_next_bytes(
                    fid, num_bytes=8, format_char_sequence="Q")[0]
                track_elems = colmap_utils.read_next_bytes(
                    fid, num_bytes=8 * track_length,
                    format_char_sequence="ii" * track_length)

                # whether point belongs to selected images
                if selected_image_ids is not None:
                    image_ids = np.array(tuple(map(int, track_elems[0::2])))
                    point_in_selected_image_count = 0
                    for image_id in image_ids:
                        if image_id in selected_image_ids:
                            point_in_selected_image_count += 1
                    if point_in_selected_image_count == 0:
                        continue

                # TODO: filter points in masked area

                xyzs.append(xyz)
                rgbs.append(rgb)
                errors.append(error)
        return np.asarray(xyzs), np.asarray(rgbs), np.asarray(errors)

    def get_outputs(self) -> DataParserOutputs:
        # load colmap sparse model
        sparse_model_dir = self.detect_sparse_model_dir()
        cameras = colmap_utils.read_cameras_binary(os.path.join(sparse_model_dir, "cameras.bin"))
        images = colmap_utils.read_images_binary(os.path.join(sparse_model_dir, "images.bin"))

        # sort images
        images = dict(sorted(images.items(), key=lambda item: item[0]))

        # filter images
        selected_image_ids = None
        selected_image_names = None
        if self.params.image_list is not None:
            # load image list
            selected_image_ids = {}
            selected_image_names = {}
            with open(self.params.image_list, "r") as f:
                for image_name in f:
                    image_name = image_name[:-1]
                    selected_image_names[image_name] = True
            # filter images by image list
            new_images = {}
            for i in images:
                image = images[i]
                if image.name in selected_image_names:
                    selected_image_ids[image.id] = True
                    new_images[i] = image
            assert len(new_images) > 0, "no image left after filtering via {}".format(self.params.image_list)

            # replace images with new_images
            images = new_images

        image_dir = self.get_image_dir()

        # build appearance dict: group name -> image name list
        if self.params.appearance_groups is None:
            print("appearance group by camera id")
            appearance_groups = {}
            for i in images:
                image_camera_id = images[i].camera_id
                if image_camera_id not in appearance_groups:
                    appearance_groups[image_camera_id] = []
                appearance_groups[image_camera_id].append(images[i].name)
        else:
            appearance_group_file_path = os.path.join(self.path, self.params.appearance_groups)
            print("loading appearance groups from {}".format(appearance_group_file_path))
            with open("{}.json".format(appearance_group_file_path), "r") as f:
                appearance_groups = json.load(f)

        # sort the appearance group name list
        appearance_group_name_list = sorted(list(appearance_groups.keys()))
        appearance_group_num = float(len(appearance_group_name_list))

        # use order as the appearance id of the group
        # map from image name to appearance id
        image_name_to_appearance_id = {}
        image_name_to_normalized_appearance_id = {}
        appearance_group_name_to_appearance_id = {}  # tuple(id, normalized id)
        for idx, appearance_group_name in enumerate(appearance_group_name_list):
            normalized_idx = idx / appearance_group_num
            appearance_group_name_to_appearance_id[appearance_group_name] = (idx, normalized_idx)
            for image_name in appearance_groups[appearance_group_name]:
                image_name_to_appearance_id[image_name] = idx
                image_name_to_normalized_appearance_id[image_name] = normalized_idx

        # convert appearance id dict to list, which use the same order as the colmap images
        image_appearance_id = []
        image_normalized_appearance_id = []
        for i in images:
            image_appearance_id.append(image_name_to_appearance_id[images[i].name])
            image_normalized_appearance_id.append(image_name_to_normalized_appearance_id[images[i].name])

        # convert points3D to ply
        # ply_path = os.path.join(sparse_model_dir, "points3D.ply")
        # while os.path.exists(ply_path) is False:
        #     if self.global_rank == 0:
        #         print("converting points3D.bin to ply format")
        #         xyz, rgb, _ = ColmapDataParser.read_points3D_binary(os.path.join(sparse_model_dir, "points3D.bin"))
        #         ColmapDataParser.convert_points_to_ply(ply_path + ".tmp", xyz=xyz, rgb=rgb)
        #         os.rename(ply_path + ".tmp", ply_path)
        #         break
        #     else:
        #         # waiting ply
        #         print("#{} waiting for {}".format(os.getpid(), ply_path))
        #         time.sleep(1)
        print("loading colmap 3D points")
        xyz, rgb, _ = ColmapDataParser.read_points3D_binary(
            os.path.join(sparse_model_dir, "points3D.bin"),
            selected_image_ids=selected_image_ids,
        )

        loaded_mask_count = 0
        # initialize lists
        R_list = []
        T_list = []
        fx_list = []
        fy_list = []
        # fov_x_list = []
        # fov_y_list = []
        cx_list = []
        cy_list = []
        width_list = []
        height_list = []
        appearance_id_list = image_appearance_id
        normalized_appearance_id_list = image_normalized_appearance_id
        camera_type_list = []
        image_name_list = []
        image_path_list = []
        mask_path_list = []

        # parse colmap sparse model
        for idx, key in enumerate(images):
            # extract image and its correspond camera
            extrinsics = images[key]
            intrinsics = cameras[extrinsics.camera_id]

            height = intrinsics.height
            width = intrinsics.width

            R = extrinsics.qvec2rotmat()
            T = np.array(extrinsics.tvec)

            if intrinsics.model == "SIMPLE_PINHOLE":
                focal_length_x = intrinsics.params[0]
                focal_length_y = focal_length_x
                cx = intrinsics.params[1]
                cy = intrinsics.params[2]
                # fov_y = focal2fov(focal_length_x, height)
                # fov_x = focal2fov(focal_length_x, width)
            elif intrinsics.model == "PINHOLE":
                focal_length_x = intrinsics.params[0]
                focal_length_y = intrinsics.params[1]
                cx = intrinsics.params[2]
                cy = intrinsics.params[3]
                # fov_y = focal2fov(focal_length_y, height)
                # fov_x = focal2fov(focal_length_x, width)
            else:
                assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

            # whether mask exists
            mask_path = None
            if self.params.mask_dir is not None:
                mask_path = os.path.join(self.params.mask_dir, "{}.png".format(extrinsics.name))
                if os.path.exists(mask_path) is True:
                    loaded_mask_count += 1
                else:
                    mask_path = None

            # append data to list
            R_list.append(R)
            T_list.append(T)
            fx_list.append(focal_length_x)
            fy_list.append(focal_length_y)
            # fov_x_list.append(fov_x)
            # fov_y_list.append(fov_y)
            cx_list.append(cx)
            cy_list.append(cy)
            width_list.append(width)
            height_list.append(height)
            camera_type_list.append(0)
            image_name_list.append(extrinsics.name)
            image_path_list.append(os.path.join(image_dir, extrinsics.name))
            mask_path_list.append(mask_path)

        # loaded mask must not be zero if self.params.mask_dir provided
        if self.params.mask_dir is not None and loaded_mask_count == 0:
            raise RuntimeError("not a mask was loaded from {}, "
                               "please remove the mask_dir parameter if this is a expected result".format(
                self.params.mask_dir
            ))

        # calculate norm
        # norm = getNerfppNorm(R_list, T_list)

        # convert data to tensor
        R = torch.tensor(np.stack(R_list, axis=0), dtype=torch.float32)
        T = torch.tensor(np.stack(T_list, axis=0), dtype=torch.float32)
        fx = torch.tensor(fx_list, dtype=torch.float32)
        fy = torch.tensor(fy_list, dtype=torch.float32)
        # fov_x = torch.tensor(fov_x_list, dtype=torch.float32)
        # fov_y = torch.tensor(fov_y_list, dtype=torch.float32)
        cx = torch.tensor(cx_list, dtype=torch.float32)
        cy = torch.tensor(cy_list, dtype=torch.float32)
        width = torch.tensor(width_list, dtype=torch.int16)
        height = torch.tensor(height_list, dtype=torch.int16)
        appearance_id = torch.tensor(appearance_id_list, dtype=torch.int)
        normalized_appearance_id = torch.tensor(normalized_appearance_id_list, dtype=torch.float32)
        camera_type = torch.tensor(camera_type_list, dtype=torch.int8)

        # recalculate intrinsics if down sample enabled
        if self.params.down_sample_factor != 1:
            rounding_func = getattr(torch, self.params.down_sample_rounding_model)
            down_sampled_width = rounding_func(width.to(torch.float) / self.params.down_sample_factor)
            down_sampled_height = rounding_func(height.to(torch.float) / self.params.down_sample_factor)
            width_scale_factor = down_sampled_width / width
            height_scale_factor = down_sampled_height / height
            fx *= width_scale_factor
            fy *= height_scale_factor
            cx *= width_scale_factor
            cy *= height_scale_factor

            width = down_sampled_width.to(torch.int16)
            height = down_sampled_height.to(torch.int16)

            print("down sample enabled")

        is_w2c_required = self.params.scene_scale != 1.0 or self.params.reorient is True
        if is_w2c_required:
            # build world-to-camera transform matrix
            w2c = torch.zeros(size=(R.shape[0], 4, 4))
            w2c[:, :3, :3] = R
            w2c[:, :3, 3] = T
            w2c[:, 3, 3] = 1.
            # convert to camera-to-world transform matrix
            c2w = torch.linalg.inv(w2c)

            # reorient
            if self.params.reorient is True:
                print("reorient scene")

                # calculate rotation transform matrix
                up = -torch.mean(c2w[:, :3, 1], dim=0)
                up = up / torch.linalg.norm(up)

                print("up vector = {}".format(up.numpy()))

                rotation = self.rotation_matrix(up, torch.Tensor([0, 0, 1]))
                rotation_transform = torch.eye(4)
                rotation_transform[:3, :3] = rotation

                # reorient cameras
                print("reorienting cameras...")
                c2w = torch.matmul(rotation_transform, c2w)
                # reorient points
                print("reorienting points...")
                xyz = np.matmul(xyz, rotation.numpy().T)

            # rescale scene size
            if self.params.scene_scale != 1.0:
                print("rescal scene with factor {}".format(self.params.scene_scale))

                # rescale camera poses
                c2w[:, :3, 3] *= self.params.scene_scale
                # rescale point cloud
                xyz *= self.params.scene_scale

                # rescale scene extent
                # norm["radius"] *= self.params.scene_scale

            # convert back to world-to-camera
            w2c = torch.linalg.inv(c2w)
            R = w2c[:, :3, :3]
            T = w2c[:, :3, 3]

        # build split indices
        training_set_indices, validation_set_indices = self.build_split_indices(image_name_list)

        # split
        image_set = []
        for index_list in [training_set_indices, validation_set_indices]:
            indices = torch.tensor(index_list, dtype=torch.int)
            cameras = Cameras(
                R=R[indices],
                T=T[indices],
                fx=fx[indices],
                fy=fy[indices],
                cx=cx[indices],
                cy=cy[indices],
                width=width[indices],
                height=height[indices],
                appearance_id=appearance_id[indices],
                normalized_appearance_id=normalized_appearance_id[indices],
                distortion_params=None,
                camera_type=camera_type[indices],
            )
            image_set.append(ImageSet(
                image_names=[image_name_list[i] for i in index_list],
                image_paths=[image_path_list[i] for i in index_list],
                mask_paths=[mask_path_list[i] for i in index_list],
                cameras=cameras
            ))

        # print information
        print("[colmap dataparser] train set images: {}, val set images: {}, loaded mask: {}".format(
            len(image_set[0]),
            len(image_set[1]),
            loaded_mask_count,
        ))

        return DataParserOutputs(
            train_set=image_set[0],
            val_set=image_set[1],
            test_set=image_set[1],
            point_cloud=PointCloud(
                xyz=xyz,
                rgb=rgb,
            ),
            # camera_extent=norm["radius"],
            appearance_group_ids=appearance_group_name_to_appearance_id,
        )

    def build_split_indices(self, image_name_list) -> Tuple[list, list]:
        assert self.params.eval_step > 1, "eval_step must > 1"
        eval_step = self.params.eval_step
        if self.params.eval_image_select_mode == "ratio":
            eval_image_num = max(math.ceil(self.params.eval_ratio * len(image_name_list)), 1)
            eval_step = len(image_name_list) // eval_image_num

        if self.params.split_mode == "experiment":
            # split train set and val set
            training_set_indices = []
            validation_set_indices = []
            for i in range(len(image_name_list)):
                if i % eval_step == 0:
                    validation_set_indices.append(i)
                else:
                    training_set_indices.append(i)
        else:
            # train set contains val set
            training_set_indices = list(range(len(image_name_list)))
            validation_set_indices = training_set_indices[::eval_step]

        return training_set_indices, validation_set_indices
