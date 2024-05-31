import json
import os.path

import torch
import numpy as np
from internal.configs.dataset import NerfiesParams
from .dataparser import DataParser, ImageSet, Cameras, PointCloud, DataParserOutputs
from ..utils.graphics_utils import get_center_and_diag_from_hstacked_xyz


class NerfiesDataparser(DataParser):
    def __init__(self, path: str, output_path: str, global_rank: int, params: NerfiesParams):
        self.path = path
        self.output_path = output_path
        self.global_rank = global_rank
        self.params = params

    def _get_image_set(self, ids: list, time: dict, scene: dict) -> ImageSet:
        image_name_list = []
        image_path_list = []
        c2w_list = []
        fx_list = []
        fy_list = []
        cx_list = []
        cy_list = []
        width_list = []
        height_list = []
        time_list = []
        distortion_list = []
        for i in ids:
            image_name = "{}.png".format(i)
            image_name_list.append(image_name)
            image_path_list.append(os.path.join(self.path, "rgb", "{}x".format(self.params.down_sample_factor), image_name))

            with open(os.path.join(self.path, "camera", "{}.json".format(i)), "r") as f:
                camera = json.load(f)
            # extrinsics
            c2w = torch.eye(4, dtype=torch.float64)
            c2w[:3, :3] = torch.tensor(camera["orientation"]).T
            c2w[:3, 3] = torch.tensor(camera["position"])
            c2w_list.append(c2w)
            # intrinsics
            fx_list.append(camera["focal_length"])
            fy_list.append(camera["focal_length"] * camera["pixel_aspect_ratio"])
            cx_list.append(camera["principal_point"][0])
            cy_list.append(camera["principal_point"][1])
            width_list.append(camera["image_size"][0])
            height_list.append(camera["image_size"][1])

            radial_distortion = camera["radial_distortion"]
            tangential_distortion = camera["tangential_distortion"]
            distortion_list.append(torch.tensor([
                radial_distortion[0],
                radial_distortion[1],
                tangential_distortion[0],
                tangential_distortion[1],
                radial_distortion[2],
            ], dtype=torch.float))  # [k1, k2, p1, p2, k3]

            # metadata
            time_list.append(time[i])

        c2w = torch.stack(c2w_list)
        c2w[:, :3, 3] -= torch.tensor(scene["center"])
        c2w[:, :3, 3] *= scene["scale"]
        w2c = torch.linalg.inv(c2w).to(torch.float)

        fx = torch.tensor(fx_list, dtype=torch.float)
        fy = torch.tensor(fy_list, dtype=torch.float)
        cx = torch.tensor(cx_list, dtype=torch.float)
        cy = torch.tensor(cy_list, dtype=torch.float)
        width = torch.tensor(width_list, dtype=torch.int16)
        height = torch.tensor(height_list, dtype=torch.int16)
        distortion = torch.stack(distortion_list)
        time = torch.tensor(time_list)

        # _, diagonal = get_center_and_diag_from_hstacked_xyz(c2w[:, :3, 3].T.numpy())
        # diagonal *= 1.1

        # resize
        if self.params.down_sample_factor != 1:
            down_sampled_width = torch.round(width.to(torch.float) / self.params.down_sample_factor)
            down_sampled_height = torch.round(height.to(torch.float) / self.params.down_sample_factor)
            width_scale_factor = down_sampled_width / width
            height_scale_factor = down_sampled_height / height
            fx *= width_scale_factor
            fy *= height_scale_factor
            cx *= width_scale_factor
            cy *= height_scale_factor

            width = down_sampled_width.to(torch.int16)
            height = down_sampled_height.to(torch.int16)

            # print("down sample enabled")

        return ImageSet(
            image_names=image_name_list,
            image_paths=image_path_list,
            mask_paths=None,
            cameras=Cameras(
                R=w2c[:, :3, :3],
                T=w2c[:, :3, 3],
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                width=width,
                height=height,
                appearance_id=torch.zeros_like(width),
                normalized_appearance_id=torch.zeros_like(fx),
                distortion_params=distortion,
                camera_type=torch.zeros_like(width),
                time=time,
            )
        )

    def get_outputs(self) -> DataParserOutputs:
        with open(os.path.join(self.path, "dataset.json"), "r") as f:
            dataset = json.load(f)
        with open(os.path.join(self.path, "metadata.json"), "r") as f:
            metadata = json.load(f)
        with open(os.path.join(self.path, "scene.json"), "r") as f:
            scene = json.load(f)

        train_ids = dataset["train_ids"]
        val_ids = dataset["val_ids"]
        if len(val_ids) == 0:
            # build val_ids from all ids
            train_ids = []
            val_ids = []
            for idx, i in enumerate(dataset["ids"][::self.params.step]):
                if idx % self.params.eval_step == 0:
                    val_ids.append(i)
                else:
                    train_ids.append(i)
        else:
            train_ids = train_ids[::self.params.step]
            val_ids = val_ids[::self.params.step]

        if self.params.split_mode == "reconstruction":
            train_ids += val_ids

        # normalize time value
        max_time = 0
        for i in metadata:
            if metadata[i]["warp_id"] > max_time:
                max_time = metadata[i]["warp_id"]
        time_dict = {}
        for i in metadata:
            time_dict[i] = metadata[i]["warp_id"] / max_time

        # parse camera parameters
        train_set = self._get_image_set(train_ids, time_dict, scene)
        val_set = self._get_image_set(val_ids, time_dict, scene)

        xyz = np.load(os.path.join(self.path, "points.npy"))
        xyz = (xyz - np.asarray(scene["center"])) * scene["scale"]

        return DataParserOutputs(
            train_set=train_set,
            val_set=val_set,
            test_set=val_set,
            point_cloud=PointCloud(
                xyz=xyz,
                rgb=np.ones_like(xyz) * 127,
                # rgb=np.random.random(xyz.shape) * 127,
            ),
            # camera_extent=radius,
            appearance_group_ids=None,
        )
