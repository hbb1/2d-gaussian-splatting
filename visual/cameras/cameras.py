from typing import Optional, Union
from dataclasses import dataclass, field

import torch
from torch import Tensor


class CameraType:
    PERSPECTIVE: int = 0
    FISHEYE: int = 1


@dataclass
class Camera:
    R: Tensor  # [3, 3]
    T: Tensor  # [3]
    fx: Tensor
    fy: Tensor
    fov_x: Tensor
    fov_y: Tensor
    cx: Tensor
    cy: Tensor
    width: Tensor
    height: Tensor
    appearance_id: Tensor
    normalized_appearance_id: Tensor
    time: Tensor
    distortion_params: Optional[Tensor]  # (k1,k2,p1,p2[,k3[,k4,k5,k6[,s1,s2,s3,s4[,τx,τy]]]]) of 4, 5, 8, 12 or 14 elements
    camera_type: Tensor

    world_to_camera: Tensor
    projection: Tensor
    full_projection: Tensor
    camera_center: Tensor

    def to_device(self, device):
        for field in Camera.__dataclass_fields__:
            value = getattr(self, field)
            if isinstance(value, torch.Tensor):
                setattr(self, field, value.to(device))

        return self


@dataclass
class Cameras:
    """
    Y down, Z forward
    world-to-camera
    """

    R: Tensor  # [n_cameras, 3, 3]
    T: Tensor  # [n_cameras, 3]
    fx: Tensor  # [n_cameras]
    fy: Tensor  # [n_cameras]
    fov_x: Tensor = field(init=False)  # [n_cameras]
    fov_y: Tensor = field(init=False)  # [n_cameras]
    cx: Tensor  # [n_cameras]
    cy: Tensor  # [n_cameras]
    width: Tensor  # [n_cameras]
    height: Tensor  # [n_cameras]
    appearance_id: Tensor  # [n_cameras]
    normalized_appearance_id: Optional[Tensor]  # [n_cameras]
    distortion_params: None #Optional[Union[Tensor, list[Tensor]]]  # [n_cameras, 2 or 4 or 5 or 8 or 12 or 14], (k1,k2,p1,p2[,k3[,k4,k5,k6[,s1,s2,s3,s4[,τx,τy]]]]) of 4, 5, 8, 12 or 14 elements
    camera_type: Tensor  # Int[n_cameras]

    world_to_camera: Tensor = field(init=False)  # [n_cameras, 4, 4], transposed
    projection: Tensor = field(init=False)
    full_projection: Tensor = field(init=False)
    camera_center: Tensor = field(init=False)

    time: Optional[Tensor] = None  # [n_cameras]

    def _calculate_fov(self):
        # calculate fov
        self.fov_x = 2 * torch.atan((self.width / 2) / self.fx)
        self.fov_y = 2 * torch.atan((self.height / 2) / self.fy)

    def _calculate_w2c(self):
        # build world-to-camera transform matrix
        self.world_to_camera = torch.zeros((self.R.shape[0], 4, 4))
        self.world_to_camera[:, :3, :3] = self.R
        self.world_to_camera[:, :3, 3] = self.T
        self.world_to_camera[:, 3, 3] = 1.
        self.world_to_camera = torch.transpose(self.world_to_camera, 1, 2)

    def _calculate_ndc_projection_matrix(self):
        """
        calculate ndc projection matrix
        http://www.songho.ca/opengl/gl_projectionmatrix.html

        TODO:
            1. support colmap refined principal points
            2. the near and far here are ignored in diff-gaussian-rasterization
        """
        zfar = 100.0
        znear = 0.01

        tanHalfFovY = torch.tan((self.fov_y / 2))
        tanHalfFovX = torch.tan((self.fov_x / 2))

        top = tanHalfFovY * znear
        bottom = -top
        right = tanHalfFovX * znear
        left = -right

        P = torch.zeros(self.fov_y.shape[0], 4, 4)

        z_sign = 1.0

        P[:, 0, 0] = 2.0 * znear / (right - left)  # = 1 / tanHalfFovX = 2 * fx / width
        P[:, 1, 1] = 2.0 * znear / (top - bottom)  # = 2 * fy / height
        P[:, 0, 2] = (right + left) / (right - left)  # = 0, right + left = 0
        P[:, 1, 2] = (top + bottom) / (top - bottom)  # = 0, top + bottom = 0
        P[:, 3, 2] = z_sign
        P[:, 2, 2] = z_sign * zfar / (zfar - znear)
        P[:, 2, 3] = -(zfar * znear) / (zfar - znear)

        self.projection = torch.transpose(P, 1, 2)

        self.full_projection = self.world_to_camera.bmm(self.projection)

    def _calculate_camera_center(self):
        self.camera_center = torch.linalg.inv(self.world_to_camera)[:, 3, :3]

    def __post_init__(self):
        self._calculate_fov()
        self._calculate_w2c()
        self._calculate_ndc_projection_matrix()
        self._calculate_camera_center()

        if self.time is None:
            self.time = torch.zeros(self.R.shape[0])
        if self.distortion_params is None:
            self.distortion_params = torch.zeros(self.R.shape[0], 4)

    def __len__(self):
        return self.R.shape[0]

    def __getitem__(self, index) -> Camera:
        return Camera(
            R=self.R[index],
            T=self.T[index],
            fx=self.fx[index],
            fy=self.fy[index],
            fov_x=self.fov_x[index],
            fov_y=self.fov_y[index],
            cx=self.cx[index],
            cy=self.cy[index],
            width=self.width[index],
            height=self.height[index],
            appearance_id=self.appearance_id[index],
            normalized_appearance_id=self.normalized_appearance_id[index],
            distortion_params=self.distortion_params[index],
            time=self.time[index],
            camera_type=self.camera_type[index],
            world_to_camera=self.world_to_camera[index],
            projection=self.projection[index],
            full_projection=self.full_projection[index],
            camera_center=self.camera_center[index],
        )
