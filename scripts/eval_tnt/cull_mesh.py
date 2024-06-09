#!/usr/bin/env python
# coding=utf-8
import numpy as np
import open3d as o3d
import os
import argparse
import torch
import trimesh
import pyrender
import copy
from copy import deepcopy
import torch.nn.functional as F
from help_func import auto_orient_and_center_poses
import cv2


def extract_depth_from_mesh(mesh,
                            c2w_list,
                            H, W, fx, fy, cx, cy,
                            far=20.0,):
    """Adapted from Go-Surf: https://github.com/JingwenWang95/go-surf"""
    os.environ['PYOPENGL_PLATFORM'] = 'egl'  # allows for GPU-accelerated rendering
    scene = pyrender.Scene()
    #mesh = trimesh.load("/home/yuzh/mnt/A100_data/sdfstudio/meshes_tnt/bakedangelo/Courthouse_fullres_1024.ply")
    #mesh = trimesh.load("/home/yuzh/mnt/A100_data/sdfstudio/meshes_tnt/bakedangelo/Caterpillar_fullres_1024.ply")
    #mesh = trimesh.load("/home/yuzh/mnt/A100_data/sdfstudio/meshes_tnt/bakedangelo/Truck_fullres_1024.ply")
    #mesh = trimesh.load("/home/yuzh/mnt/A3_data/sdfstudio/meshes_tnt/bakedangelo/Meetingroom_fullres_1024_scaleback.ply")
    mesh = trimesh.load("/home/yuzh/mnt/A3_data/sdfstudio/meshes_tnt/bakedangelo/Barn_fullres_1024.ply")
    mesh = pyrender.Mesh.from_trimesh(mesh)
    scene.add(mesh)
    """
    import glob 
    for f in glob.glob("/home/yuzh/mnt/A100/Projects/sdfstudio/tmp_meshes/*.ply"):
        mesh = trimesh.load(f) 
        mesh = pyrender.Mesh.from_trimesh(mesh)
        scene.add(mesh)
        print(f)
    """

    camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy, znear=0.01, zfar=far)
    camera_node = pyrender.Node(camera=camera, matrix=np.eye(4))
    scene.add_node(camera_node)
    renderer = pyrender.OffscreenRenderer(W, H)
    flags = pyrender.RenderFlags.OFFSCREEN | pyrender.RenderFlags.DEPTH_ONLY | pyrender.RenderFlags.SKIP_CULL_FACES

    depths = []
    for c2w in c2w_list:
        c2w = c2w.detach().numpy()
        # Convert camera coordinate system from OpenCV to OpenGL
        # Details refer to: https://pyrender.readthedocs.io/en/latest/examples/cameras.html
        c2w_gl = deepcopy(c2w)
        # nerfstudio's .json file is already OpenGL coordinate
        #c2w_gl[:3, 1] *= -1
        #c2w_gl[:3, 2] *= -1
        scene.set_pose(camera_node, c2w_gl)
        depth = renderer.render(scene, flags)
        #print(depth, depth.min(), depth.max(), depth.shape)
        #exit(-1)
        #cv2.imshow("s", depth)
        #cv2.waitKey(0)
        depth = torch.from_numpy(depth)
        depths.append(depth)

    renderer.delete()

    return depths


class Mesher(object):
    def __init__(self, H, W, fx, fy, cx, cy, far, points_batch_size=5e5):
        """
        Mesher class, given a scene representation, the mesher extracts the mesh from it.
        Args:
            cfg:                        (dict), parsed config dict
            args:                       (class 'argparse.Namespace'), argparse arguments
            slam:                       (class NICE-SLAM), NICE-SLAM main class
            points_batch_size:          (int), maximum points size for query in one batch
                                        Used to alleviate GPU memory usage. Defaults to 5e5
            ray_batch_size:             (int), maximum ray size for query in one batch
                                        Used to alleviate GPU memory usage. Defaults to 1e5
        """

        self.points_batch_size = int(points_batch_size)
        self.scale = 1.0
        self.device = 'cuda:0'
        self.forecast_radius = 0
        self.H, self.W, self.fx, self.fy, self.cx, self.cy = H, W, fx, fy, cx, cy

        self.resolution = 256
        self.level_set = 0.0
        self.remove_small_geometry_threshold = 0.2
        self.get_largest_components = True
        self.verbose = True

    @torch.no_grad()
    def point_masks(self,
                    input_points,
                    depth_list,
                    estimate_c2w_list):
        """
        Split the input points into seen, unseen, and forecast,
        according to the estimated camera pose and depth image.
        Args:
            input_points:               (Tensor), input points
            keyframe_dict:              (list), list of keyframe info dictionary
            estimate_c2w_list:          (list), estimated camera pose.
            idx:                        (int), current frame index
            device:                     (str), device name to compute on.
            get_mask_use_all_frames:

        Returns:
            seen_mask:                  (Tensor), the mask for seen area.
            forecast_mask:              (Tensor), the mask for forecast area.
            unseen_mask:                (Tensor), the mask for unseen area.

        """
        H, W, fx, fy, cx, cy = self.H, self.W, self.fx, self.fy, self.cx, self.cy
        device =self.device
        if not isinstance(input_points, torch.Tensor):
            input_points = torch.from_numpy(input_points)
        input_points = input_points.clone().detach().float()
        mask = []
        forecast_mask = []
        # this eps should be tuned for the scene
        eps = 0.005
        for _, pnts in enumerate(torch.split(input_points, self.points_batch_size, dim=0)):
            n_pts, _ = pnts.shape
            valid = torch.zeros(n_pts).to(device).bool()
            valid_num = torch.zeros(n_pts).to(device).int()
            valid_forecast = torch.zeros(n_pts).to(device).bool()
            r = self.forecast_radius
            for i in range(len(estimate_c2w_list)):
                points = pnts.to(device).float()
                c2w = estimate_c2w_list[i].to(device).float()
                # transform to opencv coordinate as nerfstudio dataparser's .json file is in opengl coordinate
                c2w[:3, 1:3] *= -1

                depth = depth_list[i].to(device)
                w2c = torch.inverse(c2w).to(device).float()
                ones = torch.ones_like(points[:, 0]).reshape(-1, 1).to(device)
                homo_points = torch.cat([points, ones], dim=1).reshape(-1, 4, 1).to(device).float()
                cam_cord_homo = w2c @ homo_points
                cam_cord = cam_cord_homo[:, :3, :]  # [N, 3, 1]
                K = np.eye(3)
                K[0, 0], K[0, 2], K[1, 1], K[1, 2] = fx, cx, fy, cy
                K = torch.from_numpy(K).to(device)

                uv = K.float() @ cam_cord.float()
                z = uv[:, -1:] + 1e-8
                uv = uv[:, :2] / z  # [N, 2, 1]
                u, v = uv[:, 0, 0].float(), uv[:, 1, 0].float()
                z = z[:, 0, 0].float()

                in_frustum = (u >= 0) & (u <= W-1) & (v >= 0) & (v <= H-1) & (z > 0)
                forecast_frustum = (u >= -r) & (u <= W-1+r) & (v >= -r) & (v <= H-1+r) & (z > 0)

                depth = depth.reshape(1, 1, H, W)
                vgrid = uv.reshape(1, 1, -1, 2)
                # normalized to [-1, 1]
                vgrid[..., 0] = (vgrid[..., 0] / (W - 1) * 2.0 - 1.0)
                vgrid[..., 1] = (vgrid[..., 1] / (H - 1) * 2.0 - 1.0)

                depth_sample = F.grid_sample(depth, vgrid, padding_mode='border', align_corners=True)
                depth_sample = depth_sample.reshape(-1)
                is_front_face = torch.where((depth_sample > 0.0), (z < (depth_sample + eps)), torch.ones_like(z).bool())
                is_forecast_face = torch.where((depth_sample > 0.0), (z < (depth_sample + eps)), torch.ones_like(z).bool())
                in_frustum = in_frustum & is_front_face

                valid = valid | in_frustum.bool()
                valid_num = valid_num + in_frustum.int()

                forecast_frustum = forecast_frustum & is_forecast_face
                forecast_frustum = in_frustum | forecast_frustum
                valid_forecast = valid_forecast | forecast_frustum.bool()
            valid = valid_num >= 20
            mask.append(valid.cpu().numpy())
            forecast_mask.append(valid_forecast.cpu().numpy())

        mask = np.concatenate(mask, axis=0)
        forecast_mask = np.concatenate(forecast_mask, axis=0)

        return mask, forecast_mask


    @torch.no_grad()
    def get_connected_mesh(self, mesh, get_largest_components=False):
        print("split")
        components = mesh.split(only_watertight=False)
        print("split completed")
        if get_largest_components:
            areas = np.array([c.area for c in components], dtype=np.float)
            mesh = components[areas.argmax()]
        else:
            new_components = []
            global_area = mesh.area
            for comp in components:
                if comp.area > self.remove_small_geometry_threshold * global_area:
                    new_components.append(comp)
            mesh = trimesh.util.concatenate(new_components)

        return mesh

    @torch.no_grad()
    def cull_mesh(self,
                  mesh,
                  estimate_c2w_list):
        """
        Extract mesh from scene representation and save mesh to file.
        Args:
            mesh_out_file:              (str), output mesh filename
            estimate_c2w_list:          (Tensor), estimated camera pose, camera coordinate system is same with OpenCV
                                        [N, 4, 4]
        """

        step = 1
        print('Start Mesh Culling', end='')

        # cull with 3d projection
        print(f' --->> {step}(Projection)', end='')
        forward_depths = extract_depth_from_mesh(
            mesh, estimate_c2w_list, H=self.H, W=self.W, fx=self.fx, fy=self.fy, cx=self.cx, cy=self.cy, far=20.0
        )
        print("after forward depth")
        """        
        backward_mesh = deepcopy(mesh)
        backward_mesh.faces[:, [1, 2]] = backward_mesh.faces[:, [2, 1]]  # make the mesh faces from, e.g., facing inside to outside
        backward_depths = extract_depth_from_mesh(
            backward_mesh, estimate_c2w_list, H=self.H, W=self.W, fx=self.fx, fy=self.fy, cx=self.cx, cy=self.cy, far=20.0
        )

        depth_list = []
        for i in range(len(forward_depths)):
            depth = torch.where(forward_depths[i] > 0, forward_depths[i], backward_depths[i])
            depth = torch.where((backward_depths[i] > 0) & (backward_depths[i] < depth), backward_depths[i], depth)
            depth_list.append(depth)
        """
        depth_list = forward_depths

        print("in point masks")
        vertices = mesh.vertices[:, :3]
        mask, forecast_mask = self.point_masks(
            vertices, depth_list, estimate_c2w_list
        )
        print(mask.shape, forecast_mask.shape, mask.mean())
        
        face_mask = mask[mesh.faces].all(axis=1)
        mesh_with_hole = deepcopy(mesh)
        mesh_with_hole.update_faces(face_mask)
        mesh_with_hole.remove_unreferenced_vertices()
        #mesh_with_hole.process(validate=True)
        step += 1

        print("compute componet")
        # cull by computing connected components
        print(f' --->> {step}(Component)', end='')
        #cull_mesh = self.get_connected_mesh(mesh_with_hole, self.get_largest_components)
        cull_mesh = mesh_with_hole
        print("after compute componet")
        step += 1

        if abs(self.forecast_radius) > 0:
            # for forecasting
            print(f' --->> {step}(Forecast:{self.forecast_radius})', end='')
            forecast_face_mask = forecast_mask[mesh.faces].all(axis=1)
            forecast_mesh = deepcopy(mesh)
            forecast_mesh.update_faces(forecast_face_mask)
            forecast_mesh.remove_unreferenced_vertices()

            cull_pc = o3d.geometry.PointCloud(
                o3d.utility.Vector3dVector(np.array(cull_mesh.vertices))
            )
            aabb = cull_pc.get_oriented_bounding_box()
            indices = aabb.get_point_indices_within_bounding_box(
                o3d.utility.Vector3dVector(np.array(forecast_mesh.vertices))
            )
            bound_mask = np.zeros(len(forecast_mesh.vertices))
            bound_mask[indices] = 1.0
            bound_mask = bound_mask.astype(np.bool_)
            forecast_face_mask = bound_mask[forecast_mesh.faces].all(axis=1)
            forecast_mesh.update_faces(forecast_face_mask)
            forecast_mesh.remove_unreferenced_vertices()
            forecast_mesh = self.get_connected_mesh(forecast_mesh, self.get_largest_components)
            step += 1
        else:
            forecast_mesh = deepcopy(cull_mesh)

        print(' --->> Done!')

        return cull_mesh, forecast_mesh

    def __call__(self, mesh_path, estimate_c2w_list):
        print(f'Loading mesh from {mesh_path}...')
        mesh = trimesh.load(mesh_path, process=True)
        mesh.merge_vertices()

        """
        print(f'Mesh loaded from {mesh_path}!')
        mask = np.linalg.norm(mesh.vertices, axis=-1) < 1.0
        print(mask.shape, mask.mean())
        face_mask = mask[mesh.faces].all(axis=1)
        mesh_with_hole = deepcopy(mesh)
        mesh_with_hole.update_faces(face_mask)
        mesh_with_hole.remove_unreferenced_vertices()
        mesh = mesh_with_hole
        print(f'Mesh clear from {mesh_path}!')
        """

        mesh_out_file = mesh_path.replace('.ply', '_cull.ply')
        cull_mesh, forecast_mesh = self.cull_mesh(
            mesh=mesh,
            estimate_c2w_list=estimate_c2w_list,
        )

        cull_mesh.export(mesh_out_file)
        if self.verbose:
            print("\nINFO: Save mesh at {}!\n".format(mesh_out_file))

        torch.cuda.empty_cache()


def get_traj(traj_path):
    print(f'Load trajectory from {traj_path}.')
    traj_to_register = []
    if traj_path.endswith('.npy'):
        ld = np.load(traj_path)
        for i in range(len(ld)):
            # traj_to_register.append(CameraPose(meta=None, mat=ld[i]))
            traj_to_register.append(ld[i])
    elif traj_path.endswith('.json'): # instant-npg or sdfstudio format
        import json
        with open(traj_path, encoding='UTF-8') as f:
            meta = json.load(f)
        poses_dict = {}
        for i, frame in enumerate(meta['frames']):
            filepath = frame['file_path']
            new_i = int(filepath[13:18]) - 1
            poses_dict[new_i] = np.array(frame['transform_matrix'])
        poses = []
        for i in range(len(poses_dict)):
            poses.append(poses_dict[i])
        poses = torch.from_numpy(np.array(poses).astype(np.float32))
        poses, _ = auto_orient_and_center_poses(poses, method='up', center_poses=True)
        scale_factor = 1.0 / float(torch.max(torch.abs(poses[:, :3, 3])))
        poses[:, :3, 3] *= scale_factor
        poses = poses.numpy()
        for i in range(len(poses)):
            traj_to_register.append(poses[i])

    else:
        # traj_to_register = read_trajectory(traj_path)
        pass
    
    for i in range(len(traj_to_register)):
        c2w = torch.from_numpy(traj_to_register[i]).float()
        if c2w.shape == (3, 4):
            c2w = torch.cat([
                c2w,
                torch.tensor([[0, 0, 0, 1]]).float()
            ], dim=0)
        traj_to_register[i] = c2w # [4, 4]

    print(f'Trajectory loaded from {traj_path}, including {len(traj_to_register)} camera views.')
    return traj_to_register


if __name__ == "__main__":
    print('Start culling...')
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--traj-path",
        type=str,
        required=True,
        help=
        "path to trajectory file. See `convert_to_logfile.py` to create this file.",
    )
    parser.add_argument(
        "--ply-path",
        type=str,
        required=True,
        help="path to reconstruction ply file",
    )
    args = parser.parse_args()

    estimate_c2w_list = get_traj(args.traj_path)

    # for TanksandTemples dataset
    H, W = 1080, 1920
    fx = 1163.8678928442187
    fy = 1172.793101201448
    cx = 962.3120628412543
    cy = 542.0667209577691
    far = 20.0

    mesher = Mesher(H, W, fx, fy, cx, cy, far, points_batch_size=5e5)
    # mesher = Mesher(H*2, W*2, fx*2, fy*2, cx*2, cy*2, far, points_batch_size=5e5)

    mesher(args.ply_path, estimate_c2w_list)

    print('Done!')

