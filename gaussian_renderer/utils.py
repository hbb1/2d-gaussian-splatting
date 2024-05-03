import torch
import numpy as np
import os
import math
from tqdm import tqdm
from utils.render_utils import save_img_f32, save_img_u8
from utils.point_utils import storePly
from functools import partial

class MeshExtractor(object):
    def __init__(self, gaussians, render, pipe, bg_color=None):
        if bg_color is None:
            bg_color = [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        self.gaussians = gaussians
        self.render = partial(render, pipe=pipe, bg_color=background)
        self.depthmaps = []
        self.alphamaps = []
        self.rgbmaps = []
        self.normals = []
        self.depth_normals = []
        self.points = []
    
    @torch.no_grad()
    def reconstruct(self, viewpoint_stack):
        self.depthmaps = []
        self.alphamaps = []
        self.rgbmaps = []
        self.viewpoint_stack = viewpoint_stack
        for i, viewpoint_cam in tqdm(enumerate(self.viewpoint_stack), desc="reconstruct radiance fields"):
            render_pkg = self.render(viewpoint_cam, self.gaussians)
            rgb = render_pkg['render']
            alpha = render_pkg['rend_alpha']
            normal = torch.nn.functional.normalize(render_pkg['rend_normal'], dim=0)
            
            depth = render_pkg['surf_depth']
            depth_normal = render_pkg['surf_normal']
            point = render_pkg['surf_point']
            self.rgbmaps.append(rgb.cpu())
            self.depthmaps.append(depth.cpu())
            self.alphamaps.append(alpha.cpu())
            self.normals.append(normal.cpu())
            self.depth_normals.append(depth_normal.cpu())
            self.points.append(point.cpu())
        
        self.rgbmaps = torch.stack(self.rgbmaps, dim=0)
        self.depthmaps = torch.stack(self.depthmaps, dim=0)
        self.alphamaps = torch.stack(self.alphamaps, dim=0)
        self.depth_normals = torch.stack(self.depth_normals, dim=0)
        self.points = torch.stack(self.points, dim=0)

    @torch.no_grad()
    def export_image(self, path):
        # import torchvision
        render_path = os.path.join(path, "renders")
        gts_path = os.path.join(path, "gt")
        vis_path = os.path.join(path, "vis")
        os.makedirs(render_path, exist_ok=True)
        os.makedirs(vis_path, exist_ok=True)
        os.makedirs(gts_path, exist_ok=True)
        for idx, viewpoint_cam in tqdm(enumerate(self.viewpoint_stack), desc="export images"):
            gt = viewpoint_cam.original_image[0:3, :, :]
            save_img_u8(gt.permute(1,2,0).cpu().numpy(), os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
            save_img_u8(self.rgbmaps[idx].permute(1,2,0).cpu().numpy(), os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
            save_img_f32(self.depthmaps[idx][0].cpu().numpy(), os.path.join(vis_path, 'depth_{0:05d}'.format(idx) + ".tiff"))
            save_img_u8(self.normals[idx].permute(1,2,0).cpu().numpy() * 0.5 + 0.5, os.path.join(vis_path, 'normal_{0:05d}'.format(idx) + ".png"))
            save_img_u8(self.depth_normals[idx].permute(1,2,0).cpu().numpy() * 0.5 + 0.5, os.path.join(vis_path, 'depth_normal_{0:05d}'.format(idx) + ".png"))

    def export_mesh_tsdf_dt(self, path):
        import numpy as np
        import trimesh
        from scipy.spatial import Delaunay
        points = self.gaussians.get_xyz
        stds = self.gaussians.get_scaling
        stds = torch.tensor([[1, 1],[1,-1],[-1,1],[-1,-1]]).cuda().unsqueeze(0) * stds.unsqueeze(1)
        stds = torch.cat([stds, 0 * torch.ones_like(stds[...,:1])], dim=-1)

        from scene.gaussian_model import build_rotation
        rots = build_rotation(self.gaussians._rotation).unsqueeze(1).repeat(1,4,1,1).reshape(-1,3,3)
        points = self.gaussians.get_xyz.unsqueeze(1).repeat(1,4,1).reshape(-1,3)
        samples = stds.reshape(-1, 3)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + points
        sdfs = -100 * torch.ones_like(new_xyz[:,0])
        valid_masks = torch.zeros_like(new_xyz[:,0])
        tri = Delaunay(new_xyz.cpu().numpy())
        f = tri.simplices

        for i, viewpoint_cam in tqdm(enumerate(self.viewpoint_stack), desc="TSDF integration progress"):
            depth = self.depthmaps[i].cuda()
            alpha = self.alphamaps[i].cuda()
            rgb = self.rgbmaps[i].cuda()
            W, H = viewpoint_cam.image_width, viewpoint_cam.image_height
            fx = W / (2 * math.tan(viewpoint_cam.FoVx / 2.))
            fy = H / (2 * math.tan(viewpoint_cam.FoVy / 2.))
            intrins = torch.eye(4).cuda()
            intrins[:3,:3] = torch.tensor(
                [[fx, 0., W/2.],
                [0., fy, H/2.],
                [0., 0., 1.0]]
            ).float().cuda()

            w2cT = viewpoint_cam.world_view_transform
            points = torch.cat([new_xyz, torch.ones_like(new_xyz[:,:1])], dim=-1) @ w2cT @ intrins.T
            points = points[:,:3]
            z = points[:, -1:]
            pix_coords = points[..., :2] / z
            pix_coords[..., 0] /= W - 1
            pix_coords[..., 1] /= H - 1
            pix_coords = (pix_coords - 0.5) * 2
            valid = ((pix_coords > -1. ) & (pix_coords < 1.)).all(dim=-1)
            pix_coords = pix_coords[valid]
            z = z[valid]
            sampled_depth = torch.nn.functional.grid_sample(depth[None], pix_coords[None, None], mode='nearest', padding_mode='zeros', align_corners=True)[0, -1, 0]
            sdf = (sampled_depth - z.flatten()) # inside is negative and outside is positive
            sdfs[valid] = torch.max(sdfs[valid],sdf)
            valid_masks[valid] = 1.0
            # import matplotlib.pyplot as plt
            # mask = self.alphamaps[i,0].cpu().numpy()
            # index = np.random.permutation(len(pix_coords))[:10000]
            # scatters = pix_coords[index].cpu().numpy()
            # far = 10
            # near = 0.02
            # color = (far*(z-near)/((far-near)*z))[index].cpu().numpy().flatten()
            # plt.scatter(scatters[:,0], scatters[:,1], c=color)
            # plt.savefig(f'test{i}')
            # if i % 10 == 0:
            #     from kaolin.ops.conversions import marching_tetrahedra
            #     local_sdf = sdfs.clone()
            #     local_sdf[valid_masks < 1] = 100
            #     verts_list, faces_list, tet_idx_list = marching_tetrahedra(torch.tensor(new_xyz.reshape(1,-1,3)), tets=torch.tensor(f).long(), sdf=local_sdf.unsqueeze(0).float(), return_tet_idx=True)
            #     mesh = trimesh.Trimesh(vertices=verts_list[0].cpu().numpy(), faces=faces_list[0].cpu().numpy())
            #     mesh.export(file_obj=f"mesh_{i}.ply")
        
        from kaolin.ops.conversions import marching_tetrahedra
        sdfs[valid_masks < 1] = 100 # outside
        verts_list, faces_list, tet_idx_list = marching_tetrahedra(torch.tensor(new_xyz.reshape(1,-1,3)), tets=torch.tensor(f).long(), sdf=sdfs.unsqueeze(0).float(), return_tet_idx=True)
        mesh = trimesh.Trimesh(vertices=verts_list[0].cpu().numpy(), faces=faces_list[0].cpu().numpy())
        mesh.export(file_obj=path)
        print("save dt_tsdf mesh into {}".format(path))
        import pdb; pdb.set_trace()

    @torch.no_grad()
    def export_mesh_tsdf(self, path, voxel_size=2/512, sdf_trunc=0.05, alpha_thres=0.5, depth_trunc=100):
        import copy
        import open3d as o3d
        # if self.aabb is not None:
        #     center = self.aabb.mean(-1)
        #     radius = np.linalg.norm(self.aabb[:,-1] - self.aabb[:, 0], axis=-1) * 0.5
        #     voxel_size = radius / 1024
        #     sdf_trunc = voxel_size * 2
        #     print("using aabb")

        print("running tsdf volume integration ...")
        print(f'voxel_size: {voxel_size}')
        print(f'sdf_trunc: {sdf_trunc}')
        print(f'depth_truc: {depth_trunc}')
        print(f'alpha_thres: {alpha_thres}')

        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length= voxel_size,
            sdf_trunc=sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )

        camera_traj = []
        rgbd_images = []
        for i, viewpoint_cam in tqdm(enumerate(self.viewpoint_stack), desc="TSDF integration progress"):
            depth = self.depthmaps[i]
            alpha = self.alphamaps[i]
            rgb = self.rgbmaps[i]

            if viewpoint_cam.gt_alpha_mask is not None:
                depth[(viewpoint_cam.gt_alpha_mask < 0.5)] = 0
            
            depth[(alpha < alpha_thres)] = 0
            # if self.aabb is not None:
                # campos = viewpoint_cam.camera_center.cpu().numpy()
                # depth_trunc = np.linalg.norm(campos - center, axis=-1) + radius
            
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(np.asarray(rgb.permute(1,2,0).cpu().numpy() * 255, order="C", dtype=np.uint8)),
                o3d.geometry.Image(np.asarray(depth.permute(1,2,0).cpu().numpy(), order="C")),
                depth_trunc = depth_trunc, convert_rgb_to_intensity=False,
                depth_scale = 1.0
            )

            intrinsic=o3d.camera.PinholeCameraIntrinsic(width=viewpoint_cam.image_width, 
                    height=viewpoint_cam.image_height, 
                    cx = viewpoint_cam.image_width/2,
                    cy = viewpoint_cam.image_height/2,
                    fx = viewpoint_cam.image_width / (2 * math.tan(viewpoint_cam.FoVx / 2.)),
                    fy = viewpoint_cam.image_height / (2 * math.tan(viewpoint_cam.FoVy / 2.)))
            
            extrinsic=np.asarray((viewpoint_cam.world_view_transform.T).cpu().numpy())
            volume.integrate(rgbd, 
                        intrinsic=intrinsic, 
                        extrinsic=extrinsic)
            
            camera = o3d.camera.PinholeCameraParameters()
            camera.extrinsic = extrinsic
            camera.intrinsic = intrinsic
            camera_traj.append(camera)
            rgbd_images.append(rgbd)

        camera_trajectory = o3d.camera.PinholeCameraTrajectory()
        camera_trajectory.parameters = camera_traj

        mesh = volume.extract_triangle_mesh()
        # write mesh
        os.makedirs(os.path.dirname(path), exist_ok=True)
        o3d.io.write_triangle_mesh(path, mesh)
        mesh_0 = copy.deepcopy(mesh)

        # if self.aabb is not None:
        #     vert_mask = ~((np.asarray(mesh_0.vertices) >= self.aabb[:,0]).all(-1) & (np.asarray(mesh_0.vertices) <= self.aabb[:,1]).all(-1))
        #     triangles_to_remove = vert_mask[np.array(mesh_0.triangles)].any(axis=-1)
        #     mesh_0.remove_triangles_by_mask(triangles_to_remove)
        
        # if you want render the textured mesh with open3d, use it and send it the following script
        # https://www.open3d.org/docs/0.12.0/tutorial/visualization/customized_visualization.html#capture-images-in-a-customized-animation
        # camera_path = os.path.join(os.path.dirname(path), 'camera_traj.json')
        # o3d.io.write_pinhole_camera_trajectory(camera_path, camera_trajectory)
        
        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
            triangle_clusters, cluster_n_triangles, cluster_area = (mesh_0.cluster_connected_triangles())

        # postprocessing
        triangle_clusters = np.asarray(triangle_clusters)
        cluster_n_triangles = np.asarray(cluster_n_triangles)
        cluster_area = np.asarray(cluster_area)
        largest_cluster_idx = cluster_n_triangles.argmax()
        # cluster_to_keep = 10
        cluster_to_keep = 1 # you can modify this to keep more clusters, depending on applications
        n_cluster = np.sort(cluster_n_triangles.copy())[-cluster_to_keep]

        triangles_to_remove = cluster_n_triangles[triangle_clusters] < n_cluster
        mesh_0.remove_triangles_by_mask(triangles_to_remove)
        mesh_0.remove_unreferenced_vertices()
        o3d.io.write_triangle_mesh(path.replace('.ply', '_post.ply'), mesh_0)
        print("num vertices raw {}".format(len(mesh.vertices)))
        print("num vertices post {}".format(len(mesh_0.vertices)))
        print("save tsdf mesh into {}".format(path))