import torch
from gaussian_renderer import render
from graphics_utils import patchmatch_propagation
from graphics_utils import check_geometric_consistency



def process_propagation(viewpoint_stack, viewpoint_cam, gaussians, pipe, background, iteration, opt, src_idxs):
    with torch.no_grad():
        loss_depth = torch.tensor(0.).cuda()
        if iteration > opt.propagation_begin and iteration < opt.propagation_after and (iteration % opt.propagation_interval == 0):
            render_pkg = render(viewpoint_cam, gaussians, pipe, background)
            projected_depth = render_pkg["surf_depth"]
            rendered_normal = render_pkg["surf_normal"]

            # get the opacity that less than the threshold, propagate depth in these region
            if viewpoint_cam.sky_mask is not None:
                sky_mask = viewpoint_cam.sky_mask.to("cuda").to(torch.bool)
            else:
                sky_mask = None

            # get the propagated depth
            propagated_depth, propagated_normal = patchmatch_propagation(viewpoint_cam, projected_depth, rendered_normal, viewpoint_stack, src_idxs, opt.patch_size)
            # transform normal to camera coordinate
            R_w2c = torch.tensor(viewpoint_cam.R.T).cuda().to(torch.float32)
            propagated_normal = (R_w2c @ propagated_normal.view(-1, 3).permute(1, 0)).view(3, viewpoint_cam.image_height, viewpoint_cam.image_width)                
                
            if sky_mask is not None:
                propagated_depth[~sky_mask] = 300
            valid_mask = propagated_depth != 300

            # calculate the abs rel depth error of the propagated depth and rendered depth
            abs_rel_error = torch.abs(propagated_depth - projected_depth) / propagated_depth
            abs_rel_error_threshold = opt.depth_error_max_threshold - (opt.depth_error_max_threshold - opt.depth_error_min_threshold) * (iteration - opt.propagation_begin) / (opt.propagation_after - opt.propagation_begin)

            #for waymo, quantile 0.6
            error_mask = (abs_rel_error > abs_rel_error_threshold)
                
            # calculate the geometric consistency
            ref_K = viewpoint_cam.K
            ref_pose = viewpoint_cam.world_view_transform.transpose(0, 1).inverse()
            geometric_counts = None
            for idx, src_idx in enumerate(src_idxs):
                src_viewpoint = viewpoint_stack[src_idx]
                #c2w
                src_pose = src_viewpoint.world_view_transform.transpose(0, 1).inverse()
                src_K = src_viewpoint.K
                src_render_pkg = render(src_viewpoint, gaussians, pipe, background)
                src_projected_depth = src_render_pkg['surf_depth']
                src_rendered_normal = src_render_pkg['surf_normal']
                src_depth, _ = patchmatch_propagation(src_viewpoint, src_projected_depth, src_rendered_normal, viewpoint_stack, src_idxs, opt.patch_size)
                mask, depth_reprojected, x2d_src, y2d_src, relative_depth_diff = check_geometric_consistency(propagated_depth.unsqueeze(0), ref_K.unsqueeze(0), 
                                                                                                                 ref_pose.unsqueeze(0), src_depth.unsqueeze(0), 
                                                                                                                 src_K.unsqueeze(0), src_pose.unsqueeze(0), thre1=2, thre2=0.01)
                if geometric_counts is None:
                    geometric_counts = mask.to(torch.uint8)
                else:
                    geometric_counts += mask.to(torch.uint8)
                        
            cost = geometric_counts.squeeze()
            cost_mask = cost >= 2
                
            propagated_mask = valid_mask & error_mask & cost_mask
            propagated_mask = propagated_mask.squeeze(0)
            depth_mask = valid_mask & cost_mask

            projected_depth = projected_depth.squeeze(0)
                
            # Res Vis
            # normal_vis = (propagated_normal + 1.0) / 2.0
            # propagated_depth_vis = vis_depth(propagated_depth.detach().cpu().numpy())[0]
            # if not os.path.exists("cache/propagated_res"):
            #     os.makedirs("cache/propagated_res")
            # if iteration % 1000 == 0:
            #     torchvision.utils.save_image(normal_vis.cpu(), os.path.join("cache/propagated_res", f'{iteration}_normal_res.png'))
            #     imageio.imwrite(os.path.join("cache/propagated_res", f'{iteration}_depth_res.png'), propagated_depth_vis)
                
            ## vis depth
            # K = viewpoint_cam.K
            # cam2world = viewpoint_cam.world_view_transform.transpose(0, 1).inverse()
            # height, width = propagated_depth.shape
            # # Create a grid of 2D pixel coordinates
            # y, x = torch.meshgrid(torch.arange(0, height), torch.arange(0, width))
            # # Stack the 2D and depth coordinates to create 3D homogeneous coordinates
            # coordinates = torch.stack([x.to(propagated_depth.device), y.to(propagated_depth.device), torch.ones_like(propagated_depth)], dim=-1)
            # # Reshape the coordinates to (height * width, 3)
            # coordinates = coordinates.view(-1, 3).to(K.device).to(torch.float32)
            # # Reproject the 2D coordinates to 3D coordinates
            # coordinates_3D = (K.inverse() @ coordinates.T).T

            # # Multiply by depth
            # coordinates_3D *= propagated_depth.view(-1, 1)

            # # convert to the world coordinate
            # world_coordinates_3D = (cam2world[:3, :3] @ coordinates_3D.T).T + cam2world[:3, 3]

            # import open3d as o3d
            # point_cloud = o3d.geometry.PointCloud()
            # point_cloud.points = o3d.utility.Vector3dVector(world_coordinates_3D.detach().cpu().numpy())
            # o3d.io.write_point_cloud("cache/propagated_res/partpc.ply", point_cloud)
            # exit()
                
            valid_depth_sum =  depth_mask.sum() + 1e-5
            loss_depth += torch.abs((projected_depth[valid_mask] - propagated_depth[valid_mask])).sum() / valid_depth_sum    

            if propagated_mask.sum() > 100:
                gaussians.densify_from_depth_propagation(viewpoint_cam, propagated_depth, propagated_mask.to(torch.bool))

            return loss_depth