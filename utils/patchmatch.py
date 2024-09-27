import torch
from gaussian_renderer import render
from utils.graphics_utils import patchmatch_propagation, check_geometric_consistency, propagation_installed


def process_propagation(viewpoint_stack, viewpoint_cam, gaussians, pipe, background, iteration, opt, src_idxs):
    if not propagation_installed:
        return
    
    with torch.no_grad():
        if iteration > opt.propagation_begin and iteration < opt.propagation_after and (iteration % opt.propagation_interval == 0):
            render_pkg = render(viewpoint_cam, gaussians, pipe, background)
            projected_depth = render_pkg["rend_depth"] / render_pkg['rend_alpha']
            rendered_normal = render_pkg["rend_normal"] / render_pkg['rend_alpha'] if viewpoint_cam.normal_prior is None else viewpoint_cam.normal_prior
            R_w2c = torch.tensor(viewpoint_cam.R.T).cuda().to(torch.float32)
            rendered_normal_cam = (R_w2c @ rendered_normal.view(3, -1)).view(3, viewpoint_cam.image_height, viewpoint_cam.image_width)                
            
            # get the propagated depth
            propagated_depth, propagated_normal = patchmatch_propagation(viewpoint_cam, projected_depth, rendered_normal_cam, viewpoint_stack, src_idxs, opt.patch_size)
            propagated_normal = propagated_normal.permute(2, 0, 1)
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
                src_projected_depth = src_render_pkg["rend_depth"] / src_render_pkg['rend_alpha']
                src_rendered_normal = src_render_pkg["rend_normal"] / src_render_pkg['rend_alpha'] if src_viewpoint.normal is None else src_viewpoint.normal
                R_w2c = torch.tensor(src_viewpoint.R.T).cuda().to(torch.float32)
                src_rendered_normal_cam = (R_w2c @ src_rendered_normal.view(3, -1)).view(3, src_viewpoint.image_height, src_viewpoint.image_width)                
                
                src_depth, _ = patchmatch_propagation(src_viewpoint, src_projected_depth, src_rendered_normal_cam, viewpoint_stack, src_idxs, opt.patch_size)
                mask, depth_reprojected, x2d_src, y2d_src, relative_depth_diff = check_geometric_consistency(propagated_depth.unsqueeze(0), ref_K.unsqueeze(0), 
                                                                                                                 ref_pose.unsqueeze(0), src_depth.unsqueeze(0), 
                                                                                                                 src_K.unsqueeze(0), src_pose.unsqueeze(0), thre1=5, thre2=0.05)
                if geometric_counts is None:
                    geometric_counts = mask.to(torch.uint8)
                else:
                    geometric_counts += mask.to(torch.uint8)
                        
            cost = geometric_counts.squeeze()
            cost_mask = cost >= 1
                
            propagated_mask = valid_mask & error_mask & cost_mask
            propagated_mask = propagated_mask.squeeze(0)
            depth_mask = valid_mask & cost_mask
            projected_depth = projected_depth.squeeze(0)
            
            viewpoint_cam.depth_prior = projected_depth
            viewpoint_cam.depth_mask = depth_mask

            if propagated_mask.sum() > 100:
                gaussians.densify_from_depth_propagation(viewpoint_cam, propagated_depth, rendered_normal.permute(1, 2, 0), propagated_mask)
                # gaussians.densify_from_depth_propagation(viewpoint_cam, propagated_depth, propagated_normal, propagated_mask)