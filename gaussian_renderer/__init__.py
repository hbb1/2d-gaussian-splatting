#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from utils.point_utils import depth_to_normal

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, 
           override_color = None, record_transmittance=False, bg_gaussians=None, skip_geometric=False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    if bg_gaussians is None:
        means3D = pc.get_xyz
        opacity = pc.get_opacity
        scales = pc.get_scaling
        rotations = pc.get_rotation
        shs = pc.get_features
    else:
        means3D = torch.cat([pc.get_xyz, bg_gaussians.get_xyz])
        opacity = torch.cat([pc.get_opacity, bg_gaussians.get_opacity])
        scales = torch.cat([pc.get_scaling, bg_gaussians.get_scaling])
        rotations = torch.cat([pc.get_rotation, bg_gaussians.get_rotation])
        shs = torch.cat([pc.get_features, bg_gaussians.get_features])
    num_fg_points = pc.get_xyz.shape[0]

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros((means3D.shape[0], 4), dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        record_transmittance=record_transmittance,
        debug=False,
        # pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means2D = screenspace_points

    output = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = None,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = None)

    if record_transmittance:
        rendered_image, radii, allmap, transmittance_avg, num_covered_pixels = output
        transmittance_avg = transmittance_avg[:num_fg_points]
        num_covered_pixels = num_covered_pixels[:num_fg_points]
    else:
        rendered_image, radii, allmap = output
        transmittance_avg = num_covered_pixels = None
    radii = radii[:num_fg_points]
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    rets =  {"render": rendered_image,
            "viewspace_points": means2D,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "pixels_num":num_covered_pixels,
            "transmittance_avg": transmittance_avg
    }

    if skip_geometric:
        return rets
    
    # additional regularizations
    render_alpha = allmap[1:2]

    # get normal map
    # transform normal from view space to world space
    render_normal = allmap[2:5]
    render_normal = (render_normal.permute(1,2,0) @ (viewpoint_camera.world_view_transform[:3,:3].T)).permute(2,0,1)
    
    # get median depth map
    render_depth_median = allmap[5:6]
    render_depth_median = torch.nan_to_num(render_depth_median, 0, 0)

    # get expected depth map
    render_depth_expected = allmap[0:1]
    render_depth_expected = (render_depth_expected / render_alpha)
    render_depth_expected = torch.nan_to_num(render_depth_expected, 0, 0)
    
    # get depth distortion map
    render_dist = allmap[6:7]

    # psedo surface attributes
    # surf depth is either median or expected by setting depth_ratio to 1 or 0
    # for bounded scene, use median depth, i.e., depth_ratio = 1; 
    # for unbounded scene, use expected depth, i.e., depth_ration = 0, to reduce disk anliasing.
    surf_depth = render_depth_expected * (1-pipe.depth_ratio) + (pipe.depth_ratio) * render_depth_median
    
    # assume the depth points form the 'surface' and generate psudo surface normal for regularizations.
    surf_normal_expected = depth_to_normal(viewpoint_camera, render_depth_expected).permute(2,0,1)
    surf_normal = depth_to_normal(viewpoint_camera, render_depth_median).permute(2,0,1)
    # remember to multiply with accum_alpha since render_normal is unnormalized.
    surf_normal_expected = surf_normal_expected * (render_alpha).detach()
    surf_normal = surf_normal * (render_alpha).detach()


    rets.update({
            'rend_alpha': render_alpha,
            'rend_normal': render_normal,
            'rend_depth': render_depth_expected,
            'rend_dist': render_dist,
            'surf_depth': surf_depth,
            'surf_normal_expected': surf_normal_expected,
            'surf_normal': surf_normal,
    })

    return rets