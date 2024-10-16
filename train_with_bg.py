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

import os
import torch
from random import randint
from utils.loss_utils import l1_loss_appearance, ssim, l1_loss, ms_l1_loss
from gaussian_renderer import render, network_gui
import sys
import torch.nn.functional as F
from scene import Scene, GaussianModel, BgGaussianModel, AppearanceModel
from utils.general_utils import safe_state
from utils.patchmatch import process_propagation
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, render_net_image
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def prune_low_contribution_gaussians(gaussians, cameras, pipe, bg, K=5, prune_ratio=0.1):
    top_list = [None, ] * K
    for i, cam in enumerate(cameras):
        trans = render(cam, gaussians, pipe, bg, record_transmittance=True, skip_geometric=True)["transmittance_avg"]
        if top_list[0] is not None:
            m = trans > top_list[0]
            if m.any():
                for i in range(K - 1):
                    top_list[K - 1 - i][m] = top_list[K - 2 - i][m]
                top_list[0][m] = trans[m]
        else:
            top_list = [trans.clone() for _ in range(K)]

    contribution = torch.stack(top_list, dim=-1).mean(-1)
    tile = torch.quantile(contribution, prune_ratio)
    prune_mask = contribution < tile
    gaussians.prune_points(prune_mask)
    torch.cuda.empty_cache()
    
def ranking_loss(error, penalize_ratio=0.7, extra_weights=None , type='mean'):
    error, indices = torch.sort(error)
    # only sum relatively small errors
    s_error = torch.index_select(error, 0, index=indices[:int(penalize_ratio * indices.shape[0])])
    if extra_weights is not None:
        weights = torch.index_select(extra_weights, 0, index=indices[:int(penalize_ratio * indices.shape[0])])
        s_error = s_error * weights

    if type == 'mean':
        return torch.mean(s_error)
    elif type == 'sum':
        return torch.sum(s_error)

def normal_gradient_loss(rend_normal, gt_normal):
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).float().unsqueeze(0).unsqueeze(0).to(rend_normal.device) / 4
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).float().unsqueeze(0).unsqueeze(0).to(rend_normal.device) / 4

    rend_grad_x = F.conv2d(rend_normal, sobel_x.repeat(3, 1, 1, 1), padding=1, groups=3)
    rend_grad_y = F.conv2d(rend_normal, sobel_y.repeat(3, 1, 1, 1), padding=1, groups=3)

    gt_grad_x = F.conv2d(gt_normal, sobel_x.repeat(3, 1, 1, 1), padding=1, groups=3)
    gt_grad_y = F.conv2d(gt_normal, sobel_y.repeat(3, 1, 1, 1), padding=1, groups=3)

    loss_x = F.mse_loss(rend_grad_x, gt_grad_x)
    loss_y = F.mse_loss(rend_grad_y, gt_grad_y)

    return loss_x + loss_y

def edge_aware_normal_gradient_loss(gt_image, rend_normal, gt_normal, prior_normal_mask, edge_threshold=1):
    # Define Sobel filters
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).float().unsqueeze(0).unsqueeze(0).to(rend_normal.device) / 8
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).float().unsqueeze(0).unsqueeze(0).to(rend_normal.device) / 8

    # Compute gradients of rendered and ground truth normals
    rend_grad_x = F.conv2d(rend_normal, sobel_x.repeat(3, 1, 1, 1), padding=1, groups=3)
    rend_grad_y = F.conv2d(rend_normal, sobel_y.repeat(3, 1, 1, 1), padding=1, groups=3)

    gt_grad_x = F.conv2d(gt_normal, sobel_x.repeat(3, 1, 1, 1), padding=1, groups=3)
    gt_grad_y = F.conv2d(gt_normal, sobel_y.repeat(3, 1, 1, 1), padding=1, groups=3)

    # Compute gradients of gt_image for edge detection
    dI_dx = torch.cat([F.conv2d(gt_image[i].unsqueeze(0), sobel_x, padding=1) for i in range(gt_image.shape[0])])
    dI_dx = torch.mean(torch.abs(dI_dx), 1, keepdim=True)
    dI_dy = torch.cat([F.conv2d(gt_image[i].unsqueeze(0), sobel_y, padding=1) for i in range(gt_image.shape[0])])
    dI_dy = torch.mean(torch.abs(dI_dy), 1, keepdim=True)

    # Compute edge strength
    edge_strength = dI_dx + dI_dy

    # Create non-edge mask
    non_edge_mask = (edge_strength < edge_threshold).float()

    # Compute loss for gradients
    loss_x = F.mse_loss(rend_grad_x, gt_grad_x)
    loss_y = F.mse_loss(rend_grad_y, gt_grad_y)
    loss = loss_x + loss_y

    # Apply non-edge mask and prior_normal_mask
    masked_loss = loss * non_edge_mask * prior_normal_mask

    # Normalize by the number of non-edge pixels
    num_non_edge_pixels = torch.sum(non_edge_mask * prior_normal_mask) + 1e-6
    normalized_loss = torch.sum(masked_loss) / num_non_edge_pixels

    return normalized_loss

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    bg_gaussians = BgGaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, bg_gaussians)
    all_cameras = scene.getTrainCameras()

    bg_gaussians.training_setup(opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    pbar = tqdm(range(5000), desc="Training Background", unit="iteration")
    viewpoint_stack = None
    for iteration in pbar:
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        render_pkg = render(viewpoint_cam, gaussians, pipe, background, record_transmittance=False, 
                            bg_gaussians=bg_gaussians, skip_geometric=True)
        total_loss = ms_l1_loss(render_pkg["render"][None], viewpoint_cam.original_image.cuda()[None])
        total_loss.backward()
        bg_gaussians.optimizer.step()
        bg_gaussians.optimizer.zero_grad(set_to_none = True)
        pbar.set_postfix({"Loss": f"{total_loss.item():.4f}"})
    bg_gaussians.optimizer = None
    # TODO: Trim unused background point
    # prune_low_contribution_gaussians(bg_gaussians, all_cameras, pipe, background, 

    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)
    
    ema_loss_for_log = 0.0
    ema_depth_for_log = 0.0
    ema_normal_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    if dataset.use_decoupled_appearance:
        appearances = AppearanceModel(len(all_cameras))
        appearances.training_setup(opt)
    else:
        appearances = None

    for iteration in range(first_iter, opt.iterations + 1):        

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()
            bg_gaussians.oneupSHdegree()

        viewpoint_idx = randint(0, len(all_cameras)-1)
        viewpoint_cam = all_cameras[viewpoint_idx]
        render_pkg = render(viewpoint_cam, gaussians, pipe, background, record_transmittance=False, bg_gaussians=bg_gaussians, 
                            skip_geometric=True)
        # render_pkg = render(viewpoint_cam, gaussians, pipe, background, record_transmittance=(iteration < opt.densify_until_iter))
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss_appearance(image, gt_image, appearances, viewpoint_idx) # use L1 loss for the transformed image if using decoupled appearance
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        
        rend_dist = render_pkg["rend_dist"]
        lambda_dist = opt.lambda_dist if iteration > 3000 else 0.0
        dist_loss = lambda_dist * (rend_dist).mean()

        # regularization
        if iteration > 15000:
            render_pkg = render(viewpoint_cam, gaussians, pipe, background, record_transmittance=False)
            lambda_normal = opt.lambda_normal if iteration > 15000 else 0.0
            lambda_depth = opt.propagation_begin if iteration > opt.propagation_begin else 0.0
            lambda_normal_prior = opt.lambda_normal_prior if iteration > 15000 else 0.0
            lambda_normal_gradient = opt.lambda_normal_gradient if iteration > 15000 else 0.0
            
            depth_loss = torch.tensor(0.).to("cuda")
            normal_loss = torch.tensor(0.).to("cuda")
            normal_prior_loss = torch.tensor(0.).to("cuda")
            
            rend_depth = render_pkg["rend_depth"]
            surf_depth = render_pkg["surf_depth"]
            if lambda_depth > 0 and viewpoint_cam.depth_prior is not None:
                depth_error = 0.6 * (surf_depth - viewpoint_cam.depth_prior).abs() + \
                                0.4 * (rend_depth - viewpoint_cam.depth_prior).abs()
                depth_mask = viewpoint_cam.depth_mask.unsqueeze(0)
                valid_depth_sum = depth_mask.sum() + 1e-5
                depth_loss += lambda_depth * (depth_error[depth_mask & ~torch.isnan(depth_error)].sum() / valid_depth_sum)

            rend_normal  = render_pkg['rend_normal']
            surf_normal_median = render_pkg['surf_normal']
            surf_normal_expected = render_pkg['surf_normal_expected']
            rend_alpha = render_pkg['rend_alpha']
            
            if lambda_normal > 0.0:
                normal_error = 0.6 * (1 - F.cosine_similarity(rend_normal, surf_normal_median, dim=0)) + \
                            0.4 * (1 - F.cosine_similarity(rend_normal, surf_normal_expected, dim=0))
                normal_error = ranking_loss(normal_error.view(-1), penalize_ratio=1.0, type='mean')
                normal_loss += lambda_normal * normal_error
                
            if lambda_normal_prior > 0 and dataset.w_normal_prior:
                prior_normal = viewpoint_cam.normal_prior * (rend_alpha).detach()
                prior_normal_mask = viewpoint_cam.normal_mask[0]

                normal_prior_error = 0.6 * (1 - F.cosine_similarity(prior_normal, rend_normal, dim=0)) + \
                                    0.4 * (1 - F.cosine_similarity(prior_normal, surf_normal_expected, dim=0))           
                normal_prior_error = ranking_loss(normal_prior_error[prior_normal_mask], 
                                                penalize_ratio=1.0, type='mean')
                
                normal_prior_loss = lambda_normal_prior * normal_prior_error
                if lambda_normal_gradient > 0.0:
                    normal_prior_loss += lambda_normal_gradient * normal_gradient_loss(surf_normal_median, prior_normal)
        else:
            depth_loss = torch.tensor(0.).to("cuda")
            normal_loss = torch.tensor(0.).to("cuda")
            normal_prior_loss = torch.tensor(0.).to("cuda")
        # loss
        total_loss = loss + dist_loss + depth_loss + normal_loss + normal_prior_loss
        
        total_loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_depth_for_log = 0.4 * depth_loss.item() + 0.6 * ema_depth_for_log
            ema_normal_for_log = 0.4 * normal_loss.item() + 0.6 * ema_normal_for_log


            if iteration % 10 == 0:
                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                    "depth": f"{ema_depth_for_log:.{5}f}",
                    "normal": f"{ema_normal_for_log:.{5}f}",
                    "Points": f"{len(gaussians.get_xyz)}"
                }
                progress_bar.set_postfix(loss_dict)

                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            if tb_writer is not None:
                tb_writer.add_scalar('train_loss_patches/dist_loss', ema_depth_for_log, iteration)
                tb_writer.add_scalar('train_loss_patches/normal_loss', ema_normal_for_log, iteration)

            # training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)


            # Densification
            if iteration < opt.densify_until_iter:
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter, None)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    prune_big_points = True if iteration > opt.opacity_reset_interval else False
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.opacity_cull, scene.cameras_extent, prune_big_points)
                
                # if render_pkg["transmittance_avg"] is not None:
                #     gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], 
                #                                                         radii[visibility_filter] * (render_pkg["transmittance_avg"][visibility_filter] > 0.01))
                # if iteration > 7000 and iteration % opt.split_interval == 0:
                #     gaussians.split_big_points(opt.max_screen_size)
                
                if iteration > opt.contribution_prune_from_iter and iteration % opt.contribution_prune_interval == 0:
                    if iteration % opt.opacity_reset_interval == opt.contribution_prune_interval or \
                        iteration % opt.opacity_reset_interval == opt.split_interval:
                        print("Skipped Pruning for", iteration)
                        continue
                    prune_low_contribution_gaussians(gaussians, all_cameras, pipe, background, 
                                                     K=1, prune_ratio=opt.contribution_prune_ratio)
                    print(f'Num gs after contribution prune: {len(gaussians.get_xyz)}')

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                # visible = radii > 0
                # gaussians.optimizer.step(visible, radii.shape[0])
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
                if appearances is not None:
                    appearances.optimizer.step()
                    appearances.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

        with torch.no_grad():        
            if network_gui.conn == None:
                network_gui.try_connect(dataset.render_items)
            while network_gui.conn != None:
                try:
                    net_image_bytes = None
                    custom_cam, do_training, keep_alive, scaling_modifer, render_mode = network_gui.receive()
                    if custom_cam != None:
                        render_pkg = render(custom_cam, gaussians, pipe, background, scaling_modifer)   
                        net_image = render_net_image(render_pkg, dataset.render_items, render_mode, custom_cam)
                        net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                    metrics_dict = {
                        "#": gaussians.get_opacity.shape[0],
                        "loss": ema_loss_for_log
                        # Add more metrics as needed
                    }
                    # Send the data
                    network_gui.send(net_image_bytes, dataset.source_path, metrics_dict)
                    if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                        break
                except Exception as e:
                    # raise e
                    network_gui.conn = None

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

@torch.no_grad()
def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/reg_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        from utils.general_utils import colormap
                        depth = render_pkg["surf_depth"]
                        norm = depth.max()
                        depth = depth / norm
                        depth = colormap(depth.cpu().numpy()[0], cmap='turbo')
                        tb_writer.add_images(config['name'] + "_view_{}/depth".format(viewpoint.image_name), depth[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)

                        try:
                            rend_alpha = render_pkg['rend_alpha']
                            rend_normal = render_pkg["rend_normal"] * 0.5 + 0.5
                            surf_normal = render_pkg["surf_normal"] * 0.5 + 0.5
                            tb_writer.add_images(config['name'] + "_view_{}/rend_normal".format(viewpoint.image_name), rend_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/surf_normal".format(viewpoint.image_name), surf_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/rend_alpha".format(viewpoint.image_name), rend_alpha[None], global_step=iteration)

                            rend_dist = render_pkg["rend_dist"]
                            rend_dist = colormap(rend_dist.cpu().numpy()[0])
                            tb_writer.add_images(config['name'] + "_view_{}/rend_dist".format(viewpoint.image_name), rend_dist[None], global_step=iteration)
                        except:
                            pass

                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[1, 7_000, 20_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[1, 7_000, 20_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint)

    # All done
    print("\nTraining complete.")