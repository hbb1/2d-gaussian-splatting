import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import os
import glob
from skimage.morphology import binary_dilation, disk
import argparse

import trimesh
from pathlib import Path
import subprocess

import sys
import render_utils as rend_util
from tqdm import tqdm

def cull_scan(scan, mesh_path, result_mesh_file, instance_dir):
    
    # load poses
    image_dir = '{0}/images'.format(instance_dir)
    image_paths = sorted(glob.glob(os.path.join(image_dir, "*.png")))
    n_images = len(image_paths)
    cam_file = '{0}/cameras.npz'.format(instance_dir)
    camera_dict = np.load(cam_file)
    scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(n_images)]
    world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(n_images)]

    intrinsics_all = []
    pose_all = []
    for scale_mat, world_mat in zip(scale_mats, world_mats):
        P = world_mat @ scale_mat
        P = P[:3, :4]
        intrinsics, pose = rend_util.load_K_Rt_from_P(None, P)
        intrinsics_all.append(torch.from_numpy(intrinsics).float())
        pose_all.append(torch.from_numpy(pose).float())
    
    # load mask
    mask_dir = '{0}/mask'.format(instance_dir)
    mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.png")))
    masks = []
    for p in mask_paths:
        mask = cv2.imread(p)
        masks.append(mask)

    # hard-coded image shape
    W, H = 1600, 1200

    # load mesh
    mesh = trimesh.load(mesh_path)
    
    # load transformation matrix

    vertices = mesh.vertices

    # project and filter
    vertices = torch.from_numpy(vertices).cuda()
    vertices = torch.cat((vertices, torch.ones_like(vertices[:, :1])), dim=-1)
    vertices = vertices.permute(1, 0)
    vertices = vertices.float()

    sampled_masks = []
    for i in tqdm(range(n_images),  desc="Culling mesh given masks"):
        pose = pose_all[i]
        w2c = torch.inverse(pose).cuda()
        intrinsic = intrinsics_all[i].cuda()

        with torch.no_grad():
            # transform and project
            cam_points = intrinsic @ w2c @ vertices
            pix_coords = cam_points[:2, :] / (cam_points[2, :].unsqueeze(0) + 1e-6)
            pix_coords = pix_coords.permute(1, 0)
            pix_coords[..., 0] /= W - 1
            pix_coords[..., 1] /= H - 1
            pix_coords = (pix_coords - 0.5) * 2
            valid = ((pix_coords > -1. ) & (pix_coords < 1.)).all(dim=-1).float()
            
            # dialate mask similar to unisurf
            maski = masks[i][:, :, 0].astype(np.float32) / 256.
            maski = torch.from_numpy(binary_dilation(maski, disk(24))).float()[None, None].cuda()

            sampled_mask = F.grid_sample(maski, pix_coords[None, None], mode='nearest', padding_mode='zeros', align_corners=True)[0, -1, 0]

            sampled_mask = sampled_mask + (1. - valid)
            sampled_masks.append(sampled_mask)

    sampled_masks = torch.stack(sampled_masks, -1)
    # filter
    
    mask = (sampled_masks > 0.).all(dim=-1).cpu().numpy()
    face_mask = mask[mesh.faces].all(axis=1)

    mesh.update_vertices(mask)
    mesh.update_faces(face_mask)
    
    # transform vertices to world 
    scale_mat = scale_mats[0]
    mesh.vertices = mesh.vertices * scale_mat[0, 0] + scale_mat[:3, 3][None]
    mesh.export(result_mesh_file)
    del mesh
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Arguments to evaluate the mesh.'
    )

    parser.add_argument('--input_mesh', type=str,  help='path to the mesh to be evaluated')
    parser.add_argument('--scan_id', type=str,  help='scan id of the input mesh')
    parser.add_argument('--output_dir', type=str, default='evaluation_results_single', help='path to the output folder')
    parser.add_argument('--mask_dir', type=str,  default='mask', help='path to uncropped mask')
    parser.add_argument('--DTU', type=str,  default='Offical_DTU_Dataset', help='path to the GT DTU point clouds')
    args = parser.parse_args()

    Offical_DTU_Dataset = args.DTU
    out_dir = args.output_dir
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    scan = args.scan_id
    ply_file = args.input_mesh
    print("cull mesh ....")
    result_mesh_file = os.path.join(out_dir, "culled_mesh.ply")
    cull_scan(scan, ply_file, result_mesh_file, instance_dir=os.path.join(args.mask_dir, f'scan{args.scan_id}'))

    script_dir = os.path.dirname(os.path.abspath(__file__))
    cmd = f"python {script_dir}/eval.py --data {result_mesh_file} --scan {scan} --mode mesh --dataset_dir {Offical_DTU_Dataset} --vis_out_dir {out_dir}"
    os.system(cmd)