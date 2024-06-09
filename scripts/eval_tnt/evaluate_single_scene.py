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
import json


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Arguments to evaluate the mesh.'
    )

    parser.add_argument('--input_mesh', type=str,  help='path to the mesh to be evaluated')
    parser.add_argument('--scene', type=str,  help='scan id of the input mesh')
    parser.add_argument('--output_dir', type=str, default='evaluation_results_single', help='path to the output folder')
    parser.add_argument('--TNT', type=str,  default='Offical_DTU_Dataset', help='path to the GT DTU point clouds')
    args = parser.parse_args()


    TNT_Dataset = args.TNT
    out_dir = args.output_dir
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    scene = args.scene
    ply_file = args.input_mesh
    result_mesh_file = os.path.join(out_dir, "culled_mesh.ply")
    # read scene.json
    f"python run.py --dataset-dir {ply_file} --traj-path {TNT_Dataset}/{scene}/{scene}_COLMAP_SfM.log --ply-path {TNT_Dataset}/{scene}/{scene}_COLMAP.ply"