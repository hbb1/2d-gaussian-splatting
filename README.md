# 2D Gaussian Splatting for Geometrically Accurate Radiance Fields

[Project page](https://surfsplatting.github.io/) | [Paper](https://arxiv.org/pdf/2403.17888) | [Video](https://www.youtube.com/watch?v=oaHCtB6yiKU) | [Surfel Rasterizer (CUDA)](https://github.com/hbb1/diff-surfel-rasterization) | [Surfel Rasterizer (Python)](https://colab.research.google.com/drive/1qoclD7HJ3-o0O1R8cvV3PxLhoDCMsH8W?usp=sharing) | [DTU+COLMAP (3.5GB)](https://drive.google.com/drive/folders/1SJFgt8qhQomHX55Q4xSvYE2C6-8tFll9) | [Viewer Pre-built for Windows](https://drive.google.com/file/d/1DRFrtFUfz27QvQKOWbYXbRS2o2eSgaUT/view?usp=sharing)<br>

![Teaser image](assets/teaser.jpg)

This repo contains the official implementation for the paper "2D Gaussian Splatting for Geometrically Accurate Radiance Fields". Our work represents a scene with a set of 2D oriented disks (surface elements) and rasterizes the surfels with [perspective correct differentiable raseterization](https://colab.research.google.com/drive/1qoclD7HJ3-o0O1R8cvV3PxLhoDCMsH8W?usp=sharing). Our work also develops regularizations that enhance the reconstruction quality. We also devise meshing approaches for Gaussian splatting.


## ⭐ New Features 


### How to use
Firstly open the viewer, 
```shell
<path to downloaded/compiled viewer>/bin/SIBR_remoteGaussian_app_rwdi.exe
```
and then
```shell
# Monitor the training process
python train.py -s <path to COLMAP or NeRF Synthetic dataset> 
# View the trained model
python view.py -s <path to COLMAP or NeRF Synthetic dataset> -m <path to trained model> 
```


## Citation
If you find our code or paper helps, please consider citing:
```bibtex
@inproceedings{Huang2DGS2024,
    title={2D Gaussian Splatting for Geometrically Accurate Radiance Fields},
    author={Huang, Binbin and Yu, Zehao and Chen, Anpei and Geiger, Andreas and Gao, Shenghua},
    publisher = {Association for Computing Machinery},
    booktitle = {SIGGRAPH 2024 Conference Papers},
    year      = {2024},
    doi       = {10.1145/3641519.3657428}
}
```