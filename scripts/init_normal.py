import click
import torch
import glob
from PIL import Image
import os
from tqdm import tqdm
import numpy as np
from transformers import AutoModelForImageSegmentation
import torchvision.transforms as transforms

@click.command()
@click.option('--source_path', '-s', required=True, help='Path to the dataset')
def main(source_path: str) -> None:
    # Load StableNormal model
    normal_predictor = torch.hub.load("hugoycj/StableNormal", "StableNormal", 
                                      trust_repo=True, yoso_version='yoso-normal-v1-5',
                                      local_cache_dir='/workspace/code/InverseRendering/StableNormal_git/weights')

    output_normal_dir = os.path.join(source_path, "normals")
    os.makedirs(output_normal_dir, exist_ok=True)
    
    for image_path in tqdm(glob.glob(f"{source_path}/images/*.jpg", recursive=True)[::5] + \
                            glob.glob(f"{source_path}/images/*.jpeg", recursive=True)[::5] + \
                            glob.glob(f"{source_path}/images/*.png", recursive=True)[::5]):
        image_name = os.path.basename(image_path.split("/")[-1]).split(".")[0]
        
        input_image = Image.open(image_path)
        output_normal_path = os.path.join(output_normal_dir, image_name+'.png')

        # Generate normal map if it doesn't exist
        if not os.path.exists(output_normal_path):
            normal_image = normal_predictor(input_image, data_type='outdoor')
            normal_image.save(output_normal_path)

if __name__ == "__main__":
    main()