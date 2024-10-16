import click
import torch
import glob
from PIL import Image
import os
from tqdm import tqdm

@click.command()
@click.option('--source_path', '-s', required=True, help='Path to the dataset')
def main(source_path: str) -> None:
    normal_predictor = torch.hub.load("Stable-X/StableNormal", "StableNormal", trust_repo=True)
    output_normal_dir = os.path.join(source_path, "normals")
    output_mask_dir = os.path.join(source_path, "masks")
    os.makedirs(output_normal_dir, exist_ok=True)
    os.makedirs(output_mask_dir, exist_ok=True)
    
    image_pattern = f"{source_path}/images/*.jpg"
    for image_path in tqdm(glob.glob(image_pattern, recursive=True)[::3]):
        image_name = os.path.basename(image_path.split("/")[-1]).split(".")[0]
        
        input_image = Image.open(image_path)
        output_normal_path = os.path.join(output_normal_dir, image_name+'.png')
        if not os.path.exists(output_normal_path):
            normal_image = normal_predictor(input_image)
            normal_image.save(output_normal_path)

if __name__ == "__main__":
    main()