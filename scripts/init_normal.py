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
    normal_predictor = torch.hub.load("Stable-X/StableNormal", "StableNormal_turbo", 
                                      trust_repo=True, yoso_version='yoso-normal-v1-0')
    
    # Load BiRefNet model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    birefnet = AutoModelForImageSegmentation.from_pretrained('zhengpeng7/BiRefNet', trust_remote_code=True)
    birefnet.to(device)
    birefnet.eval()

    output_normal_dir = os.path.join(source_path, "normals")
    output_mask_dir = os.path.join(source_path, "masks")
    os.makedirs(output_normal_dir, exist_ok=True)
    os.makedirs(output_mask_dir, exist_ok=True)
    
    def generate_mask(image):
        image_size = (1024, 1024)
        transform_image = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        input_images = transform_image(image).unsqueeze(0).to(device)

        with torch.no_grad():
            preds = birefnet(input_images)[-1].sigmoid().cpu()
        pred = preds[0].squeeze()
        mask = transforms.ToPILImage()(pred)
        mask = mask.resize(image.size)
        return mask

    image_pattern = f"{source_path}/images/*.jpeg"
    for image_path in tqdm(glob.glob(image_pattern, recursive=True)):
        image_name = os.path.basename(image_path.split("/")[-1]).split(".")[0]
        
        input_image = Image.open(image_path)
        output_normal_path = os.path.join(output_normal_dir, image_name+'.png')
        output_mask_path = os.path.join(output_mask_dir, image_name+'.png')

        # Load or generate mask
        if os.path.exists(output_mask_path):
            mask = Image.open(output_mask_path)
        else:
            mask = generate_mask(input_image)
            mask.save(output_mask_path)

        # Apply mask to input image
        input_array = np.array(input_image)
        mask_array = np.array(mask)
        # masked_input = np.where(mask_array[:,:, np.newaxis] > 128, input_array, 0)
        masked_input = np.where(mask_array[:,:] > 128, input_array, 0)
        masked_input_image = Image.fromarray(masked_input)

        # Generate normal map if it doesn't exist
        if not os.path.exists(output_normal_path):
            normal_image = normal_predictor(masked_input_image)
            normal_image.save(output_normal_path)

if __name__ == "__main__":
    main()