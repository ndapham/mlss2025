import os
import sys
import matplotlib.pyplot as plt
from pydantic import config
import torch 
from torch.utils.data import DataLoader
from torchsummary import summary
from tqdm import tqdm
import wandb
import argparse

root = os.path.abspath(os.path.join(os.path.dirname("."), '../..'))
sys.path.insert(0, root)

from src.utils.io import load_dataset, read_yaml_file
from src.utils.util import *
from src.models.unet_2 import UNet2
from src.models.unet import UNet
from src.models.res_unet import ResUNet
from src.models.res_attn_unet import ResAttentionUNet
from src.loader.datasets import ScribbleDataset 
from src.criterion.metric_based import DiceBCELoss
from src.models.dino import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, default="/Users/pducanh/Desktop/mlss2025/data/output/checkpoints/unet_100_88_0.5952545125453903.pt")
    parser.add_argument('--config_path', type=str)

    args = parser.parse_args()
    checkpoint_path = args.checkpoint_path
    config_path = args.config_path 
    cfg = read_yaml_file(config_path)


    file_name = checkpoint_path.split("/")[-1]
    print(f"Loading checkpoint: {file_name} ...")

    model_name = file_name.split("_")[0]

    prediction_folder= cfg["paths"]["predictions"]
    output_folder_path = os.path.join(prediction_folder, file_name)
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    if model_name == "unet":
        model = UNet()
    elif model_name == "resunet":
        model = ResUNet()
    elif model_name == "unet2":
        model = UNet2()
    else: 
        model = ResAttentionUNet()

    model.load_state_dict(torch.load(checkpoint_path, map_location= "cpu"))
    print("Model loaded ...")

    test_images_folder = cfg["paths"]["test_images"]
    test_scribbles_folder = cfg["paths"]["test_scribbles"]
    test_ground_truth_folder = cfg["paths"]["test_ground_truth"]

    images_test, scrib_test, gt_test, fnames_test, palette = load_dataset(
       test_images_folder, test_scribbles_folder, test_ground_truth_folder
    )
    device = "cpu"

    test_dataset = ScribbleDataset(images_test, scrib_test, fnames_test, gt_test, augment_rate=0)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    preds = []
    file_names = []

    for i, batch_infor in tqdm(enumerate(test_loader)):
        batch_images, batch_scribbles, batch_annos, batch_fnames = batch_infor
        B, C, H, W = batch_images.shape
        batch_images = batch_images.to(device)
        batch_scribbles = batch_scribbles.to(device)
        batch_annos = batch_annos.to(device)
        file_names.extend(batch_fnames)

        batch_input = torch.cat([batch_images, batch_scribbles], dim=1)
        batch_logits = model(batch_input)

        
        batch_preds = (batch_logits > 0).to(batch_logits.dtype) # (B , 1, H, W)
        preds.append(batch_preds)

    total_pred = torch.cat(preds, dim=0).detach().cpu().numpy()
    for i in range(total_pred.shape[0]):
        image_tensor = total_pred[i]
        image_tensor = image_tensor.view(1, H, W)
        convert_to_pil_image = transforms.ToPILImage()
        pil_image = convert_to_pil_image(image_tensor)
        image_file_name = file_names[i]
        image_file_path = os.path.join(output_folder_path, image_file_name)
        pil_image.save(image_file_path)


        
        
