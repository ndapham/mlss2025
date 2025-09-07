import os
import sys
import matplotlib.pyplot as plt
from pydantic import config
import numpy as np 
import torch 
from torch.utils.data import DataLoader
from torchsummary import summary
from tqdm import tqdm
import wandb
import argparse
import cv2
from PIL import Image


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


def draw_convex_hull(mask, mode='convex'):
    
    img = np.zeros(mask.shape)
    contours, hier = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        if mode=='rect': # simple rectangle
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 255), -1)
        if mode=='convex': # minimum convex hull
            hull = cv2.convexHull(c)
            cv2.drawContours(img, [hull], 0, (255, 255, 255),-1)
        else: # minimum area rectangle
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(img, [box], 0, (255, 255, 255),-1)
    return img/255.

def post_process(probability, threshold, min_size):
    """
    This is slightly different from other kernels as we draw convex hull here itself.
    Post processing of each predicted mask, components with lesser number of pixels
    than `min_size` are ignored
    """
    # don't remember where I saw it
    mask = (cv2.threshold(probability, threshold, 1, cv2.THRESH_BINARY)[1])
    mask = draw_convex_hull(mask.astype(np.uint8))
    num_component, component = cv2.connectedComponents(mask.astype(np.uint8))
    predictions = np.zeros((375, 500), np.float32)
    num = 0
    for c in range(1, num_component):
        p = component == c
        if p.sum() > min_size:
            predictions[p] = 1
            num += 1
    return predictions, num



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
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location= device))
    print(f"Model loaded to device {device}...")

    test_images_folder = cfg["paths"]["test_images"]
    test_scribbles_folder = cfg["paths"]["test_scribbles"]

    images_test, scrib_test, fnames_test = load_dataset(
       test_images_folder, test_scribbles_folder
    )

    test_dataset = ScribbleDataset(images_test, scrib_test, fnames_test, augment_rate=0)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    preds = []
    file_names = []
    model.eval()
    for i, batch_infor in tqdm(enumerate(test_loader)):
        batch_images, batch_scribbles, batch_fnames = batch_infor
        B, C, H, W = batch_images.shape
        batch_images = batch_images.to(device)
        batch_scribbles = batch_scribbles.to(device)
        file_names.extend(batch_fnames)

        batch_input = torch.cat([batch_images, batch_scribbles], dim=1)
        batch_logits = model(batch_input)
        batch_logits = batch_logits.detach().cpu()
        preds.append(batch_logits)

    total_pred = torch.cat(preds, dim=0)
    for i in range(total_pred.shape[0]):
        image_tensor = total_pred[i]
        image_tensor = torch.sigmoid(image_tensor)
        image_tensor = image_tensor.view((H, W, 1))

        prediction, num = post_process(image_tensor.numpy(), 0.5, 200)
        # prediction = (image_tensor > 0.5).to(image_tensor.dtype).squeeze().numpy()
        mask_img = (prediction).astype(np.uint8)
        img = Image.fromarray(mask_img) 
        
        image_file_name = file_names[i]
        image_file_path = os.path.join(output_folder_path, image_file_name)
        img.save(image_file_path)
        


        
        
