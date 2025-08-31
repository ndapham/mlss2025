import os
import sys
import matplotlib.pyplot as plt
import torch 
from torch.utils.data import DataLoader
from torchsummary import summary
from tqdm import tqdm
import wandb
import argparse

root = os.path.abspath(os.path.join(os.path.dirname("."), '..'))
sys.path.insert(0, root)

from src.utils.io import load_dataset, read_yaml_file
from src.utils.util import *
from src.models.unet import UNet
from src.models.res_unet import ResUNet
from src.models.res_attn_unet import ResAttentionUNet
from src.loader.datasets import ScribbleDataset 
from src.criterion.metric_based import DiceBCELoss
from src.models.dino import *

set_seed(42)

class Train_state:
    """Track number of steps"""
    step: int = 0  



def train_epoch(cfg, train_state, model, data_loader, optimizer, device, criterion):
    model.train()
    model.to(device)

    total_loss = 0
    iou = 0

    for i, batch_infor in tqdm(enumerate(data_loader)):
        batch_images, batch_scribbles, batch_annos, batch_fnames = batch_infor
        batch_images = batch_images.to(device)
        batch_scribbles = batch_scribbles.to(device)
        batch_annos = batch_annos.to(device)

        batch_input = torch.cat([batch_images, batch_scribbles], dim=1)
        batch_preds = model(batch_input)

        loss = criterion(batch_preds, batch_annos)
        
        total_loss+= loss.item()
        iou_train = iou(batch_preds, batch_annos)
        iou += iou_train
        train_state.step += 1

        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return total_loss / len(data_loader), iou/len(data_loader), train_state

def eval_epoch(cfg, model, val_loader, device, criterion):
    model.eval()
    model.to(device)

    total_loss = 0
    iou = 0

    for i, batch_infor in tqdm(enumerate(val_loader)):
        batch_images, batch_scribbles, batch_annos, batch_fnames = batch_infor
        batch_images = batch_images.to(device)
        batch_scribbles = batch_scribbles.to(device)
        batch_annos = batch_annos.to(device)

        batch_input = torch.cat([batch_images, batch_scribbles], dim=1)
        batch_preds = model(batch_input)

        loss = criterion(batch_preds, batch_annos)
        total_loss+= loss.item()
        iou_val = iou(batch_preds, batch_annos)
        iou += iou_val


    return total_loss / len(val_loader), iou/len(val_loader)


def test_epoch(model, test_loader, criterion, device):
    model.eval()
    model.to(device)

    total_loss = 0
    preds = []
    annos = []
    fg_iou = 0
    bg_iou = 0
    miou = 0

    for i, batch_infor in tqdm(enumerate(test_loader)):
        B, C, H, W = batch_images.shape
        batch_images, batch_scribbles, batch_annos, batch_fnames = batch_infor
        batch_images = batch_images.to(device)
        batch_scribbles = batch_scribbles.to(device)
        batch_annos = batch_annos.to(device)

        batch_input = torch.cat([batch_images, batch_scribbles], dim=1)
        batch_preds = model(batch_input)

        loss = criterion(batch_preds, batch_annos)
        total_loss+= loss.item()
        
        preds.append(batch_preds)
        annos.append(batch_annos)
    
    total_pred = torch.cat(preds, dim=0).view(-1, H, W).detach().numpy()
    total_anno =  torch.cat(annos, dim=0).view(-1, H, W).detach().numpy()
    ious = evaluate_binary_miou(total_pred, total_anno)
    print("Results on test set:", ious)

    fg_iou = ious["iou_object"]
    bg_iou = ious["iou_background"]
    m_iou = ious["miou"]

    return  total_loss / len(test_loader), m_iou, fg_iou, bg_iou


def train(cfg, model_name):
    # Model
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Training on device: {device}")
    if model_name == "unet":
        model = UNet()
    elif model_name == "resunet":
        model = ResUNet()
    else: 
        model = ResAttentionUNet()
    model.to(device)

    # Data 
    train_images_folder = cfg["paths"]["train_images"]
    train_scribbles_folder = cfg["paths"]["train_scribbles"]
    train_ground_truth_folder = cfg["paths"]["train_ground_truth"]

    test_images_folder = cfg["paths"]["test_images"]
    test_scribbles_folder = cfg["paths"]["test_scribbles"]
    test_ground_truth_folder = cfg["paths"]["test_ground_truth"]

    # Train
    images_train, scrib_train, gt_train, fnames_train, palette = load_dataset(
       train_images_folder, train_scribbles_folder, train_ground_truth_folder
    )
    train_dataset = ScribbleDataset(images_train, scrib_train, fnames_train, gt_train, augment_rate=0.5)
    
    # Test 
    images_test, scrib_test, gt_test, fnames_test, palette = load_dataset(
       test_images_folder, test_scribbles_folder, test_ground_truth_folder
    )
    test_dataset = ScribbleDataset(images_test, scrib_test, fnames_test, gt_test, augment_rate=0)
    
    # Dataloader 
    train_loader = DataLoader(train_dataset, batch_size=cfg["training"]["batch_size"], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg["training"]["batch_size"], shuffle=True)

    # Loss
    criterion = DiceBCELoss()

    # Optimizer 
    model_optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["lr"])

    # Dino 
    if cfg["training"]["dino_epoch"]:
        dino_dataset = ScribbleDataset(images_train, scrib_train, fnames_train, gt_train, augment_rate=0)
        dino_loader = DataLoader(dino_dataset, batch_size=cfg["training"]["batch_size"], shuffle=True) 
        dino = None 
        if model_name == "unet":
            dino = DINO(UNet, UNet, device)
        elif model_name == "resunet":
            dino = DINO(ResUNet, ResUNet, device)
        else: 
            dino = DINO(ResAttentionUNet, ResAttentionUNet, device)
        dino_optimizer = torch.optim.AdamW(dino.parameters(), lr=1e-4)
        model = train_dino(dino, 
                            dino_loader,
                            dino_optimizer, 
                            device, 
                            num_epochs=cfg["training"]["dino_epoch"], 
                            tps=0.9,
                            tpt= 0.04,
                            beta= 0.9,
                            m= 0.9)
    # Wandb 
    wandb.login()
    run = wandb.init(
        # Set the project where this run will be logged
        project="mlss2025",
        name=f"{model_name}_{cfg["training"]["dino_epoch"]}",
        # Track hyperparameters and run metadata
        config={
            "name_method": model_name
        },
    )

    # Training Loop 
    best_val_iou = 1
    train_state = Train_state()
    for epoch in range(cfg["training"]["epochs"]):
        print("="*20 + "Epoch: %d"%(epoch) + "="*20)
        model.train()
        mean_train_loss, mean_train_iou, train_state = train_epoch(cfg, train_state, model, train_loader, model_optimizer, device, criterion)
        wandb.log({
                "train_loss": mean_train_loss,
                "train_iou": mean_train_iou
            }, step=train_state.step)
        print("Mean train loss: ", mean_train_iou)
        print("Mean Iou score: ", mean_train_iou)

        model.eval()
        with torch.no_grad():
            mean_test_loss, m_iou_score, fg_score, bg_score = test_epoch(model, test_loader, criterion, device)
        wandb.log({
                "test_loss": mean_test_loss,
                "miou_score": m_iou_score, 
                "fg_score": fg_score,
                "bg_score": bg_score
            }, step=train_state.step)
        print("Miou score:", m_iou_score)   

        if epoch % 30 == 1 or m_iou_score < best_val_iou:
            best_val_iou = m_iou_score
            save_file_name = f'{model_name}_{cfg["training"]["epochs"]}_{epoch}_{m_iou_score}.pt'
            save_path = os.path.join(cfg["paths"]["checkpoint"], save_file_name,)
            torch.save(model.state_dict(), save_path)
    return model
        



if __name__ == "__main__":
    cfg = read_yaml_file("src/configs/config.yml")
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="unet", choices=['unet', 'resunet', 'resattnunet'])
    parser.add_argument('--config_path', type=str, default="../configs/lora.json")

    args = parser.parse_args()
    model_name = args.model
    train(cfg, model_name)