import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from typing import Any
import yaml

def read_yaml_file(yaml_path):
    with open(yaml_path, "r") as f:
        inf = yaml.safe_load(f)
    return inf


def _open_image(path, convert_to):
    if convert_to == "RGB":
        return Image.open(path).convert("RGB")
    if convert_to == "grayscale":
        return Image.open(path).convert("L")
    return np.array(Image.open(path))

def _get_file_names(folder):
    return sorted(
        [file for file in os.listdir(folder) if not file.startswith('.')]
    )

def _load_images(images_folder_path, convert_to):
    filenames = _get_file_names(images_folder_path)
    filepaths = [
        os.path.join(images_folder_path, filename) for filename in filenames
    ]
    return np.stack([_open_image(file, convert_to) for file in filepaths])

def _get_palette(ground_truth_dir, filename):
    filepath = os.path.join(ground_truth_dir, filename)
    return Image.open(filepath).getpalette()

def _get_filenames(scribbles_dir):
    filenames = _get_file_names(scribbles_dir)
    return filenames

def load_dataset(
    images_dir: str,
    scribbles_dir: str,
    ground_truth_dir: str | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Any]:
    """
    Load images, scribbles, and ground truth masks from a dataset folder.
    
    Args:
        folder_path (str): Path to the dataset folder (e.g., 'dataset/training').
        images_dir (str): folder name for images.
        scribbles_dir (str): folder name for scribbles.
        ground_truth_dir (str): folder name for ground truth images.
        
        
    Returns:
        images (np.ndarray): Array of shape (N, H, W, 3) with RGB images.
        scribbles (np.ndarray): Array of shape (N, H, W) with scribble labels.
        ground_truth (np.ndarray): Array of shape (N, H, W) with class labels.
        filenames (list[str]): List of filenames for storing predictions
        palette (_type_): _description_
    """

    images = _load_images(images_dir, "RGB")
    scribbles = _load_images(scribbles_dir, "grayscale")
    filenames = _get_filenames(scribbles_dir)
    if ground_truth_dir is None:
        return images, scribbles, filenames
    
    ground_truth = _load_images(ground_truth_dir, None)
    palette = _get_palette(ground_truth_dir, filenames[0])
    return images, scribbles, ground_truth, filenames, palette

def store_predictions(
    predictions: np.ndarray,
    folder_path: str,
    predictions_dir: str,
    filenames: list[str],
    palette: Any
):
    """Takes a stack of segmented images and stores them indvidually in the given folder.

    Args:
        predictions (np.ndarray): Array of shape (N, H, W) with predicted class labels.
        folder_path (str): Path to the dataset folder (e.g., 'dataset/training').
        predictions_dir (str): folder name for predictions.
        storage_info (Any): Useful info from load_dataset method for storing.
    """
    pred_dir_path = os.path.join(folder_path, predictions_dir)
    if not os.path.exists(pred_dir_path):
        os.makedirs(pred_dir_path)
    for filename, pred_array in zip(filenames, predictions):
        filepath = os.path.join(pred_dir_path, filename)
        pred_image = Image.fromarray(pred_array.astype(np.uint8), mode='P')
        pred_image.putpalette(palette)
        pred_image.save(filepath)