import os
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from typing import Any
from sklearn.metrics import jaccard_score
import torch
from torch.nn.functional import softmax


######### Reproducibility
def set_seed(seed: int = 33):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


######### Methods for baseline model

def segment_with_knn(
    image: np.ndarray,
    scribble: np.ndarray,
    k: int = 3
) -> np.ndarray:
    """
    Segment an image using K-Nearest Neighbors classifier based on RGB scribble.

    Parameters:
        image (np.ndarray): Color image of shape (H, W, 3).
        scribble (np.ndarray): Scribble mask of shape (H, W) with values:
                                0 (background), 1 (foreground), 255 (unmarked).
        k (int): Number of neighbors to use in KNN.

    Returns:
        np.ndarray: Predicted segmentation mask of shape (H, W) with values 0 or 1.
    """
    H, W, C = image.shape
    assert C == 3, "Image must be RGB."

    # Reshape image to (H*W, 3)
    image_flat = image.reshape(-1, 3)

    # Flatten scribble mask
    scribbles_flat = scribble.flatten()

    # Create mask for labeled and unlabeled pixels
    labeled_mask = (scribbles_flat != 255)
    unlabeled_mask = (scribbles_flat == 255)

    # Prepare training data
    X_train = image_flat[labeled_mask]
    y_train = scribbles_flat[labeled_mask]

    # Prepare test data
    X_test = image_flat[unlabeled_mask]

    # Train KNN
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    # Predict on unlabeled pixels
    y_pred = knn.predict(X_test)

    # Reconstruct full prediction mask
    predicted_mask = np.zeros_like(scribbles_flat)
    predicted_mask[labeled_mask] = y_train
    predicted_mask[unlabeled_mask] = y_pred

    # Reshape to (H, W)
    return predicted_mask.reshape(H, W)




#### eval metric with mIoU ####

def evaluate_binary_miou(
    predictions: np.ndarray,
    ground_truths: np.ndarray,
) -> dict[str, float]:
    """
    Compute IoU for background (class 0) and object (class 1), and return mean IoU (mIoU).

    Args:
        predictions: np.ndarray of shape (N, H, W) or (H, W) with values {0, 1}.
        ground_truths: np.ndarray of shape (N, H, W) or (H, W) with values {0, 1}.

    Returns:
        Dict with keys:
            - 'iou_background'
            - 'iou_object'
            - 'miou'
    """

    ious = {}
    miou_accumulator = 0.0
    for cls_label, cls_key in [(0, 'iou_background'), (1, 'iou_object')]:
        pred_mask = (predictions == cls_label)
        gt_mask = (ground_truths == cls_label)

        intersection = np.logical_and(pred_mask, gt_mask).sum(dtype=np.float64)
        union = np.logical_or(pred_mask, gt_mask).sum(dtype=np.float64)

        # if a class is absent in both prediction and GT (union == 0), treat IoU as 1.0
        iou = 1.0 if union == 0 else float(intersection / union)
        ious[cls_key] = iou
        miou_accumulator += iou

    ious['miou'] = miou_accumulator / 2.0
    return ious