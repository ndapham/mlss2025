import os
import numpy as np
import matplotlib.pyplot as plt

def _overlay_scribbles(
    image, scribble, color_fg=(255, 0, 0), color_bg=(0, 0, 255), alpha=0.6
):
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Input image must be RGB")
    if scribble.shape != image.shape[:2]:
        raise ValueError("Scribble must match image spatial size")
    
    overlaid = image.copy().astype(np.float32)
    
    mask_fg = scribble == 1
    mask_bg = scribble == 0
    
    for mask, color in [(mask_fg, color_fg), (mask_bg, color_bg)]:
        for c in range(3):
            overlaid[..., c][mask] = (
                alpha * color[c] + (1 - alpha) * overlaid[..., c][mask]
            )
    
    return overlaid.astype(np.uint8)

def visualize(
    image: np.ndarray,
    scribbles: np.ndarray,
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    alpha: float=0.6
):
    """
    Shows a 1x3 subplot of:
      1. Original image overlaid with scribbles
      2. Ground truth segmentation mask
      3. Prediction mask
      
      Blue = background, Red = foreground.

    Parameters:
        image (H, W, 3)        : RGB image
        scribbles (H, W)       : scribble mask with values {0, 1, 255}
        ground_truth (H, W)    : ground truth mask with values {0, 1}
        prediction (H, W)      : predicted mask with values {0, 1}
        alpha (float)          : alpha blending for overlay
    """
    
    image_with_scribbles = _overlay_scribbles(image, scribbles, alpha=alpha)
    
    cmap = plt.get_cmap('bwr')
    
    _, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(image_with_scribbles)
    axes[0].set_title("Image + Scribbles")

    axes[1].imshow(ground_truth, cmap=cmap, vmin=0, vmax=1)
    axes[1].set_title("Ground Truth")

    axes[2].imshow(prediction, cmap=cmap, vmin=0, vmax=1)
    axes[2].set_title("Model Prediction")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.show()
