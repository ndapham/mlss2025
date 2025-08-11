import numpy as np
from sklearn.neighbors import KNeighborsClassifier


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