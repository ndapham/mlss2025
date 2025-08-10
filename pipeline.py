import numpy as np
from typing import Callable
from util import load_dataset, store_predictions, visualize

# code taken from utils.py
def run_pipeline(
    segmenter: Callable[[np.ndarray, np.ndarray], np.ndarray],
    training_folder: str = "dataset/train",
    test_folder: str = "dataset/test",
    images_dir: str = "images",
    scribbles_dir: str = "scribbles",
    ground_truth_dir: str = "ground_truth",
    predictions_dir: str = "predictions",
    visualize_random: bool = True,
) -> None:

    # load training set
    images_train, scrib_train, gt_train, fnames_train, palette = load_dataset(
        training_folder, images_dir, scribbles_dir, ground_truth_dir
    )

    # inference on training set
    preds_train = np.stack(
        [segmenter(image, scribble).astype(np.uint8)
         for image, scribble in zip(images_train, scrib_train)],
        axis=0
    )

    # save training predictions
    store_predictions(preds_train, training_folder, predictions_dir, fnames_train, palette)

    # visualize a random training example
    if visualize_random and images_train.shape[0] > 0:
        idx = np.random.randint(0, images_train.shape[0])
        visualize(images_train[idx], scrib_train[idx], gt_train[idx], preds_train[idx])

    # load test set (no ground truth)
    images_test, scrib_test, fnames_test = load_dataset(
        test_folder, images_dir, scribbles_dir
    )

    # inference on test set
    preds_test = np.stack(
        [segmenter(image, scribble).astype(np.uint8)
         for image, scribble in zip(images_test, scrib_test)],
        axis=0
    )

    # save test predictions (reuse training palette)
    store_predictions(preds_test, test_folder, predictions_dir, fnames_test, palette)


if __name__ == "__main__":
    # example code 
    from util import segment_with_knn
    run_pipeline(segment_with_knn)

    
    