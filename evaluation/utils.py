import json
import numpy as np
from PIL import Image


def load_mask(path: str, threshold: int = 128) -> np.ndarray:
    """
    Load a mask image and convert it to a binary numpy array.
    Any pixel >= threshold is considered foreground (1).
    """
    img = Image.open(path).convert("L")  # Convert to grayscale
    arr = np.array(img)
    binary_mask = (arr >= threshold).astype(np.uint8)
    return binary_mask


def load_lines(path: str) -> dict:
    """
    Load line coordinates from JSON file.
    Expected JSON format:
    {
        "center": [[x1, y1], [x2, y2], ...],
        "left": [[x1, y1], [x2, y2], ...],
        "right": [[x1, y1], [x2, y2], ...]
    }
    """
    with open(path, "r") as f:
        data = json.load(f)

    # Ensure keys exist
    required_keys = ["center", "left", "right"]
    for key in required_keys:
        if key not in data:
            data[key] = []

    return data


def save_results(results: dict, path: str):
    """
    Save evaluation results as a JSON file.
    """
    with open(path, "w") as f:
        json.dump(results, f, indent=4)


def compare_file_pairs(gt_mask_path: str, pred_mask_path: str,
                       gt_lines_path: str, pred_lines_path: str):
    """
    Load ground truth and prediction pairs (mask + lines).
    Returns mask_gt, mask_pred, gt_lines, pred_lines
    """
    mask_gt = load_mask(gt_mask_path)
    mask_pred = load_mask(pred_mask_path)

    gt_lines = load_lines(gt_lines_path)
    pred_lines = load_lines(pred_lines_path)

    return mask_gt, mask_pred, gt_lines, pred_lines
