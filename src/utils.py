import json
import numpy as np
from PIL import Image

def load_mask(path: str, threshold: int = 128) -> np.ndarray:
    img = Image.open(path).convert("L")
    arr = np.array(img)
    binary_mask = (arr >= threshold).astype(np.uint8)
    return binary_mask

def load_lines(path: str) -> dict:
    with open(path, "r") as f:
        data = json.load(f)
    required_keys = ["center", "left", "right"]
    for key in required_keys:
        if key not in data:
            data[key] = []
    return data

def save_results(results: dict, path: str):
    with open(path, "w") as f:
        json.dump(results, f, indent=4)

def compare_file_pairs(gt_mask_path: str, pred_mask_path: str,
                       gt_lines_path: str, pred_lines_path: str):
    mask_gt = load_mask(gt_mask_path)
    mask_pred = load_mask(pred_mask_path)
    gt_lines = load_lines(gt_lines_path)
    pred_lines = load_lines(pred_lines_path)
    return mask_gt, mask_pred, gt_lines, pred_lines
