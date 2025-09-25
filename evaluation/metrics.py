import numpy as np
from shapely.geometry import Point, Polygon, LineString


def iou_score(mask_gt: np.ndarray, mask_pred: np.ndarray) -> float:
    """
    Compute Intersection over Union (IoU) between ground truth and predicted masks.
    Both inputs should be binary numpy arrays of the same shape.
    """
    intersection = np.logical_and(mask_gt, mask_pred).sum()
    union = np.logical_or(mask_gt, mask_pred).sum()
    return intersection / union if union != 0 else 0.0


def dice_score(mask_gt: np.ndarray, mask_pred: np.ndarray) -> float:
    """
    Compute Dice Coefficient between ground truth and predicted masks.
    Both inputs should be binary numpy arrays of the same shape.
    """
    intersection = np.logical_and(mask_gt, mask_pred).sum()
    total = mask_gt.sum() + mask_pred.sum()
    return (2 * intersection) / total if total != 0 else 0.0


def anchor_score(gt_line: list, pred_line: list) -> float:
    """
    Custom metric: Anchor Score
    gt_line, pred_line are lists of (x, y) coordinates.
    Returns average point-to-point distance normalized by line length.
    """
    if len(gt_line) != len(pred_line) or len(gt_line) == 0:
        return 0.0

    gt_line = np.array(gt_line)
    pred_line = np.array(pred_line)

    distances = np.linalg.norm(gt_line - pred_line, axis=1)
    mean_dist = distances.mean()

    # Normalize by diagonal length of bounding box of GT line
    bbox_diag = np.linalg.norm(gt_line.max(axis=0) - gt_line.min(axis=0))
    return 1 - (mean_dist / bbox_diag) if bbox_diag > 0 else 0.0


def boolean_score(center_line: list, left_edge: list, right_edge: list) -> bool:
    """
    Boolean Score:
    Check if all points of the center line lie within the polygon
    formed by left and right edge lines.
    """
    # Create polygon by joining left edge + reversed right edge
    polygon_points = left_edge + right_edge[::-1]
    polygon = Polygon(polygon_points)

    for pt in center_line:
        if not polygon.contains(Point(pt)):
            return False
    return True


def compute_all_metrics(mask_gt: np.ndarray, mask_pred: np.ndarray,
                        gt_lines: dict, pred_lines: dict) -> dict:
    """
    Compute all metrics and return as dictionary.
    gt_lines and pred_lines should have keys: 'center', 'left', 'right'
    """
    results = {
        "IoU": iou_score(mask_gt, mask_pred),
        "Dice": dice_score(mask_gt, mask_pred),
        "Anchor_Center": anchor_score(gt_lines["center"], pred_lines["center"]),
        "Anchor_Left": anchor_score(gt_lines["left"], pred_lines["left"]),
        "Anchor_Right": anchor_score(gt_lines["right"], pred_lines["right"]),
        "Boolean": boolean_score(pred_lines["center"], gt_lines["left"], gt_lines["right"])
    }
    return results
