import numpy as np
from shapely.geometry import Point, Polygon

def iou_score(mask_gt: np.ndarray, mask_pred: np.ndarray) -> float:
    mask_gt = mask_gt.astype(bool)
    mask_pred = mask_pred.astype(bool)
    intersection = np.logical_and(mask_gt, mask_pred).sum()
    union = np.logical_or(mask_gt, mask_pred).sum()
    return intersection / union if union != 0 else 0.0

def anchor_score(gt_coords: np.ndarray, pred_coords: np.ndarray) -> float:
    if gt_coords.shape != pred_coords.shape:
        return 0.0
    gt_points = gt_coords.reshape(-1, 2)
    pred_points = pred_coords.reshape(-1, 2)
    distances = np.sqrt(np.sum((gt_points - pred_points)**2, axis=1))
    mean_dist = np.mean(distances)
    min_coords = np.min(gt_points, axis=0)
    max_coords = np.max(gt_points, axis=0)
    bbox_diag = np.linalg.norm(max_coords - min_coords)
    if bbox_diag < 1:
        return 0.0
    score = 1.0 - (mean_dist / bbox_diag)
    return max(0.0, score)

def boolean_score(pred_coords: np.ndarray) -> int:
    try:
        points = pred_coords.reshape(-1, 2)
        left_edge_pts = list(map(tuple, points[0:2]))
        right_edge_pts = list(map(tuple, points[2:4]))
        center_line_pts = list(map(tuple, points[4:6]))
        polygon_pts = [left_edge_pts[0], left_edge_pts[1], right_edge_pts[1], right_edge_pts[0]]
        polygon = Polygon(polygon_pts)
        p1 = Point(center_line_pts[0])
        p2 = Point(center_line_pts[1])
        is_p1_inside = polygon.contains(p1)
        is_p2_inside = polygon.contains(p2)
        return 1 if is_p1_inside and is_p2_inside else 0
    except (IndexError, ValueError):
        return 0

