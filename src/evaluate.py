# src/evaluate.py
import numpy as np

def calculate_iou(pred_mask, gt_mask):
    """
    Calculates the Intersection over Union (IoU) score for segmentation masks.
    """
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    iou_score = intersection / union if union > 0 else 0
    return iou_score

def calculate_anchor_score(pred_lines, gt_lines):
    """
    Placeholder for the Anchor Score calculation.
    This will depend on the specific definition of the score in the project details.
    Typically, it involves measuring the distance between predicted and ground truth line endpoints.
    """
    # Example: Calculate Mean Squared Error between coordinates
    # This is a simplified version and should be replaced with the official metric.
    if not gt_lines or not pred_lines:
        return 0.0
    
    pred = np.array(pred_lines).flatten()
    gt = np.array(gt_lines).flatten()
    
    # Ensure arrays have the same length
    min_len = min(len(pred), len(gt))
    pred = pred[:min_len]
    gt = gt[:min_len]

    mse = np.mean((pred - gt) ** 2)
    # A lower MSE is better. You might need to invert or scale it.
    # For example, score = 1 / (1 + mse)
    score = 1 / (1 + np.sqrt(mse))
    return score

def calculate_boolean_score(pred_left_edge, pred_right_edge, pred_center_line):
    """
    Placeholder for the Boolean Score calculation.
    Checks if the predicted center line is located between the left and right edges.
    """
    # This requires a geometric check. A simple way is to check if the x-coordinates
    # of the centerline fall between the x-coordinates of the edges.
    # This is a simplified check and assumes lines are mostly vertical.
    
    # Assuming pred_left_edge = [x1, y1, x2, y2]
    # We take the average x-coordinate
    if not pred_left_edge or not pred_right_edge or not pred_center_line:
        return 0
        
    left_x_avg = (pred_left_edge[0] + pred_left_edge[2]) / 2
    right_x_avg = (pred_right_edge[0] + pred_right_edge[2]) / 2
    center_x_avg = (pred_center_line[0] + pred_center_line[2]) / 2

    # Ensure left is actually to the left of right
    min_x = min(left_x_avg, right_x_avg)
    max_x = max(left_x_avg, right_x_avg)

    if min_x < center_x_avg < max_x:
        return 1
    else:
        return 0
