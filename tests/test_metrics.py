import numpy as np
from evaluation import metrics


def test_iou_score():
    mask_gt = np.array([[1, 1], [0, 0]])
    mask_pred = np.array([[1, 0], [0, 1]])
    score = metrics.iou_score(mask_gt, mask_pred)
    # Intersection = 1, Union = 3 → IoU = 1/3
    assert abs(score - 1/3) < 1e-6


def test_dice_score():
    mask_gt = np.array([[1, 1], [0, 0]])
    mask_pred = np.array([[1, 0], [0, 1]])
    score = metrics.dice_score(mask_gt, mask_pred)
    # Intersection = 1, Total = 4 → Dice = 2/4 = 0.5
    assert abs(score - 0.5) < 1e-6


def test_anchor_score_perfect_match():
    gt_line = [(0, 0), (1, 1), (2, 2)]
    pred_line = [(0, 0), (1, 1), (2, 2)]
    score = metrics.anchor_score(gt_line, pred_line)
    assert score == 1.0  # perfect match


def test_anchor_score_mismatch():
    gt_line = [(0, 0), (2, 0)]
    pred_line = [(0, 1), (2, 1)]  # shifted up
    score = metrics.anchor_score(gt_line, pred_line)
    assert 0 <= score < 1  # some penalty


def test_boolean_score_inside():
    center = [(1, 1), (2, 2)]
    left = [(0, 0), (0, 3)]
    right = [(3, 0), (3, 3)]
    assert metrics.boolean_score(center, left, right) is True


def test_boolean_score_outside():
    center = [(5, 5)]  # way outside
    left = [(0, 0), (0, 3)]
    right = [(3, 0), (3, 3)]
    assert metrics.boolean_score(center, left, right) is False
