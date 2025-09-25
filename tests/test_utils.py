import os
import json
import numpy as np
from PIL import Image
import tempfile

from evaluation import utils


def test_load_mask_binary(tmp_path):
    # Create a simple black/white mask
    mask_array = np.array([[0, 255], [128, 200]], dtype=np.uint8)
    mask_img = Image.fromarray(mask_array)
    mask_path = tmp_path / "mask.png"
    mask_img.save(mask_path)

    mask = utils.load_mask(str(mask_path))
    # Expected: [[0,1],[1,1]] since threshold=128
    expected = np.array([[0, 1], [1, 1]], dtype=np.uint8)
    assert np.array_equal(mask, expected)


def test_load_lines(tmp_path):
    # Create a dummy JSON file
    data = {
        "center": [[0, 0], [1, 1]],
        "left": [[0, 0], [0, 2]],
        "right": [[2, 0], [2, 2]]
    }
    json_path = tmp_path / "lines.json"
    with open(json_path, "w") as f:
        json.dump(data, f)

    lines = utils.load_lines(str(json_path))
    assert "center" in lines and "left" in lines and "right" in lines
    assert lines["center"] == [[0, 0], [1, 1]]


def test_load_lines_missing_keys(tmp_path):
    # Create a JSON missing "right"
    data = {"center": [[0, 0]], "left": [[1, 1]]}
    json_path = tmp_path / "lines.json"
    with open(json_path, "w") as f:
        json.dump(data, f)

    lines = utils.load_lines(str(json_path))
    # Should fill missing "right" with empty list
    assert "right" in lines
    assert lines["right"] == []


def test_save_results(tmp_path):
    results = {"IoU": 0.9, "Dice": 0.95}
    out_path = tmp_path / "results.json"
    utils.save_results(results, str(out_path))

    with open(out_path, "r") as f:
        saved = json.load(f)

    assert saved == results


def test_compare_file_pairs(tmp_path):
    # Create dummy mask images
    mask_array = np.array([[0, 255]], dtype=np.uint8)
    gt_mask_path = tmp_path / "gt_mask.png"
    pred_mask_path = tmp_path / "pred_mask.png"
    Image.fromarray(mask_array).save(gt_mask_path)
    Image.fromarray(mask_array).save(pred_mask_path)

    # Create dummy line JSONs
    line_data = {
        "center": [[0, 0], [1, 1]],
        "left": [[0, 0], [0, 1]],
        "right": [[1, 0], [1, 1]]
    }
    gt_lines_path = tmp_path / "gt_lines.json"
    pred_lines_path = tmp_path / "pred_lines.json"
    with open(gt_lines_path, "w") as f:
        json.dump(line_data, f)
    with open(pred_lines_path, "w") as f:
        json.dump(line_data, f)

    mask_gt, mask_pred, gt_lines, pred_lines = utils.compare_file_pairs(
        str(gt_mask_path), str(pred_mask_path),
        str(gt_lines_path), str(pred_lines_path)
    )

    assert mask_gt.shape == mask_pred.shape
    assert "center" in gt_lines and "center" in pred_lines
