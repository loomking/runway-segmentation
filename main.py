import os
import argparse
from glob import glob
from evaluation import metrics, utils


def evaluate_all(gt_dir: str, pred_dir: str, out_path: str = None):
    """
    Run evaluation across all ground truth and prediction pairs.
    Matches files based on names (mask + lines).
    """

    results = {}

    # Find all ground truth masks
    gt_masks = sorted(glob(os.path.join(gt_dir, "*_mask.png")))

    for gt_mask_path in gt_masks:
        # Match by replacing "_mask.png" with filenames in predictions
        basename = os.path.basename(gt_mask_path).replace("_mask.png", "")

        pred_mask_path = os.path.join(pred_dir, f"{basename}_mask_pred.png")
        gt_lines_path = os.path.join(gt_dir, f"{basename}_lines.json")
        pred_lines_path = os.path.join(pred_dir, f"{basename}_lines_pred.json")

        if not (os.path.exists(pred_mask_path) and os.path.exists(gt_lines_path) and os.path.exists(pred_lines_path)):
            print(f" Skipping {basename} (missing files)")
            continue

        # Load data
        mask_gt, mask_pred, gt_lines, pred_lines = utils.compare_file_pairs(
            gt_mask_path, pred_mask_path, gt_lines_path, pred_lines_path
        )

        # Compute metrics
        scores = metrics.compute_all_metrics(mask_gt, mask_pred, gt_lines, pred_lines)
        results[basename] = scores

        print(f" {basename}: {scores}")

    # Save results if path given
    if out_path:
        utils.save_results(results, out_path)
        print(f"\n Results saved to {out_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate segmentation and line predictions.")
    parser.add_argument("--gt", required=True, help="Path to ground truth directory")
    parser.add_argument("--pred", required=True, help="Path to predictions directory")
    parser.add_argument("--out", default="evaluation/outputs/results.json", help="Path to save results JSON")

    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    evaluate_all(args.gt, args.pred, args.out)
