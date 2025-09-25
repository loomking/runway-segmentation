# src/predict.py
import torch
import os
import cv2
import pandas as pd
from tqdm import tqdm
import numpy as np

import config
from model import RunwayDetector
from dataset import get_transforms
from evaluate import calculate_iou, calculate_anchor_score, calculate_boolean_score

def predict_and_evaluate():
    print("Loading model for prediction...")
    model = RunwayDetector(
        encoder=config.PRETRAINED_ENCODER,
        encoder_weights=None, # Weights are loaded from the checkpoint
        num_seg_classes=config.NUM_CLASSES,
        num_line_coords=config.NUM_LINE_COORDS
    ).to(config.DEVICE)
    model.load_state_dict(torch.load(f"{config.MODEL_OUTPUT_DIR}/best_model.pth"))
    model.eval()

    test_images = sorted(os.listdir(config.TEST_IMG_DIR))
    transform = get_transforms(config.IMAGE_WIDTH, config.IMAGE_HEIGHT)
    results = []

    print("Running predictions on the test set...")
    for img_name in tqdm(test_images):
        img_path = os.path.join(config.TEST_IMG_DIR, img_name)
        image = cv2.imread(img_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Preprocess image
        augmented = transform(image=image_rgb)
        input_tensor = augmented['image'].unsqueeze(0).to(config.DEVICE)

        with torch.no_grad():
            seg_pred, line_pred = model(input_tensor)

            # --- Process Segmentation Output ---
            seg_pred = torch.softmax(seg_pred, dim=1)
            seg_pred = (seg_pred > config.CONFIDENCE_THRESHOLD).float()
            # Get the runway class (channel 1) and resize to original image size
            pred_mask = seg_pred[0, 1, :, :].cpu().numpy()
            pred_mask = cv2.resize(pred_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

            # --- Process Line Output ---
            pred_coords = line_pred.cpu().numpy().flatten()

            # --- Load Ground Truth for Evaluation ---
            gt_mask_path = os.path.join(config.TEST_MASK_DIR, img_name)
            gt_mask = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)
            _, gt_mask = cv2.threshold(gt_mask, 1, 1, cv2.THRESH_BINARY) # Binarize

            # --- Calculate Scores ---
            iou_score = calculate_iou(pred_mask, gt_mask)
            # Note: For Anchor and Boolean scores, you'll need to parse ground truth lines
            # This is a placeholder for the logic.
            # You would load test_labels_640x360.json here.
            anchor_score = 0.0 # Placeholder
            boolean_score = 0 # Placeholder

            results.append({
                "image": img_name,
                "iou_score": iou_score,
                "anchor_score": anchor_score,
                "boolean_score": boolean_score
            })

    # Create and save submission CSV
    df = pd.DataFrame(results)
    mean_scores = df[["iou_score", "anchor_score", "boolean_score"]].mean()
    mean_row = pd.DataFrame([{"image": "mean", **mean_scores}], columns=df.columns)

    submission_df = pd.concat([df, mean_row], ignore_index=True)
    submission_df.to_csv(f"{config.RESULTS_OUTPUT_DIR}/submission.csv", index=False)
    print(f"Submission file saved to {config.RESULTS_OUTPUT_DIR}/submission.csv")
    print("Mean Scores:")
    print(mean_scores)


if __name__ == "__main__":
    predict_and_evaluate()
