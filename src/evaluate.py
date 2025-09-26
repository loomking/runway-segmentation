import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader

import config
from model import RunwayDetector
from dataset import RunwayDataset, get_transforms
from metrics import iou_score, anchor_score, boolean_score

def main():
    device = torch.device(config.DEVICE)

    model = RunwayDetector(encoder=config.PRETRAINED_ENCODER, encoder_weights=None).to(device)
    model_path = os.path.join(config.MODEL_OUTPUT_DIR, "best_model.pth")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("Model loaded successfully.")

    val_transform = get_transforms(config.IMAGE_WIDTH, config.IMAGE_HEIGHT)
    val_dataset = RunwayDataset(
        image_dir=config.VAL_IMG_DIR,
        mask_dir=config.VAL_MASK_DIR,
        line_json_path=config.VAL_LINE_JSON,
        transform=val_transform
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    results_list = []

    with torch.no_grad():
        for i, (image, gt_mask, gt_norm_coords) in enumerate(tqdm(val_loader, desc="Evaluating")):
            image_name = val_dataset.image_files[i]
            image = image.to(device)

            pred_mask_logits, pred_norm_coords = model(image)

            pred_mask = (torch.sigmoid(pred_mask_logits) > config.CONFIDENCE_THRESHOLD).cpu().numpy().squeeze()
            pred_coords = pred_norm_coords.cpu().numpy().flatten()
            pred_coords[0::2] *= config.IMAGE_WIDTH
            pred_coords[1::2] *= config.IMAGE_HEIGHT

            gt_mask = gt_mask.numpy().squeeze()
            gt_coords = gt_norm_coords.numpy().flatten()
            gt_coords[0::2] *= config.IMAGE_WIDTH
            gt_coords[1::2] *= config.IMAGE_HEIGHT

            iou = iou_score(gt_mask, pred_mask)
            anchor = anchor_score(gt_coords, pred_coords)
            boolean = boolean_score(pred_coords)
            mean_score = np.mean([iou, anchor, boolean])

            results_list.append({
                'Image Name': image_name,
                'IOU score': iou,
                'Anchor Score': anchor,
                'Boolen_score': boolean,
                'Mean Score': mean_score
            })

    df = pd.DataFrame(results_list)

    mean_row = {
        'Image Name': 'Mean Score',
        'IOU score': df['IOU score'].mean(),
        'Anchor Score': df['Anchor Score'].mean(),
        'Boolen_score': df['Boolen_score'].mean(),
        'Mean Score': df['Mean Score'].mean()
    }

    mean_df = pd.DataFrame([mean_row])
    df_final = pd.concat([df, mean_df], ignore_index=True)

    output_path = os.path.join(config.RESULTS_OUTPUT_DIR, "submission.csv")
    df_final.to_csv(output_path, index=False)

    print("\n--- Evaluation Complete ---")
    print(f"Mean IoU Score:     {mean_row['IOU score']:.4f}")
    print(f"Mean Anchor Score:  {mean_row['Anchor Score']:.4f}")
    print(f"Mean Boolean Score: {mean_row['Boolen_score']:.4f}")
    print(f"\nSubmission file saved to: {output_path}")


if __name__ == "__main__":
    main()

