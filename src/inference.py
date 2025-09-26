import torch
import matplotlib.pyplot as plt
import cv2
import numpy as np
from torch.utils.data import DataLoader

import config
from dataset import RunwayDataset, get_transforms
from model import RunwayDetector


def denormalize_image(tensor):
    """Undo normalization to get a proper [0,1] RGB image."""
    tensor = tensor.clone()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    tensor = tensor.permute(1, 2, 0).cpu().numpy()
    tensor = std * tensor + mean
    tensor = np.clip(tensor, 0, 1)
    return tensor


def draw_lines_on_image(image, coords, color=(0, 255, 0), thickness=2):
    """
    Draw line segments from predicted or GT coordinates.
    image: must be uint8 BGR (for cv2).
    coords: flat array of coordinates [x1,y1, x2,y2, ...].
    """
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)

    if len(image.shape) == 2:  # grayscale → convert to 3-channel
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    coords = coords.reshape(-1, 2, 2)
    for line in coords:
        pt1 = (int(line[0, 0]), int(line[0, 1]))
        pt2 = (int(line[1, 0]), int(line[1, 1]))
        cv2.line(image, pt1, pt2, color, thickness)
    return image


def visualize_prediction(model, loader, device, num_samples=5):
    model.eval()
    shown = 0
    with torch.no_grad():
        for i, (image_tensor, gt_mask_tensor, gt_norm_coords_tensor) in enumerate(loader):
            if shown >= num_samples:
                break

            image_tensor = image_tensor.to(device)

            # forward pass
            pred_mask_logits, pred_norm_coords = model(image_tensor)

            # segmentation mask → uint8 grayscale
            pred_mask = (torch.sigmoid(pred_mask_logits) > config.CONFIDENCE_THRESHOLD).cpu().squeeze().numpy().astype(np.uint8)
            gt_mask = gt_mask_tensor.cpu().squeeze().numpy().astype(np.uint8)

            # coordinates
            pred_coords = pred_norm_coords.cpu().numpy().flatten()
            gt_coords = gt_norm_coords_tensor.cpu().numpy().flatten()

            pred_coords[0::2] *= config.IMAGE_WIDTH
            pred_coords[1::2] *= config.IMAGE_HEIGHT
            gt_coords[0::2] *= config.IMAGE_WIDTH
            gt_coords[1::2] *= config.IMAGE_HEIGHT

            # input image
            img_np = denormalize_image(image_tensor.squeeze())

            # build display images
            gt_display = cv2.cvtColor(gt_mask * 255, cv2.COLOR_GRAY2BGR)
            gt_display = draw_lines_on_image(gt_display, gt_coords, color=(0, 255, 0))

            pred_display = cv2.cvtColor(pred_mask * 255, cv2.COLOR_GRAY2BGR)
            pred_display = draw_lines_on_image(pred_display, pred_coords, color=(255, 0, 0))

            # matplotlib expects RGB, convert from BGR
            gt_display = cv2.cvtColor(gt_display, cv2.COLOR_BGR2RGB)
            pred_display = cv2.cvtColor(pred_display, cv2.COLOR_BGR2RGB)

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(img_np)
            axes[0].set_title("Input Image")
            axes[1].imshow(gt_display)
            axes[1].set_title("Ground Truth")
            axes[2].imshow(pred_display)
            axes[2].set_title("Prediction")

            for ax in axes:
                ax.axis("off")
            plt.show()

            shown += 1


if __name__ == "__main__":
    print(f"Using device: {config.DEVICE}")

    model = RunwayDetector(
        encoder=config.PRETRAINED_ENCODER,
        encoder_weights=None
    ).to(config.DEVICE)
    model.load_state_dict(torch.load(f"{config.MODEL_OUTPUT_DIR}/best_model.pth", map_location=config.DEVICE))

    test_transform = get_transforms(config.IMAGE_WIDTH, config.IMAGE_HEIGHT)
    test_dataset = RunwayDataset(
        image_dir=config.TEST_IMG_DIR,
        mask_dir=config.TEST_MASK_DIR,
        line_json_path=config.TEST_LINE_JSON,
        transform=test_transform
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    visualize_prediction(model, test_loader, config.DEVICE, num_samples=5)
