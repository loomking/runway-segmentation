import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

import config
from dataset import RunwayDataset, get_transforms
from model import RunwayDetector

# 1. Load trained model
def load_model(checkpoint_path, device):
    model = RunwayDetector(
        encoder=config.PRETRAINED_ENCODER,
        encoder_weights=config.ENCODER_WEIGHTS,
        num_seg_classes=config.NUM_CLASSES,
        num_line_coords=config.NUM_LINE_COORDS
    ).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model

# 2. Visualize prediction vs ground truth
def visualize_prediction(model, loader, device, num_samples=10):
    model.eval()
    shown = 0
    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device)
            gt_mask = batch["mask"].squeeze().cpu().numpy()

            # Forward pass
            seg_pred, _ = model(image)
            seg_pred = torch.argmax(seg_pred, dim=1).squeeze().cpu().numpy()

            # Convert input tensor to numpy for plotting
            img_np = image.squeeze().permute(1, 2, 0).cpu().numpy()

            # Show input, GT, and prediction
            fig, axes = plt.subplots(1, 3, figsize=(10, 5))
            axes[0].imshow(img_np)
            axes[0].set_title("Input Image")
            axes[1].imshow(gt_mask, cmap="gray")
            axes[1].set_title("Ground Truth")
            axes[2].imshow(seg_pred, cmap="gray")
            axes[2].set_title("Prediction")
            for ax in axes:
                ax.axis("off")
            plt.show()

            shown += 1
            if shown >= num_samples:
                break

if __name__ == "__main__":
    print(f"Using device: {config.DEVICE}")

    # Load best trained model
    model = load_model(f"{config.MODEL_OUTPUT_DIR}/best_model.pth", config.DEVICE)

    # Load test dataset
    test_transform = get_transforms(config.IMAGE_WIDTH, config.IMAGE_HEIGHT)
    test_dataset = RunwayDataset(
        image_dir=config.TEST_IMG_DIR,
        mask_dir=config.TEST_MASK_DIR,
        line_json_path=config.TEST_LINE_JSON,
        transform=test_transform
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    # Show predictions vs ground truth
    visualize_prediction(model, test_loader, config.DEVICE, num_samples=10)
