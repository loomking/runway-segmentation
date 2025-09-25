# src/train.py
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.optim as optim

import config
from dataset import RunwayDataset, get_transforms
from model import RunwayDetector
from loss import CombinedLoss

def train_fn(loader, model, optimizer, loss_fn, scaler, device):
    """A single training epoch."""
    loop = tqdm(loader, leave=True)
    mean_loss = 0

    model.train()
    for batch_idx, data in enumerate(loop):
        images = data["image"].to(device=device)
        seg_targets = data["mask"].to(device=device)
        line_targets = data["lines"].to(device=device)

        # Forward
        with torch.cuda.amp.autocast():
            seg_preds, line_preds = model(images)
            loss = loss_fn(seg_preds, line_preds, seg_targets, line_targets)

        # Backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Update tqdm loop
        mean_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    return mean_loss / len(loader)


def main():
    print(f"Using device: {config.DEVICE}")
    model = RunwayDetector(
        encoder=config.PRETRAINED_ENCODER,
        encoder_weights=config.ENCODER_WEIGHTS,
        num_seg_classes=config.NUM_CLASSES,
        num_line_coords=config.NUM_LINE_COORDS
    ).to(config.DEVICE)

    loss_fn = CombinedLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler()

    # Create DataLoaders
    train_transform = get_transforms(config.IMAGE_WIDTH, config.IMAGE_HEIGHT)
    train_dataset = RunwayDataset(
        image_dir=config.TRAIN_IMG_DIR,
        mask_dir=config.TRAIN_MASK_DIR,
        line_json_path=config.TRAIN_LINE_JSON,
        transform=train_transform
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        shuffle=True
    )

    best_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(config.NUM_EPOCHS):
        train_loss = train_fn(train_loader, model, optimizer, loss_fn, scaler, config.DEVICE)
        print(f"Epoch {epoch+1}/{config.NUM_EPOCHS} -> Train Loss: {train_loss:.4f}")

        # Early Stopping Check
        if train_loss < best_loss - config.EARLY_STOPPING_DELTA:
            best_loss = train_loss
            epochs_no_improve = 0
            # Save model checkpoint
            torch.save(model.state_dict(), f"{config.MODEL_OUTPUT_DIR}/best_model.pth")
            print("-> Model Saved")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= config.EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

if __name__ == "__main__":
    main()
