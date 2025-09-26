import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.optim as optim
import torch.nn as nn

import config
from dataset import RunwayDataset, get_transforms
from model import RunwayDetector

def train_fn(loader, model, optimizer, loss_fns, scaler, device):
    loop = tqdm(loader, leave=True)
    mean_loss = 0.0

    seg_loss_fn, line_loss_fn = loss_fns

    model.train()
    for batch_idx, (images, seg_targets, line_targets) in enumerate(loop):
        images = images.to(device=device)
        seg_targets = seg_targets.to(device=device)
        line_targets = line_targets.to(device=device)

        with torch.amp.autocast("cuda"):
            seg_preds, line_preds = model(images)
            
            loss_seg = seg_loss_fn(seg_preds, seg_targets.float())
            loss_line = line_loss_fn(line_preds, line_targets)
            
            total_loss = loss_seg + loss_line

        optimizer.zero_grad()
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        mean_loss += total_loss.item()
        loop.set_postfix(loss=total_loss.item())

    return mean_loss / len(loader)


def main():
    print(f"Using device: {config.DEVICE}")
    
    model = RunwayDetector(
        encoder=config.PRETRAINED_ENCODER,
        encoder_weights=config.ENCODER_WEIGHTS,
    ).to(config.DEVICE)

    seg_loss_fn = nn.BCEWithLogitsLoss()
    line_loss_fn = nn.MSELoss()
    
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scaler = torch.amp.GradScaler("cuda")

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

    best_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(config.NUM_EPOCHS):
        train_loss = train_fn(train_loader, model, optimizer, (seg_loss_fn, line_loss_fn), scaler, config.DEVICE)
        print(f"Epoch {epoch+1}/{config.NUM_EPOCHS} -> Train Loss: {train_loss:.4f}")

        if train_loss < best_loss - config.EARLY_STOPPING_DELTA:
            best_loss = train_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), f"{config.MODEL_OUTPUT_DIR}/best_model.pth")
            print("-> Model Saved")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= config.EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break
        
if __name__ == "__main__":
    main()

