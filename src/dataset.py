import os
import json
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
from albumentations.pytorch import ToTensorV2
import albumentations as A

class RunwayDataset(Dataset):
    """
    PyTorch Dataset for loading FS2020 runway images, segmentation masks, and line coordinates.
    """
    def __init__(self, image_dir, mask_dir, line_json_path, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg'))])

        # Load and parse line data from JSON
        with open(line_json_path, 'r') as f:
            self.line_data = json.load(f)

        # If JSON is a list of dicts, flatten into one dict keyed by filename
        if isinstance(self.line_data, list):
            merged = {}
            for entry in self.line_data:
                if isinstance(entry, dict):
                    merged.update(entry)
            self.line_data = merged

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)

        # Load image and mask
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # --- Fix shape mismatch: resize mask to image size ---
        if mask is not None and mask.shape[:2] != image.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Binarize the mask: runway pixels are non-zero, background is zero
        _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
        mask[mask == 255] = 1  # Convert to 0 for background, 1 for runway

        # Get line coordinates safely
        lines = self.line_data.get(img_name, {})
        if not isinstance(lines, dict):
            lines = {}

        ledg = lines.get('LEDG', [[0, 0], [0, 0]])
        redg = lines.get('REDG', [[0, 0], [0, 0]])
        ctl = lines.get('CTL', [[0, 0], [0, 0]])

        # Flatten coordinates into a single tensor
        line_coords = np.array(ledg + redg + ctl, dtype=np.float32).flatten()

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        return {
            "image": image,
            "mask": mask.long(),  # Use long for CrossEntropyLoss
            "lines": torch.from_numpy(line_coords)
        }

def get_transforms(width, height):
    """Returns a set of Albumentations transformations for training."""
    return A.Compose([
        A.Resize(height, width),  # this will resize BOTH image and mask consistently
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ])
