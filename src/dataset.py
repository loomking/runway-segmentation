import os
import json
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
from albumentations.pytorch import ToTensorV2
import albumentations as A

class RunwayDataset(Dataset):
    def __init__(self, image_dir, mask_dir, line_json_path, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg'))])

        with open(line_json_path, 'r') as f:
            self.line_data = json.load(f)

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

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if mask is not None and mask.shape[:2] != image.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

        orig_h, orig_w, _ = image.shape

        if mask is not None:
            _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
            mask[mask == 255] = 1.0

        lines = self.line_data.get(img_name, {})
        if not isinstance(lines, dict):
            lines = {}

        left_edge = lines.get('LEDG', [[0, 0], [0, 0]])
        right_edge = lines.get('REDG', [[0, 0], [0, 0]])
        center_line = lines.get('CTL', [[0, 0], [0, 0]])

        line_coords = np.array(left_edge + right_edge + center_line, dtype=np.float32).flatten()

        line_coords[0::2] /= orig_w
        line_coords[1::2] /= orig_h
        
        line_coords = np.clip(line_coords, 0.0, 1.0)
        
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        line_coords_tensor = torch.from_numpy(line_coords)

        mask = mask.unsqueeze(0)

        return image, mask, line_coords_tensor

def get_transforms(width, height):
    return A.Compose([
        A.Resize(height, width),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ])

