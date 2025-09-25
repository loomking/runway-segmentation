# src/model.py
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

class RunwayDetector(nn.Module):
    """
    A U-Net based model with a pretrained encoder for simultaneous runway
    segmentation and line coordinate prediction.
    """
    def __init__(self, encoder='resnet34', encoder_weights='imagenet', num_seg_classes=2, num_line_coords=12):
        super(RunwayDetector, self).__init__()

        # --- Segmentation Head ---
        self.seg_model = smp.Unet(
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=num_seg_classes,
        )

        # --- Line Prediction (Regression) Head ---
        encoder_out_channels = self.seg_model.encoder.out_channels[-1]

        self.regression_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Global Average Pooling
            nn.Flatten(),
            nn.Linear(encoder_out_channels, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_line_coords)
        )

    def forward(self, x):
        # Segmentation output directly from SMP Unet
        seg_mask = self.seg_model(x)  # (B, num_seg_classes, H, W)

        # Extract encoder features for line regression
        features = self.seg_model.encoder(x)
        bottleneck = features[-1]  # deepest feature map

        line_coords = self.regression_head(bottleneck)

        return seg_mask, line_coords
