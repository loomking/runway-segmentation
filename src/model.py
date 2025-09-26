import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

class RunwayDetector(nn.Module):
    def __init__(self, encoder='resnet34', encoder_weights='imagenet'):
        super(RunwayDetector, self).__init__()

        smp_model = smp.Unet(
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=1,
        )
        
        self.encoder = smp_model.encoder
        self.decoder = smp_model.decoder
        self.segmentation_head = smp_model.segmentation_head

        encoder_out_channels = self.encoder.out_channels[-1]

        self.regression_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(encoder_out_channels, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 12),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.encoder(x)
        
        bottleneck = features[-1]
        
        # --- THE FIX ---
        # Pass the 'features' list as a single argument, without the '*'
        decoder_output = self.decoder(features)
        
        seg_mask = self.segmentation_head(decoder_output)
        line_coords = self.regression_head(bottleneck)

        return seg_mask, line_coords

