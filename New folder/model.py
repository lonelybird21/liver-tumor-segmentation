"""
3D U-Net for Liver Tumor Segmentation.
Input:  (B, 1, D, H, W)  - single-channel CT patch
Output: (B, 3, D, H, W)  - 3 classes: background, liver, tumor
"""

import torch
import torch.nn as nn


def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.InstanceNorm3d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.InstanceNorm3d(out_channels),
        nn.ReLU(inplace=True),
    )


class UNet3D(nn.Module):
    def __init__(self, in_channels=1, num_classes=3, base_filters=16):
        super().__init__()

        f = base_filters

        # Encoder
        self.enc1 = conv_block(in_channels, f)
        self.enc2 = conv_block(f, f * 2)
        self.enc3 = conv_block(f * 2, f * 4)
        self.enc4 = conv_block(f * 4, f * 8)

        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = conv_block(f * 8, f * 16)

        # Decoder
        self.up4 = nn.ConvTranspose3d(f * 16, f * 8, kernel_size=2, stride=2)
        self.dec4 = conv_block(f * 16, f * 8)

        self.up3 = nn.ConvTranspose3d(f * 8, f * 4, kernel_size=2, stride=2)
        self.dec3 = conv_block(f * 8, f * 4)

        self.up2 = nn.ConvTranspose3d(f * 4, f * 2, kernel_size=2, stride=2)
        self.dec2 = conv_block(f * 4, f * 2)

        self.up1 = nn.ConvTranspose3d(f * 2, f, kernel_size=2, stride=2)
        self.dec1 = conv_block(f * 2, f)

        # Output
        self.out_conv = nn.Conv3d(f, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b = self.bottleneck(self.pool(e4))

        # Decoder with skip connections
        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.out_conv(d1)


if __name__ == "__main__":
    # Quick sanity check
    model = UNet3D(in_channels=1, num_classes=3, base_filters=16)
    dummy_input = torch.randn(1, 1, 64, 128, 128)  # small patch for testing
    output = model(dummy_input)
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")  # expect (1, 3, 64, 128, 128)
