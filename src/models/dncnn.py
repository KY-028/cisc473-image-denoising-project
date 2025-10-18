import torch.nn as nn
"""
Simple lightweight DnCNN model for grayscale image denoising.
It learns to predict noise and removes it from the input image.
"""
class DnCnn(nn.Module):
    def __init__(self, image_channels=1, n_channels=64):
        """
        Initialize the DnCnn model.

        Args:
            image_channels: number of input/output channels. For grey = 1
            n_channels: number of neurons in each hidden layer
        """
        super().__init__()
        # Layer 1: Conv + ReLU
        # bias=True since it works on raw pixels (no BatchNorm here)
        # preventing gradient vanishing
        self.layer1 = nn.Sequential(
                nn.Conv2d(image_channels, n_channels, kernel_size=3, padding=1, bias=True),
                nn.ReLU(inplace=True)
        )
        
        # Layer 2–10: Conv + BN + ReLU
        # bias=False because BatchNorm has included
        self.layer2 = nn.Sequential(
                nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(n_channels),
                nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
                nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(n_channels),
                nn.ReLU(inplace=True)
        )
        self.layer4 = nn.Sequential(
                nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(n_channels),
                nn.ReLU(inplace=True)
        )
        self.layer5 = nn.Sequential(
                nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(n_channels),
                nn.ReLU(inplace=True)
        )
        self.layer6 = nn.Sequential(
                nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(n_channels),
                nn.ReLU(inplace=True)
        )
        self.layer7 = nn.Sequential(
                nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(n_channels),
                nn.ReLU(inplace=True)
        )
        self.layer8 = nn.Sequential(
                nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(n_channels),
                nn.ReLU(inplace=True)
        )
        self.layer9 = nn.Sequential(
                nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(n_channels),
                nn.ReLU(inplace=True)
        )
        self.layer10 = nn.Sequential(
                nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(n_channels),
                nn.ReLU(inplace=True)
        )
        # Combine 64 feature maps into one predicted noise map (1 channel)
        self.output = nn.Conv2d(n_channels, image_channels, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        """
        Forward pass of the DnCnn model.
        
        Args:
            x: noisy input image tensor
        Returns:
            denoised image tensor
        """
        # extract noise features and perdict noise image
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)
        out = self.layer10(out)
        noise = self.output(out)
        # input image - noise image = clean image
        return x - noise