import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Minimal NAFNet model for image denoising.
Based on "Simple Baselines for Image Restoration" (Chen et al., 2022),
keeping only the inference architecture (no local inference helpers).
"""


class LayerNorm2d(nn.Module):
    """
    LayerNorm applied channel-wise on 2D feature maps.
    Matches the behavior used in the original NAFNet blocks.
    """
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return x_norm * self.weight + self.bias


class SimpleGate(nn.Module):
    """Splits features in half along channels and multiplies them."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    """
    Core building block of NAFNet:
    depthwise convolution, simplified channel attention, and FFN-style projection.
    """
    def __init__(
        self,
        channels: int,
        dw_expand: int = 2,
        ffn_expand: int = 2,
        drop_out_rate: float = 0.0,
    ):
        super().__init__()
        dw_channels = channels * dw_expand
        ffn_channels = channels * ffn_expand

        self.norm1 = LayerNorm2d(channels)
        self.norm2 = LayerNorm2d(channels)

        self.conv1 = nn.Conv2d(channels, dw_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(
            dw_channels,
            dw_channels,
            kernel_size=3,
            padding=1,
            groups=dw_channels,
        )
        self.sg = SimpleGate()
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channels // 2, dw_channels // 2, kernel_size=1),
        )
        self.conv3 = nn.Conv2d(dw_channels // 2, channels, kernel_size=1)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0 else nn.Identity()

        self.conv4 = nn.Conv2d(channels, ffn_channels, kernel_size=1)
        self.conv5 = nn.Conv2d(ffn_channels // 2, channels, kernel_size=1)

        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0 else nn.Identity()

        # Trainable rescaling factors
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        x = self.dropout1(x)
        x = residual + x * self.beta

        y = self.conv4(self.norm2(x))
        y = self.sg(y)
        y = self.conv5(y)
        y = self.dropout2(y)

        return x + y * self.gamma


class NAFNet(nn.Module):
    """
    U-Net style encoder/decoder built from NAFBlocks with residual image prediction.

    Args:
        img_channels: number of input/output channels (RGB=3, gray=1).
        width: base channel width for the first stage.
        middle_blk_num: number of NAFBlocks in the bottleneck.
        enc_blk_nums: number of NAFBlocks in each encoder stage.
        dec_blk_nums: number of NAFBlocks in each decoder stage.
    """
    def __init__(
        self,
        img_channels: int = 1,
        width: int = 16,
        middle_blk_num: int = 1,
        enc_blk_nums=(1, 1, 1, 1),
        dec_blk_nums=(1, 1, 1, 1),
    ):
        super().__init__()
        self.intro = nn.Conv2d(img_channels, width, kernel_size=3, padding=1)
        self.ending = nn.Conv2d(width, img_channels, kernel_size=3, padding=1)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        channels = width
        for num_blocks in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[NAFBlock(channels) for _ in range(num_blocks)]))
            self.downs.append(nn.Conv2d(channels, channels * 2, kernel_size=2, stride=2))
            channels *= 2

        self.middle_blks = nn.Sequential(*[NAFBlock(channels) for _ in range(middle_blk_num)])

        for num_blocks in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(channels, channels * 2, kernel_size=1, bias=False),
                    nn.PixelShuffle(2),
                )
            )
            channels //= 2
            self.decoders.append(nn.Sequential(*[NAFBlock(channels) for _ in range(num_blocks)]))

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict clean image via residual learning (output + input).

        Args:
            x: noisy input image.
        """
        b, c, h, w = x.shape
        x_padded = self._pad_image_size(x)

        x_feat = self.intro(x_padded)
        enc_feats = []

        for encoder, down in zip(self.encoders, self.downs):
            x_feat = encoder(x_feat)
            enc_feats.append(x_feat)
            x_feat = down(x_feat)

        x_feat = self.middle_blks(x_feat)

        for decoder, up, skip in zip(self.decoders, self.ups, reversed(enc_feats)):
            x_feat = up(x_feat)
            x_feat = x_feat + skip
            x_feat = decoder(x_feat)

        out = self.ending(x_feat)
        out = out + x_padded
        return out[:, :, :h, :w]

    def _pad_image_size(self, x: torch.Tensor) -> torch.Tensor:
        """Pad to multiples of the encoder stride to avoid size mismatch."""
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        return F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
