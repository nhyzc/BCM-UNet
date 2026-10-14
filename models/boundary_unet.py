"""BCM-UNet: a boundary-guided multi-scale U-Net for crack segmentation.

Three modules are stacked on top of a plain U-Net and can each be enabled
independently, which is what the ablation study in config.py switches between:

  * boundary module   - a high-resolution edge map is predicted from the first
    encoder stage and used to gate every decoder stage. The edge prediction and
    the joint edge + segmentation loss are two halves of the same module: the
    attention is only meaningful when the edge branch is supervised, so
    use_boundary controls the architecture and the loss together.
  * continuity module - a dilated multi-receptive-field block at the bottleneck
    that helps reconnect cracks broken by noise;
  * multi-scale fusion - decoder features from several resolutions are resampled
    to a common size and fused before the final 1x1 projection.

With all three disabled the model reduces to a plain U-Net decoder.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .parts import DoubleConv, Down, Up, OutConv


class BoundaryExtractor(nn.Module):
    """Predict a single-channel edge map from a high-resolution feature map."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, 1, kernel_size=1)
        )

    def forward(self, x):
        return self.conv(x)


class BoundaryAttention(nn.Module):
    """Re-weight decoder features with the predicted edge map.

    The edge map is resampled to the decoder resolution and mapped to a
    per-channel gate.
    """

    def __init__(self, in_channels):
        super().__init__()
        self.edge_conv = nn.Sequential(
            nn.Conv2d(1, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x, edge_map):
        edge_down = F.interpolate(edge_map, size=x.shape[2:], mode='bilinear',
                                  align_corners=True)
        attention_weight = self.edge_conv(edge_down)
        return x + x * attention_weight


class ContinuityModule(nn.Module):
    """Dilated multi-branch block that widens the receptive field.

    Four parallel 3x3 branches with dilation rates 1, 2, 4 and 8 capture
    context at several scales, which helps bridge gaps in thin cracks without
    losing the fine detail needed for their width.
    """

    def __init__(self, in_channels):
        super().__init__()
        mid_channels = in_channels // 4
        self.branch1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3,
                                 padding=1, dilation=1)
        self.branch2 = nn.Conv2d(in_channels, mid_channels, kernel_size=3,
                                 padding=2, dilation=2)
        self.branch3 = nn.Conv2d(in_channels, mid_channels, kernel_size=3,
                                 padding=4, dilation=4)
        self.branch4 = nn.Conv2d(in_channels, mid_channels, kernel_size=3,
                                 padding=8, dilation=8)
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        out = torch.cat([b1, b2, b3, b4], dim=1)
        return x + self.fusion(out)


class BoundaryUNet(nn.Module):
    """U-Net with an optional boundary module, continuity module and fusion.

    Args:
        use_boundary: build the edge extractor and the four boundary-attention
            blocks. When True, forward() returns ``(logits, edge_map)`` and the
            model is trained with JointEdgeSegLoss; when False it returns the
            logits alone and is trained with BCEDiceLoss.
        use_continuity: build the dilated continuity block at the bottleneck.
        use_multiscale: fuse the decoder stages before the final projection.
    """

    def __init__(self, in_channels=1, out_channels=1, bilinear=False,
                 use_boundary=False, use_continuity=False, use_multiscale=False):
        super().__init__()
        self.n_channels = in_channels
        self.n_classes = out_channels
        self.bilinear = bilinear

        self.use_boundary = use_boundary
        self.use_continuity = use_continuity
        self.use_multiscale = use_multiscale
        factor = 2 if bilinear else 1

        # --- U-Net encoder ---
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024 // factor)

        # --- Module 2: continuity enhancement, applied at the bottleneck ---
        if self.use_continuity:
            self.continuity = ContinuityModule(1024 // factor)

        # --- Module 1: boundary extraction and attention ---
        # Built only when the module is enabled.
        if self.use_boundary:
            self.boundary_extractor = BoundaryExtractor(64, 32)
            self.bound_att1 = BoundaryAttention(512 // factor)
            self.bound_att2 = BoundaryAttention(256 // factor)
            self.bound_att3 = BoundaryAttention(128 // factor)
            self.bound_att4 = BoundaryAttention(64)

        # --- U-Net decoder ---
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)

        # --- Module 3: multi-scale fusion of the decoder stages ---
        if self.use_multiscale:
            cat_channels = (512 // factor) + (256 // factor) + (128 // factor) + 64
            self.ms_fusion = nn.Sequential(
                nn.Conv2d(cat_channels, 64, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            )

        self.outc = OutConv(64, out_channels)

    def forward(self, x):
        # Encoding path.
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Continuity enhancement at the bottleneck.
        if self.use_continuity:
            x5 = self.continuity(x5)

        # Boundary guidance: predict the edge map once and reuse it to gate
        # every decoder stage.
        edge_map = self.boundary_extractor(x1) if self.use_boundary else None

        # Decoding path, with boundary attention when the module is enabled.
        out1 = self.up1(x5, x4)
        if self.use_boundary:
            out1 = self.bound_att1(out1, edge_map)

        out2 = self.up2(out1, x3)
        if self.use_boundary:
            out2 = self.bound_att2(out2, edge_map)

        out3 = self.up3(out2, x2)
        if self.use_boundary:
            out3 = self.bound_att3(out3, edge_map)

        out4 = self.up4(out3, x1)
        if self.use_boundary:
            out4 = self.bound_att4(out4, edge_map)

        # Fuse the decoder stages, or use the finest one directly.
        if self.use_multiscale:
            target_size = out4.shape[2:]
            up_out1 = F.interpolate(out1, size=target_size, mode='bilinear',
                                    align_corners=True)
            up_out2 = F.interpolate(out2, size=target_size, mode='bilinear',
                                    align_corners=True)
            up_out3 = F.interpolate(out3, size=target_size, mode='bilinear',
                                    align_corners=True)

            ms_features = torch.cat([up_out1, up_out2, up_out3, out4], dim=1)
            out = self.ms_fusion(ms_features)
            logits = self.outc(out)
        else:
            logits = self.outc(out4)

        # The edge map is returned only when the boundary module is enabled.
        if self.use_boundary:
            return logits, edge_map
        return logits
