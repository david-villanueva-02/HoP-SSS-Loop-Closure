from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pytorch_msssim
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
# from torch.amp import GradScaler, autocast # Version compatibility
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup


@dataclass
class TrainConfig:
    """Configuration container for training."""
    # Training parameters
    num_epochs: int = 200
    batch_size: int = 10
    num_workers: int = 2
    lr: float = 1e-3
    betas: Tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 5e-4
    image_size: Tuple[int, int] = (512, 512)

    # Pre-trained weight path that need to be loaded
    # checkpoint_path: str =  r'C:\LastPC\Research\Research\dataset\threeD\optical_sonar\weight\best_model_v610_jaguar.pth'
    checkpoint_path: str =  r'model/best_model_v610_jaguar.pth'

    # Weight path saved after training
    save_path: str = r'model/final.pth'

    # Various training information log files
    other_loss_log_file: str = r"model/logs/train_other_loss_v606.txt"
    train_loss_file: str = r"model/logs/train_loss_v606.txt"
    train_z_loss_file: str = r"model/logs/train_z_loss_v606.txt"
    train_shadow_loss_file: str = r"model/logs/train_shadow_loss_v606.txt"
    train_sss_loss_file: str = r"model/logs/train_sss_loss_v606.txt"
    val_loss_file: str = r"model/logs/val_loss_v606.txt"

    # Training set data path
    train_image_path: str = r"output/train/images"
    train_range_path: str = r"output/train/range"
    train_altitude_path: str = r"output/train/altitude"
    train_shadow_path: str = r"output/train/shadow"

    # Validation set data path
    val_image_path: str = r"output/val/images"
    val_range_path: str = r"output/val/range"
    val_altitude_path: str = r"output/val/altitude"
    val_shadow_path: str = r"output/val/shadow"

# =============================================================================
# Model building blocks
# =============================================================================


class DoubleConv(nn.Module):
    """Apply two consecutive Conv-BN-ReLU blocks."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    """Downsample with max pooling followed by a double convolution block."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.down = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(x)


class Up(nn.Module):
    """Upsample, align feature maps, concatenate skip features, and refine."""

    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True) -> None:
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)

        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)
        x1 = F.pad(
            x1,
            [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2],
        )

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Project features to the final output channels."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class AttentionBlock(nn.Module):
    """Lightweight spatial attention block."""

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn = self.attention(x)
        return x * attn


class TimeEmbedding(nn.Module):
    """Sinusoidal time embedding module kept for future extension."""

    def __init__(self, channels: int, t_min: float = 0.0, t_max: float = 1000.0) -> None:
        super().__init__()
        if channels % 2 != 0:
            raise ValueError("channels must be even.")

        self.channels = channels
        self.t_min = t_min
        self.t_max = t_max

        i = torch.arange(0, channels // 2, dtype=torch.float32)
        freq = 1.0 / (10000 ** (i / channels))
        self.register_buffer("freq_buffer", freq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, _, _ = x.shape
        t = torch.rand(batch_size, 1, device=x.device) * (self.t_max - self.t_min) + self.t_min
        angles = t * self.freq_buffer.to(x.device)
        time_embed = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        time_embed = time_embed.view(batch_size, channels, 1, 1)
        return x + time_embed


class UNet(nn.Module):
    """Attention-augmented U-Net for single-channel sonar inputs."""

    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        bilinear: bool = True,
        final_activation: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.final_activation = final_activation

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.attn2 = AttentionBlock(256)
        self.down3 = Down(256, 512)
        self.attn3 = AttentionBlock(512)

        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.attn2(self.down2(x2))
        x4 = self.attn3(self.down3(x3))
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)

        if self.final_activation is not None:
            logits = self.final_activation(logits)
        return logits


class PhysDNet(nn.Module):
    """Three-branch physical decomposition network.

    The three branches predict:
    1. z   : seafloor height
    2. rho : bottom reflectivity
    3. path: propagation loss
    """

    def __init__(self) -> None:
        super().__init__()
        self.model1 = UNet(n_channels=1, n_classes=1, final_activation=nn.Softplus())
        self.model2 = UNet(n_channels=1, n_classes=1, final_activation=nn.Softplus())
        self.model3 = UNet(n_channels=1, n_classes=1, final_activation=nn.Softplus())

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        x3: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        y1 = self.model1(x1)
        y2 = self.model2(x2)
        y3_raw = self.model3(x3)

        # Force the third branch to be row-wise consistent.
        y3_row_vector = torch.mean(y3_raw, dim=2, keepdim=True)
        y3 = y3_row_vector.repeat(1, 1, y3_raw.shape[2], 1)
        return y1, y2, y3


# =============================================================================
# Geometry utilities
# =============================================================================


def compute_real_dis(a: torch.Tensor, range_vector: torch.Tensor) -> torch.Tensor:
    """Compute slant range for each pixel column."""
    m, n = a.shape
    if range_vector.shape[0] != m:
        raise ValueError("range_vector length must match the number of rows in the image.")

    col_indices = torch.arange(n, device=a.device, dtype=a.dtype)
    return (range_vector[:, None].to(device=a.device, dtype=a.dtype) * col_indices[None, :]) / 512.0



def compute_real_dis_horizontal(slant: torch.Tensor, altitude: torch.Tensor) -> torch.Tensor:
    """Convert slant range to horizontal range."""
    if slant.ndim != 2 or altitude.ndim != 1:
        raise ValueError("slant must be 2D and altitude must be 1D.")
    if slant.shape[0] != altitude.shape[0]:
        raise ValueError("altitude length must match the number of rows in slant.")

    altitude = altitude.to(device=slant.device, dtype=slant.dtype)
    altitude_squared = altitude.unsqueeze(1) ** 2
    return torch.sqrt((slant ** 2 - altitude_squared).clamp(min=0))



def compute_theta(c: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """Legacy theta computation function kept for compatibility."""
    if a.shape != c.shape:
        raise ValueError("a and c must have the same shape.")

    m, n = a.shape
    out = torch.full((m, n), 0, dtype=torch.float64, device=a.device)

    a_inner = a[1:, 1:]
    c_inner = c[1:, 1:]
    c_up = c[:-1, 1:]
    c_left = c[1:, :-1]

    numerator = a_inner * (c_left - c_inner) - c_inner
    denominator = torch.sqrt((c_up - c_inner) ** 2 + (c_left - c_inner) ** 2 + 1) * torch.sqrt(
        a_inner**2 + c_inner**2
    )
    out[1:, 1:] = numerator / denominator
    return out



def compute_theta_new(z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Compute local incidence cosine from height and horizontal distance."""
    if x.shape != z.shape:
        raise ValueError("x and z must have the same shape.")

    m, n = x.shape
    out = torch.full((m, n), 0, dtype=z.dtype, device=x.device)

    x_inner = x[1:, 1:]
    z_inner = z[1:, 1:]
    z_up = z[:-1, 1:]
    z_left = z[1:, :-1]
    x_left = x[1:, :-1]

    numerator = x_inner * z_left - x_left * z_inner
    denominator = torch.sqrt((x_left - x_inner) ** 2 + (z_left - z_inner) ** 2 + (z_up - z_inner) ** 2 + 1e-8) * torch.sqrt(
        x_inner**2 + z_inner**2
    )
    inner = torch.where(x_inner == 0, torch.full_like(numerator, -1), numerator / denominator)
    out[1:, 1:] = inner
    return out



def compute_theta_90(z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Alternative theta computation variant used for 90-degree geometry."""
    if x.shape != z.shape:
        raise ValueError("x and z must have the same shape.")

    m, n = x.shape
    row_values = torch.arange(512, device=x.device, dtype=z.dtype) * 0.2
    y_grid = row_values.unsqueeze(1).repeat(1, 512)
    out = torch.full((m, n), 0, dtype=z.dtype, device=x.device)

    x_inner = x[1:, 1:]
    y_inner = y_grid[1:, 1:]
    z_inner = z[1:, 1:]
    z_up = z[:-1, 1:]
    z_left = z[1:, :-1]
    y_up = y_grid[:-1, 1:]

    numerator = y_inner * z_up - y_up * z_inner
    denominator = torch.sqrt((y_up - y_inner) ** 2 + (z_left - z_inner) ** 2 + (z_up - z_inner) ** 2 + 1e-8) * torch.sqrt(
        y_inner**2 + z_inner**2
    )
    inner = torch.where(x_inner == 0, torch.full_like(numerator, -1), numerator / denominator)
    out[1:, 1:] = inner
    return out


# =============================================================================
# Loss utilities
# =============================================================================


def laplacian_loss(z: torch.Tensor) -> torch.Tensor:
    """Encourage spatial smoothness in both directions."""
    dx = torch.abs(z[1:, :] - z[:-1, :])
    dy = torch.abs(z[:, 1:] - z[:, :-1])
    return dx.mean() + dy.mean()



def columnwise_laplacian_loss_mean(z: torch.Tensor) -> torch.Tensor:
    """Compute column-wise average absolute row difference."""
    dx = torch.abs(z[1:, :] - z[:-1, :])
    return dx.mean(dim=0).mean()



def rowwise_laplacian_loss_mean(z: torch.Tensor) -> torch.Tensor:
    """Compute row-wise average absolute column difference."""
    dx = torch.abs(z[:, 1:] - z[:, :-1])
    return dx.mean(dim=0).mean()



def z_score_normalize(img: torch.Tensor) -> torch.Tensor:
    """Apply z-score normalization."""
    return (img - img.mean()) / (img.std() + 1e-8)



def column_variance_loss(output: torch.Tensor) -> torch.Tensor:
    """Penalize large variance along each column."""
    return torch.var(output, dim=0, unbiased=False).mean()



def z_positive_loss(output: torch.Tensor) -> torch.Tensor:
    """Penalize negative height predictions."""
    return torch.mean(torch.maximum(torch.zeros_like(output), -output))



def min_max_normalize(x: torch.Tensor) -> torch.Tensor:
    """Normalize a tensor to the range [0, 1]."""
    return (x - x.min()) / (x.max() - x.min() + 1e-8)



def path_loss(output: torch.Tensor) -> torch.Tensor:
    """Encourage path predictions to decrease monotonically."""

    def strict_monotonic_decreasing_loss(pred: torch.Tensor, epsilon: float = 1e-3) -> torch.Tensor:
        diff = pred[:, 1:] - pred[:, :-1]
        violation = F.relu(diff + epsilon)
        return (violation**2).mean()

    return strict_monotonic_decreasing_loss(output)



def preprocess_single_image(img_tensor: torch.Tensor, device: Optional[torch.device] = None) -> torch.Tensor:
    """Convert a grayscale tensor to a 3-channel 224x224 tensor for feature extractors."""
    if img_tensor.dim() == 2:
        img_tensor = img_tensor.unsqueeze(0)
    if img_tensor.size(0) == 1:
        img_tensor = img_tensor.unsqueeze(0)
    img_tensor = img_tensor.repeat(1, 3, 1, 1)
    img_tensor = F.interpolate(img_tensor, size=(224, 224), mode="bilinear", align_corners=False)
    if device is not None:
        img_tensor = img_tensor.to(device)
    return img_tensor



def prepare_input(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Convert a grayscale image to [B, 1, H, W] float tensor on the target device."""
    if tensor.dim() == 2:
        tensor = tensor.unsqueeze(0).unsqueeze(0)
    elif tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
    return tensor.float().to(device)


class PerceptualLossMobileNetV3Large(nn.Module):
    """Perceptual loss based on MobileNetV3-Large feature maps."""

    def __init__(self, alpha: float = 1.0, use_gpu: bool = True) -> None:
        super().__init__()
        self.alpha = alpha

        weights = models.MobileNet_V3_Large_Weights.DEFAULT
        mobilenet_v3 = models.mobilenet_v3_large(weights=weights)
        self.features = nn.Sequential(*list(mobilenet_v3.features.children())[:10])

        for param in self.features.parameters():
            param.requires_grad = False

        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        self.features = self.features.to(self.device)

        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_rgb = pred.repeat(1, 3, 1, 1)
        target_rgb = target.repeat(1, 3, 1, 1)

        pred_norm = (pred_rgb - self.mean) / self.std
        target_norm = (target_rgb - self.mean) / self.std

        feat_pred = self.features(pred_norm)
        feat_target = self.features(target_norm)
        return self.alpha * F.mse_loss(feat_pred, feat_target)



def compute_shadow(hori: torch.Tensor, z_b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute a binary shadow mask using cumulative grazing angle comparison."""
    tan_phi = hori / z_b
    phi_rad = torch.atan(tan_phi)
    phi_deg = torch.rad2deg(phi_rad)
    cummax, _ = torch.cummax(phi_deg, dim=1)
    mask = (phi_deg < cummax) | (tan_phi <= 0)
    mask_primary = phi_deg < cummax
    return mask.to(torch.uint8), phi_rad, mask_primary.to(torch.uint8)



def compute_shadow_soft(
    hori: torch.Tensor,
    z_b: torch.Tensor,
    theta: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute a hard shadow mask while retaining the angle map for analysis."""
    delta = 0
    tan_phi = hori / z_b
    phi_rad = torch.atan(tan_phi)
    phi_deg = torch.rad2deg(phi_rad)
    cummax, _ = torch.cummax(phi_deg, dim=1)
    mask = (phi_deg < cummax - delta) | (tan_phi <= 0)
    mask_theta = theta < 0
    return mask.to(torch.uint8), phi_deg, mask_theta.to(torch.uint8)



def compute_shadow_new(z_b: torch.Tensor) -> torch.Tensor:
    """Generate a high-value region mask using mean-plus-std thresholding."""
    threshold = torch.mean(z_b) + torch.std(z_b)
    return (z_b > threshold).to(torch.uint8)



def compute_shadow_loss_amp(
    z_b: torch.Tensor,
    hori: torch.Tensor,
    gt_mask: torch.Tensor,
    valid_mask: torch.Tensor,
    delta: float = 0.1,
    temperature: float = 0.05,
    alpha: float = 0.25,
    gamma: float = 2.0,
    weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Mixed shadow loss combining BCE, Dice, and Focal terms."""
    del weight

    tan_phi = hori / z_b.clamp(min=1e-3)
    phi_rad = torch.atan(tan_phi)
    phi_deg = torch.rad2deg(phi_rad)
    cummax, _ = torch.cummax(phi_deg, dim=1)
    shadow_logits = (cummax - phi_deg - delta) / temperature

    pred_prob = torch.sigmoid(shadow_logits)
    gt_mask = gt_mask.float()

    mask = valid_mask.bool()
    shadow_logits_masked = shadow_logits[mask]
    gt_mask_masked = gt_mask[mask]
    pred_prob_masked = pred_prob[mask]

    smooth = 1e-6
    intersection = (pred_prob_masked * gt_mask_masked).sum()
    dice_loss = 1 - (2.0 * intersection + smooth) / (
        pred_prob_masked.sum() + gt_mask_masked.sum() + smooth
    )

    pt = pred_prob_masked.clamp(min=1e-6, max=1.0 - 1e-6)
    focal_loss = -alpha * (1 - pt) ** gamma * gt_mask_masked * torch.log(pt) - (1 - alpha) * pt**gamma * (
        1 - gt_mask_masked
    ) * torch.log(1 - pt)
    focal_loss = focal_loss.mean()

    bce_loss = F.binary_cross_entropy_with_logits(shadow_logits_masked, gt_mask_masked)
    # return pred_mask, bce_loss + 0.1 * dice_loss + 0.1 * focal_loss
    return bce_loss + 0.1 * dice_loss + 0.1 * focal_loss



def compute_shadow_loss_soft(
    z_b: torch.Tensor,
    hori: torch.Tensor,
    cos_theta: torch.Tensor,
    delta: float = 0,
    temperature: float = 0.05,
    weight: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute shadow prediction for inference."""
    del weight

    tan_phi = hori / z_b.clamp(min=1e-3)
    phi_rad = torch.atan(tan_phi)
    phi_deg = torch.rad2deg(phi_rad)
    cummax, _ = torch.cummax(phi_deg, dim=1)
    shadow_logits = (cummax - phi_deg - delta) / temperature
    pred_mask = (torch.sigmoid(shadow_logits) > 0.5).to(torch.uint8) | (tan_phi <= 0)
    mask_theta = cos_theta <= 0
    return pred_mask, phi_rad, mask_theta.to(torch.uint8)


# =============================================================================
# Composite training loss
# =============================================================================


def compute_total_loss(
    z_pre: torch.Tensor,
    rho: torch.Tensor,
    path: torch.Tensor,
    x: torch.Tensor,
    sss_range: torch.Tensor,
    sss_altitude: torch.Tensor,
    sss_true: torch.Tensor,
    shadow_true: torch.Tensor,
    device: torch.device,
    perceptual_loss_fn: Optional[nn.Module],
    epoch: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the stage-wise multi-term training loss for one batch."""
    batch_size = z_pre.shape[0]

    loss_sss_total = 0.0
    loss_shadow_total = 0.0
    loss_z_total = 0.0
    loss_supervised_total = 0.0
    loss_sss_ssim_total = 0.0
    sss_positive_penalty_total = 0.0
    loss_smooth_total = 0.0
    loss_col_smooth_total = 0.0

    stage = min(epoch // 30, 2)

    for b in range(batch_size):
        z_b = z_pre[b, 0]
        rho_b = rho[b, 0]
        path_b = path[b, 0]
        x_b = x[b, 0]
        range_b = sss_range[b]
        altitude_b = sss_altitude[b]
        sss_true_b = sss_true[b, 0]
        shadow_true_b = shadow_true[b, 0]

        slant = compute_real_dis(x_b, range_b)
        hori = compute_real_dis_horizontal(slant, altitude_b)
        cos_theta = compute_theta_new(z_b, hori)

        sss_pred_b = path_b * cos_theta * rho_b
        _, _, _ = compute_shadow_soft(hori, z_b, cos_theta)

        sss_positive_penalty = torch.mean(torch.clamp(-sss_pred_b, min=0))
        sss_pred_b_norm = min_max_normalize(sss_pred_b)
        sss_true_b_norm = min_max_normalize(sss_true_b)

        shadow_mask = (hori > 0).float()
        loss_shadow = compute_shadow_loss_amp(z_b, hori, shadow_true_b, shadow_mask)
        loss_sss = F.mse_loss(sss_pred_b_norm.float(), sss_true_b_norm.float())

        loss_smooth = laplacian_loss(z_b)
        loss_raw_smooth = rowwise_laplacian_loss_mean(z_b)

        sss_true_col_mean = sss_true_b.mean(dim=0)
        path_col_mean = path_b.mean(dim=0)
        loss_path = F.mse_loss(sss_true_col_mean, path_col_mean)
        loss_rho_smooth = laplacian_loss(rho_b)

        if stage == 0:
            loss_sss_ssim = torch.tensor(0.0, device=device)
            width = z_b.shape[1]
            height_map_initial = altitude_b.unsqueeze(1).expand(-1, width)
            loss_z = F.l1_loss(z_b, height_map_initial)
            loss_supervised = loss_z + loss_raw_smooth

        elif stage == 1:
            width = z_b.shape[1]
            height_map_initial = altitude_b.unsqueeze(1).expand(-1, width)
            loss_z = F.l1_loss(z_b, height_map_initial)
            sss_pred_4d = sss_pred_b_norm.unsqueeze(0).unsqueeze(0)
            sss_true_4d = sss_true_b_norm.unsqueeze(0).unsqueeze(0)
            loss_sss_ssim = 1 - pytorch_msssim.ssim(sss_pred_4d.float(), sss_true_4d.float(), data_range=1.0)
            loss_supervised = 0.1 * (loss_z + loss_smooth) + loss_shadow + 0.1 * loss_sss_ssim

        else:
            if perceptual_loss_fn is None:
                raise ValueError("perceptual_loss_fn must be provided when epoch stage >= 2.")
            input_pred = prepare_input(sss_pred_b_norm, device)
            input_gt = prepare_input(sss_true_b_norm, device)
            loss_sss_ssim = perceptual_loss_fn(input_pred, input_gt).float()
            loss_z = torch.tensor(0.0, device=device)
            loss_supervised = (
                0.05 * loss_smooth
                + 0.1 * loss_shadow
                + loss_path
                + loss_sss
                + loss_rho_smooth
                + 0.1 * loss_sss_ssim
            )

        loss_supervised_total += loss_supervised
        loss_sss_total += loss_sss
        loss_shadow_total += loss_shadow
        loss_z_total += loss_z
        loss_sss_ssim_total += loss_sss_ssim
        sss_positive_penalty_total += sss_positive_penalty
        loss_smooth_total += loss_rho_smooth
        loss_col_smooth_total += loss_path

    loss_supervised_total /= batch_size
    loss_sss_total /= batch_size
    loss_shadow_total /= batch_size
    loss_z_total /= batch_size
    loss_sss_ssim_total /= batch_size
    sss_positive_penalty_total /= batch_size
    loss_smooth_total /= batch_size
    loss_col_smooth_total /= batch_size

    loss_other = torch.tensor(
        [
            loss_sss_ssim_total.item(),
            sss_positive_penalty_total.item(),
            loss_smooth_total.item(),
            loss_col_smooth_total.item(),
        ]
    )

    return (
        loss_supervised_total,
        loss_sss_total,
        loss_shadow_total,
        loss_z_total,
        loss_other,
    )



def log_losses_to_txt(
    epoch: int,
    loss_tensor: torch.Tensor,
    loss_names: List[str],
    log_file: str,
) -> None:
    """Append named loss values to a text log file."""
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"Epoch {epoch}:\n")
        for name, val in zip(loss_names, loss_tensor):
            f.write(f"  {name}: {val.item():.6f}\n")
        f.write("\n")


# =============================================================================
# Datasets
# =============================================================================


class SonarImageDataset(Dataset):
    """Dataset for training and validation with sonar image, shadow, range, and altitude."""

    def __init__(
        self,
        data_dir: str,
        range_dir: str,
        altitude_dir: str,
        shadow_dir: str,
        transform: Optional[transforms.Compose] = None,
    ) -> None:
        self.data_dir = data_dir
        self.range_dir = range_dir
        self.altitude_dir = altitude_dir
        self.shadow_dir = shadow_dir
        self.transform = transform

        self.image_files = sorted([f for f in os.listdir(data_dir) if f.endswith((".png", ".jpg", ".jpeg"))])
        self.shadow_files = sorted([f for f in os.listdir(shadow_dir) if f.endswith((".png", ".jpg", ".jpeg"))])
        self.range_files = sorted([f for f in os.listdir(range_dir) if f.endswith(".npy")])
        self.altitude_files = sorted([f for f in os.listdir(altitude_dir) if f.endswith(".npy")])

        dataset_length = len(self.image_files)
        if not (len(self.shadow_files) == len(self.range_files) == len(self.altitude_files) == dataset_length):
            raise ValueError("Input folders must contain the same number of files.")

    def __len__(self) -> int:
        return len(self.image_files)

    @staticmethod
    def _load_grayscale_image(image_path: str) -> torch.Tensor:
        image = Image.open(image_path).convert("L")
        image_np = np.array(image).astype(np.float32) / 255.0
        return torch.tensor(image_np).unsqueeze(0)

    @staticmethod
    def _resize_vector(vector: np.ndarray, target_length: int = 512) -> np.ndarray:
        resized = cv2.resize(vector.reshape(-1, 1), (1, target_length), interpolation=cv2.INTER_LINEAR)
        return resized.flatten().astype(np.float32)

    def __getitem__(self, idx: int):
        image_name = self.image_files[idx]
        shadow_name = self.shadow_files[idx]
        range_name = self.range_files[idx]
        altitude_name = self.altitude_files[idx]

        input_tensor = self._load_grayscale_image(os.path.join(self.data_dir, image_name))
        shadow_tensor = self._load_grayscale_image(os.path.join(self.shadow_dir, shadow_name))

        if self.transform is not None:
            input_tensor = self.transform(input_tensor)
            shadow_tensor = self.transform(shadow_tensor)
            shadow_tensor = (shadow_tensor > 0.5).float()

        sss_range = np.load(os.path.join(self.range_dir, range_name)).astype(np.float32)
        sss_altitude = np.load(os.path.join(self.altitude_dir, altitude_name)).astype(np.float32)

        sss_range_resized = self._resize_vector(sss_range)
        sss_altitude_resized = self._resize_vector(sss_altitude)

        return input_tensor, shadow_tensor, sss_range_resized, sss_altitude_resized, image_name


class TestSonarImageDataset(Dataset):
    """Dataset for inference without shadow labels."""

    def __init__(
        self,
        data_dir: str,
        range_dir: str,
        altitude_dir: str,
        transform: Optional[transforms.Compose] = None,
    ) -> None:
        self.data_dir = data_dir
        self.range_dir = range_dir
        self.altitude_dir = altitude_dir
        self.transform = transform

        self.image_files = sorted([f for f in os.listdir(data_dir) if f.endswith((".png", ".jpg", ".jpeg"))])
        self.range_files = sorted([f for f in os.listdir(range_dir) if f.endswith(".npy")])
        self.altitude_files = sorted([f for f in os.listdir(altitude_dir) if f.endswith(".npy")])

        dataset_length = len(self.image_files)
        if not (len(self.range_files) == len(self.altitude_files) == dataset_length):
            raise ValueError("Input folders must contain the same number of files.")

    def __len__(self) -> int:
        return len(self.image_files)

    @staticmethod
    def _load_grayscale_image(image_path: str) -> torch.Tensor:
        image = Image.open(image_path).convert("L")
        image_np = np.array(image).astype(np.float32) / 255.0
        return torch.tensor(image_np).unsqueeze(0)

    @staticmethod
    def _resize_vector(vector: np.ndarray, target_length: int = 512) -> np.ndarray:
        resized = cv2.resize(vector.reshape(-1, 1), (1, target_length), interpolation=cv2.INTER_LINEAR)
        return resized.flatten().astype(np.float32)

    def __getitem__(self, idx: int):
        image_name = self.image_files[idx]
        range_name = self.range_files[idx]
        altitude_name = self.altitude_files[idx]

        input_tensor = self._load_grayscale_image(os.path.join(self.data_dir, image_name))
        if self.transform is not None:
            input_tensor = self.transform(input_tensor)

        sss_range = np.load(os.path.join(self.range_dir, range_name)).astype(np.float32)
        sss_altitude = np.load(os.path.join(self.altitude_dir, altitude_name)).astype(np.float32)

        sss_range_resized = self._resize_vector(sss_range)
        sss_altitude_resized = self._resize_vector(sss_altitude)

        return input_tensor, sss_range_resized, sss_altitude_resized, image_name


# =============================================================================
# Training and inference utilities
# =============================================================================


def reset_log_files(*file_paths: str) -> None:
    """Create or clear a set of log files."""
    for path in file_paths:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        open(path, "w", encoding="utf-8").close()



def build_dataloaders(config: TrainConfig) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation dataloaders."""
    custom_transform = transforms.Compose([transforms.Resize(config.image_size)])

    train_dataset = SonarImageDataset(
        config.train_image_path,
        config.train_range_path,
        config.train_altitude_path,
        config.train_shadow_path,
        transform=custom_transform,
    )
    val_dataset = SonarImageDataset(
        config.val_image_path,
        config.val_range_path,
        config.val_altitude_path,
        config.val_shadow_path,
        transform=custom_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader



def build_optimizer_and_scheduler(
    model: nn.Module,
    train_loader: DataLoader,
    config: TrainConfig,
):
    """Build optimizer and cosine warmup scheduler."""
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.lr,
        betas=config.betas,
        weight_decay=config.weight_decay,
    )

    # Use the training loader length instead of the validation loader length.
    num_training_steps = config.num_epochs * len(train_loader)
    num_warmup_steps = 20 * len(train_loader)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    return optimizer, scheduler



def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    perceptual_loss_fn: Optional[nn.Module],
    config: TrainConfig,
) -> None:
    """Run the full training and validation loop."""
    best_train_loss = float("inf")

    reset_log_files(
        config.train_loss_file,
        config.val_loss_file,
        config.train_z_loss_file,
        config.train_shadow_loss_file,
        config.train_sss_loss_file,
        config.other_loss_log_file,
    )

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    scaler = GradScaler()

    for epoch in range(config.num_epochs):
        model.train()
        train_loss = 0.0
        sss_loss_total = 0.0
        shadow_loss_total = 0.0
        z_loss_total = 0.0
        other_loss_total = 0.0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.num_epochs}")
        for x, shadow, sss_range, sss_altitude, _ in progress_bar:
            x = x.to(device, non_blocking=True)
            shadow = shadow.to(device, non_blocking=True)
            sss_range = sss_range.to(device, non_blocking=True)
            sss_altitude = sss_altitude.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                z, rho, path = model(x, x, x)
                loss, loss_sss, loss_shadow, loss_z, loss_other = compute_total_loss(
                    z,
                    rho,
                    path,
                    x,
                    sss_range,
                    sss_altitude,
                    x,
                    shadow,
                    device,
                    perceptual_loss_fn,
                    epoch,
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_loss += loss.item()
            sss_loss_total += loss_sss.item()
            shadow_loss_total += loss_shadow.item()
            z_loss_total += loss_z.item()
            other_loss_total += loss_other.detach()

        current_lr = scheduler.get_last_lr()[0]
        avg_train_loss = train_loss / len(train_loader)
        avg_sss_loss = sss_loss_total / len(train_loader)
        avg_shadow_loss = shadow_loss_total / len(train_loader)
        avg_z_loss = z_loss_total / len(train_loader)
        avg_other_loss = other_loss_total / len(train_loader)

        print(f"Epoch {epoch + 1} - LR: {current_lr:.6f}")
        print(f"Epoch [{epoch + 1}/{config.num_epochs}] - Train Loss: {avg_train_loss:.4f}")
        print(f"Epoch [{epoch + 1}/{config.num_epochs}] - Train SSS Loss: {avg_sss_loss:.4f}")
        print(f"Epoch [{epoch + 1}/{config.num_epochs}] - Train Shadow Loss: {avg_shadow_loss:.4f}")
        print(f"Epoch [{epoch + 1}/{config.num_epochs}] - Train Z Loss: {avg_z_loss:.4f}")

        loss_names = ["loss_sss_ssim", "sss_positive_penalty", "loss_smooth", "loss_col_smooth"]
        for name, val in zip(loss_names, avg_other_loss):
            print(f"{name}: {val.item():.4f}")

        log_losses_to_txt(epoch + 1, avg_other_loss, loss_names, config.other_loss_log_file)

        with open(config.train_loss_file, "a", encoding="utf-8") as f:
            f.write(f"{epoch + 1},{avg_train_loss:.6f}\n")
        with open(config.train_z_loss_file, "a", encoding="utf-8") as f:
            f.write(f"{epoch + 1},{avg_z_loss:.6f}\n")
        with open(config.train_shadow_loss_file, "a", encoding="utf-8") as f:
            f.write(f"{epoch + 1},{avg_shadow_loss:.6f}\n")
        with open(config.train_sss_loss_file, "a", encoding="utf-8") as f:
            f.write(f"{epoch + 1},{avg_sss_loss:.6f}\n")

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, shadow, sss_range, sss_altitude, _ in val_loader:
                x = x.to(device, non_blocking=True)
                shadow = shadow.to(device, non_blocking=True)
                sss_range = sss_range.to(device, non_blocking=True)
                sss_altitude = sss_altitude.to(device, non_blocking=True)

                with autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                    z, rho, path = model(x, x, x)
                    loss, *_ = compute_total_loss(
                        z,
                        rho,
                        path,
                        x,
                        sss_range,
                        sss_altitude,
                        x,
                        shadow,
                        device,
                        perceptual_loss_fn,
                        epoch,
                    )
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        with open(config.val_loss_file, "a", encoding="utf-8") as f:
            f.write(f"{epoch + 1},{avg_val_loss:.6f}\n")

        print(f"Epoch {epoch + 1} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")
        if avg_train_loss < best_train_loss:
            best_train_loss = avg_train_loss
            Path(config.save_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), config.save_path)

# =============================================================================
# Main entry
# =============================================================================


def main() -> None:
    """Set up the pipeline and start training."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config = TrainConfig()

    model = PhysDNet().to(device)

    if os.path.exists(config.checkpoint_path):
        checkpoint = torch.load(config.checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint)
        print(f"Loaded checkpoint from: {config.checkpoint_path}")
    else:
        print(f"Checkpoint not found, training from scratch: {config.checkpoint_path}")

    train_loader, val_loader = build_dataloaders(config)
    optimizer, scheduler = build_optimizer_and_scheduler(model, train_loader, config)
    perceptual_loss_fn = PerceptualLossMobileNetV3Large(alpha=1.0, use_gpu=True)

    print("Start training...")
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        perceptual_loss_fn=perceptual_loss_fn,
        config=config,
    )


if __name__ == "__main__":
    main()
