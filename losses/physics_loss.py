"""
Physics-Informed Loss Functions for PI-SwinIR.

Provides a composite loss that combines standard pixel-wise reconstruction
loss with three physics-based regularisation terms:

1. **Gradient Consistency Loss** — Sobel-filtered gradient matching.
2. **Total Variation (TV) Loss** — spatial smoothness regularisation.
3. **Spectral Loss** — frequency-domain (FFT magnitude) matching.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GradientLoss(nn.Module):
    """Gradient consistency loss using Sobel filters.

    Penalises differences in the horizontal and vertical image gradients
    between the super-resolved (SR) output and the high-resolution (HR)
    ground truth, encouraging edge and structure preservation.

    Args:
        reduction (str): ``'mean'`` or ``'sum'``. Default: ``'mean'``.
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

        # Sobel kernels — registered as buffers so they follow the model device
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def _apply_sobel(self, img: torch.Tensor):
        """Apply Sobel filters channel-wise.

        Args:
            img: (B, C, H, W) tensor in [0, 1].

        Returns:
            Tuple of gradient tensors (grad_x, grad_y), each (B, C, H, W).
        """
        B, C, H, W = img.shape
        img_flat = img.reshape(B * C, 1, H, W)
        grad_x = F.conv2d(img_flat, self.sobel_x, padding=1)
        grad_y = F.conv2d(img_flat, self.sobel_y, padding=1)
        grad_x = grad_x.reshape(B, C, H, W)
        grad_y = grad_y.reshape(B, C, H, W)
        return grad_x, grad_y

    def forward(self, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        """Compute gradient consistency loss.

        Args:
            sr: super-resolved image, (B, C, H, W).
            hr: high-resolution target, (B, C, H, W).

        Returns:
            Scalar loss tensor.
        """
        sr_gx, sr_gy = self._apply_sobel(sr)
        hr_gx, hr_gy = self._apply_sobel(hr)
        loss = F.l1_loss(sr_gx, hr_gx, reduction=self.reduction) + F.l1_loss(
            sr_gy, hr_gy, reduction=self.reduction
        )
        return loss


class TotalVariationLoss(nn.Module):
    """Anisotropic Total Variation (TV) loss.

    Encourages spatial smoothness by penalising large pixel differences
    between horizontally and vertically adjacent pixels in the SR output.

    Args:
        reduction (str): ``'mean'`` or ``'sum'``. Default: ``'mean'``.
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, sr: torch.Tensor) -> torch.Tensor:
        """Compute total variation loss.

        Args:
            sr: super-resolved image, (B, C, H, W).

        Returns:
            Scalar loss tensor.
        """
        diff_h = sr[:, :, 1:, :] - sr[:, :, :-1, :]
        diff_w = sr[:, :, :, 1:] - sr[:, :, :, :-1]
        if self.reduction == "mean":
            return diff_h.abs().mean() + diff_w.abs().mean()
        return diff_h.abs().sum() + diff_w.abs().sum()


class SpectralLoss(nn.Module):
    """Frequency-domain spectral loss using 2D FFT magnitude.

    Penalises differences in the FFT magnitude spectrum between the SR
    output and the HR ground truth, enforcing frequency-domain fidelity.

    Args:
        reduction (str): ``'mean'`` or ``'sum'``. Default: ``'mean'``.
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        """Compute spectral loss.

        Args:
            sr: super-resolved image, (B, C, H, W).
            hr: high-resolution target, (B, C, H, W).

        Returns:
            Scalar loss tensor.
        """
        sr_fft = torch.fft.fft2(sr, norm="ortho")
        hr_fft = torch.fft.fft2(hr, norm="ortho")
        sr_mag = torch.abs(sr_fft)
        hr_mag = torch.abs(hr_fft)
        if self.reduction == "mean":
            return F.l1_loss(sr_mag, hr_mag, reduction="mean")
        return F.l1_loss(sr_mag, hr_mag, reduction="sum")


class PhysicsInformedLoss(nn.Module):
    """Composite physics-informed loss for PI-SwinIR.

    Combines a standard pixel reconstruction loss with three
    physics-based regularisation terms:

    .. math::

        L_{total} = L_{pixel}
                  + \\lambda_{grad} \\cdot L_{grad}
                  + \\lambda_{tv}   \\cdot L_{tv}
                  + \\lambda_{spec} \\cdot L_{spectral}

    Args:
        loss_type (str): base pixel loss — ``'l1'`` or ``'l2'``.
            Default: ``'l1'``.
        lambda_grad (float): weight for gradient consistency loss.
            Default: ``0.1``.
        lambda_tv (float): weight for total variation loss.
            Default: ``0.01``.
        lambda_spectral (float): weight for spectral loss.
            Default: ``0.05``.
        reduction (str): reduction for all sub-losses.
            Default: ``'mean'``.
    """

    def __init__(
        self,
        loss_type: str = "l1",
        lambda_grad: float = 0.1,
        lambda_tv: float = 0.01,
        lambda_spectral: float = 0.05,
        reduction: str = "mean",
    ):
        super().__init__()
        self.lambda_grad = lambda_grad
        self.lambda_tv = lambda_tv
        self.lambda_spectral = lambda_spectral

        if loss_type == "l1":
            self.pixel_loss = nn.L1Loss(reduction=reduction)
        elif loss_type == "l2":
            self.pixel_loss = nn.MSELoss(reduction=reduction)
        else:
            raise ValueError(f"Unsupported loss_type '{loss_type}'. Use 'l1' or 'l2'.")

        self.grad_loss = GradientLoss(reduction=reduction)
        self.tv_loss = TotalVariationLoss(reduction=reduction)
        self.spectral_loss = SpectralLoss(reduction=reduction)

    def forward(self, sr: torch.Tensor, hr: torch.Tensor):
        """Compute the composite physics-informed loss.

        Args:
            sr: super-resolved output, (B, C, H, W), values in [0, 1].
            hr: high-resolution ground truth, (B, C, H, W), values in [0, 1].

        Returns:
            Tuple[Tensor, dict]: total loss scalar and a dict of
            individual loss components for logging.
        """
        l_pixel = self.pixel_loss(sr, hr)
        l_grad = self.grad_loss(sr, hr)
        l_tv = self.tv_loss(sr)
        l_spectral = self.spectral_loss(sr, hr)

        total = (
            l_pixel
            + self.lambda_grad * l_grad
            + self.lambda_tv * l_tv
            + self.lambda_spectral * l_spectral
        )

        components = {
            "loss_pixel": l_pixel.item(),
            "loss_grad": l_grad.item(),
            "loss_tv": l_tv.item(),
            "loss_spectral": l_spectral.item(),
            "loss_total": total.item(),
        }
        return total, components
