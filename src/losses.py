"""
=============================================================================
losses.py — Stage 3: Physics-Informed Loss Functions
=============================================================================
Composite loss for DEM refinement with four differentiable components:

  1. L1 Loss           — pixel-wise reconstruction error
  2. Slope Loss        — Sobel gradient agreement (∂z/∂x, ∂z/∂y)
  3. Curvature Loss    — Laplacian curvature agreement (∇²z)
  4. Flow Routing Loss — lightweight MFD-based flow accumulation

All spatial kernels are fixed-weight nn.Conv2d (no learnable params,
no extra libraries).  Designed for RTX 3070 8 GB.
=============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════
#  Fixed Spatial Kernels
# ═══════════════════════════════════════════════════════════════════════════

def _make_sobel_x():
    """3×3 Sobel kernel for ∂z/∂x (horizontal gradient)."""
    k = torch.tensor([[-1, 0, 1],
                      [-2, 0, 2],
                      [-1, 0, 1]], dtype=torch.float32) / 8.0
    return k.unsqueeze(0).unsqueeze(0)  # (1, 1, 3, 3)


def _make_sobel_y():
    """3×3 Sobel kernel for ∂z/∂y (vertical gradient)."""
    k = torch.tensor([[-1, -2, -1],
                      [ 0,  0,  0],
                      [ 1,  2,  1]], dtype=torch.float32) / 8.0
    return k.unsqueeze(0).unsqueeze(0)


def _make_laplacian():
    """3×3 Laplacian kernel for ∇²z (curvature)."""
    k = torch.tensor([[ 0,  1,  0],
                      [ 1, -4,  1],
                      [ 0,  1,  0]], dtype=torch.float32)
    return k.unsqueeze(0).unsqueeze(0)


# D8 direction offsets — (drow, dcol) for the 8 neighbours
_D8_OFFSETS = [(-1, -1), (-1, 0), (-1, 1),
               ( 0, -1),          ( 0, 1),
               ( 1, -1), ( 1, 0), ( 1, 1)]

# Diagonal distances (√2 for corners, 1 for edges)
_D8_DISTANCES = [1.414, 1.0, 1.414,
                 1.0,        1.0,
                 1.414, 1.0, 1.414]


def _make_d8_kernels():
    """
    Build 8 fixed Conv2d kernels, each extracting one D8 neighbour's value.
    Returns a single (8, 1, 3, 3) weight tensor.
    """
    kernels = torch.zeros(8, 1, 3, 3)
    for i, (dr, dc) in enumerate(_D8_OFFSETS):
        kernels[i, 0, dr + 1, dc + 1] = 1.0
    return kernels


# ═══════════════════════════════════════════════════════════════════════════
#  Component 1 — L1 Loss
# ═══════════════════════════════════════════════════════════════════════════

class L1Loss(nn.Module):
    """Simple L1 pixel loss."""
    def forward(self, pred, target):
        return F.l1_loss(pred, target)


# ═══════════════════════════════════════════════════════════════════════════
#  Component 2 — Slope (Gradient) Loss
# ═══════════════════════════════════════════════════════════════════════════

class SlopeLoss(nn.Module):
    """
    L1 loss on Sobel-derived terrain slopes.
    Compares ∂z/∂x and ∂z/∂y between prediction and target.
    """
    def __init__(self):
        super().__init__()
        self.register_buffer("sobel_x", _make_sobel_x())
        self.register_buffer("sobel_y", _make_sobel_y())

    def _gradient(self, x):
        """Compute (∂z/∂x, ∂z/∂y) via fixed Sobel convolution."""
        gx = F.conv2d(x, self.sobel_x, padding=1)
        gy = F.conv2d(x, self.sobel_y, padding=1)
        return gx, gy

    def forward(self, pred, target):
        gx_p, gy_p = self._gradient(pred)
        gx_t, gy_t = self._gradient(target)
        return F.l1_loss(gx_p, gx_t) + F.l1_loss(gy_p, gy_t)


# ═══════════════════════════════════════════════════════════════════════════
#  Component 3 — Curvature (Laplacian) Loss
# ═══════════════════════════════════════════════════════════════════════════

class CurvatureLoss(nn.Module):
    """
    L1 loss on Laplacian curvature.
    Ensures predicted surface has similar concavity/convexity as target.
    """
    def __init__(self):
        super().__init__()
        self.register_buffer("laplacian", _make_laplacian())

    def forward(self, pred, target):
        lap_p = F.conv2d(pred, self.laplacian, padding=1)
        lap_t = F.conv2d(target, self.laplacian, padding=1)
        return F.l1_loss(lap_p, lap_t)


# ═══════════════════════════════════════════════════════════════════════════
#  Component 4 — Lightweight Flow Routing Loss
# ═══════════════════════════════════════════════════════════════════════════

class FlowRoutingLoss(nn.Module):
    """
    Lightweight differentiable flow accumulation using a simplified
    Multi-Flow Direction (MFD) approach.

    Instead of iterating to steady state (expensive for GPU < 16 GB),
    we run only `num_iters` diffusion-like passes.  This captures
    local drainage patterns without the full flow network.

    Steps:
      1. Compute downslope gradient to each of 8 D8 neighbours.
      2. Apply softmax to get fractional flow proportions.
      3. Run `num_iters` accumulation passes via convolution.
      4. Compare log(FA+1) between pred and target.
    """

    def __init__(self, num_iters=2):
        super().__init__()
        self.num_iters = num_iters

        # D8 neighbour extraction kernels — (8, 1, 3, 3)
        self.register_buffer("d8_kernels", _make_d8_kernels())

        # D8 distances for slope computation
        self.register_buffer(
            "d8_distances",
            torch.tensor(_D8_DISTANCES, dtype=torch.float32).view(8, 1, 1),
        )

        # Distribution kernel — gathers flow from upstream neighbours
        # Each of the 8 kernels picks the *opposite* neighbour
        gather_kernels = torch.zeros(8, 1, 3, 3)
        for i, (dr, dc) in enumerate(_D8_OFFSETS):
            gather_kernels[i, 0, -dr + 1, -dc + 1] = 1.0
        self.register_buffer("gather_kernels", gather_kernels)

    def _compute_flow_fractions(self, dem):
        """
        Compute MFD-like flow fractions from a DEM.

        Parameters
        ----------
        dem : (B, 1, H, W)

        Returns
        -------
        fractions : (B, 8, H, W) — fractional flow to each D8 neighbour
        """
        B, _, H, W = dem.shape

        # Extract neighbour values via depthwise convolution: (B, 8, H, W)
        # expand is memory-efficient (no copy), groups=8 applies each kernel
        neighbours = F.conv2d(
            dem.expand(-1, 8, -1, -1),
            self.d8_kernels,
            padding=1,
            groups=8,
        )

        # Downslope gradient: (centre - neighbour) / distance
        centre = dem.expand_as(neighbours)
        slope = (centre - neighbours) / self.d8_distances.unsqueeze(0)

        # Only downslope (positive slope = water flows to lower neighbour)
        slope = F.relu(slope)

        # Temperature-scaled softmax over 8 directions → flow fractions
        # Temperature > 1 sharpens the distribution, preventing uniform 1/8
        # on flat areas where all slopes are zero after ReLU
        temperature = 10.0
        fractions = F.softmax(slope * temperature, dim=1)  # (B, 8, H, W)

        return fractions

    def _accumulate_flow(self, fractions):
        """
        Run a few diffusion passes to approximate flow accumulation.

        Parameters
        ----------
        fractions : (B, 8, H, W)

        Returns
        -------
        fa : (B, 1, H, W) — approximate flow accumulation
        """
        B, _, H, W = fractions.shape

        # Start with uniform rainfall = 1 everywhere
        fa = torch.ones(B, 1, H, W, device=fractions.device, dtype=fractions.dtype)

        for _ in range(self.num_iters):
            # Multiply fa by fractions → outgoing flow per direction
            outflow = fa * fractions   # (B, 8, H, W)

            # Gather inflow from upstream neighbours
            inflow = F.conv2d(
                outflow,
                self.gather_kernels,
                padding=1,
                groups=8,
            )  # (B, 8, H, W)

            # Sum inflow from all 8 directions + base rainfall
            fa = inflow.sum(dim=1, keepdim=True) + 1.0  # (B, 1, H, W)

        return fa

    def forward(self, pred, target):
        """
        Compute flow routing loss.

        Parameters
        ----------
        pred, target : (B, 1, H, W) — DEM predictions in normalised space
        """
        frac_p = self._compute_flow_fractions(pred)
        frac_t = self._compute_flow_fractions(target)

        fa_p = self._accumulate_flow(frac_p)
        fa_t = self._accumulate_flow(frac_t)

        # Log-space comparison (flow accumulation can span orders of magnitude)
        return F.l1_loss(torch.log1p(fa_p), torch.log1p(fa_t))


# ═══════════════════════════════════════════════════════════════════════════
#  Composite Physics-Informed Loss
# ═══════════════════════════════════════════════════════════════════════════

class PhysicsInformedLoss(nn.Module):
    """
    Weighted composite loss:

        L = λ₁·L1 + λ₂·Slope + λ₃·Curvature + λ₄·Flow

    Parameters
    ----------
    lambda_l1, lambda_slope, lambda_curvature, lambda_flow : float
        Loss component weights.
    flow_iters : int
        Number of flow accumulation diffusion passes (default 2).
    """

    def __init__(self, lambda_l1=1.0, lambda_slope=0.5,
                 lambda_curvature=0.2, lambda_flow=0.1,
                 flow_iters=2):
        super().__init__()

        self.lambda_l1 = lambda_l1
        self.lambda_slope = lambda_slope
        self.lambda_curvature = lambda_curvature
        self.lambda_flow = lambda_flow

        self.l1        = L1Loss()
        self.slope     = SlopeLoss()
        self.curvature = CurvatureLoss()
        self.flow      = FlowRoutingLoss(num_iters=flow_iters)

    def forward(self, pred, target):
        """
        Parameters
        ----------
        pred, target : (B, 1, H, W) — DEM in normalised space

        Returns
        -------
        total : scalar tensor  — composite loss
        components : dict      — individual loss values (detached, for logging)
        """
        loss_l1   = self.l1(pred, target)
        loss_sl   = self.slope(pred, target)
        loss_curv = self.curvature(pred, target)
        loss_flow = self.flow(pred, target)

        total = (
            self.lambda_l1        * loss_l1
            + self.lambda_slope     * loss_sl
            + self.lambda_curvature * loss_curv
            + self.lambda_flow      * loss_flow
        )

        components = {
            "l1":        loss_l1.item(),
            "slope":     loss_sl.item(),
            "curvature": loss_curv.item(),
            "flow":      loss_flow.item(),
            "total":     total.item(),
        }

        return total, components


# ═══════════════════════════════════════════════════════════════════════════
#  Quick self-test
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Testing PhysicsInformedLoss ...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    criterion = PhysicsInformedLoss().to(device)

    pred   = torch.randn(4, 1, 128, 128, device=device, requires_grad=True)
    target = torch.randn(4, 1, 128, 128, device=device)

    loss, comps = criterion(pred, target)
    print(f"  Total loss : {loss.item():.6f}")
    for k, v in comps.items():
        print(f"    {k:12s}: {v:.6f}")

    # Backward pass test
    loss.backward()
    print("  ✓ Backward pass OK — gradients computed.")
