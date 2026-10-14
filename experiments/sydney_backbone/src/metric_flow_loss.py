"""Phase 2 metric, downhill-only local terrain-flow regularization.

Routing uses positive metric D8 slopes, normalized proportionally (power 1).
There is no softmax temperature. Fractions are invariant to a common positive
height scaling: metric units remove the historical compression/temperature
interaction, not that invariance. Sinks have zero outgoing fractions.

A_0=valid; A_(k+1)=valid+incoming(A_k*f) is a finite local accumulation
diagnostic. Sink water terminates rather than being stored across passes. This
is neither full-catchment accumulation nor a hydrodynamic conservation/PDE model.
The composite retains normalized L1/Sobel/Laplacian terms to isolate flow repair;
its default weights are diagnostic defaults, not a final production selection.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.losses import _make_sobel_x, _make_sobel_y, _make_laplacian
from src.validity import (D8_OFFSETS, clean_surfaces, d8_pair_mask, masked_mean,
                          square_mask, stencil_mask, validate_mask)

LOSS_VERSION = 'cageo-r1-metric-downslope-mfd-p1-v2'
TERM_NAMES = ('l1', 'slope', 'curvature', 'flow')


def _finite_parameter(value, name, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f'{name} must be a finite real number')
    if not math.isfinite(value) or (positive and value <= 0):
        raise ValueError(f'{name} must be finite' + (' and positive' if positive else ''))
    return float(value)


def _surface(dem, valid):
    if not isinstance(dem, torch.Tensor) or not dem.is_floating_point():
        raise TypeError('DEM must be a floating point tensor in declared units')
    valid = validate_mask(valid, dem)
    if not torch.isfinite(dem[valid]).all():
        raise ValueError('Nonfinite elevation on valid support')
    dtype = torch.float64 if dem.dtype == torch.float64 else torch.float32
    return torch.where(valid, dem.to(dtype=dtype), 0.), valid


class MetricFlowRouting(nn.Module):
    """Metric D8 MFD (power 1); rows increase south, columns east.

    ``routing`` takes metres, not normalized elevation. Positive fractions route
    from a center to its D8 offset. ``accumulation`` gathers from the opposite
    offset. Cardinal distances are cell_size and diagonals sqrt(2)*cell_size.
    """
    def __init__(self, cell_size=10., num_iters=2):
        super().__init__()
        self.cell_size = _finite_parameter(cell_size, 'cell_size', positive=True)
        if isinstance(num_iters, bool) or not isinstance(num_iters, int) or num_iters < 0:
            raise ValueError('num_iters must be a nonnegative integer')
        self.num_iters = num_iters
        extract = torch.zeros(8, 1, 3, 3)
        gather = torch.zeros_like(extract)
        distances = []
        for i, (dr, dc) in enumerate(D8_OFFSETS):
            extract[i, 0, dr+1, dc+1] = 1.
            gather[i, 0, -dr+1, -dc+1] = 1.
            distances.append(self.cell_size * math.hypot(dr, dc))
        self.register_buffer('extract_kernels', extract)
        self.register_buffer('gather_kernels', gather)
        # Construct in float64 so float64 callers retain exact sqrt(2).
        self.register_buffer('distances', torch.tensor(distances, dtype=torch.float64).view(1, 8, 1, 1))

    def routing(self, z_m, valid):
        """Return fractions, signed valid-pair slopes (m/m), eligible pairs.

        Invalid/out-of-bounds pair slopes are zero. Valid flat/upslope slopes
        remain zero/negative in the diagnostic output but receive no flow mass.
        All-invalid cells and sinks yield finite, differentiable zero fractions.
        """
        dem, valid = _surface(z_m, valid)
        with torch.autocast(device_type=dem.device.type, enabled=False):
            kernels = self.extract_kernels.to(device=dem.device, dtype=dem.dtype)
            neighbors = F.conv2d(dem, kernels, padding=1)
            pairs = d8_pair_mask(valid)
            slopes = (dem - neighbors) / self.distances.to(device=dem.device, dtype=dem.dtype)
            slopes = torch.where(pairs, slopes, 0.)
            if not torch.isfinite(slopes).all():
                raise ValueError('Metric slope arithmetic overflow on valid support')
            eligible = pairs & (slopes > 0)
            weights = torch.where(eligible, slopes, 0.)
            # Max rescaling is algebraically neutral and avoids overflow in the
            # sum or an absolute epsilon dominating shallow positive slopes.
            largest = weights.amax(dim=1, keepdim=True)
            scaled = weights / torch.where(largest > 0, largest, 1.)
            denominator = scaled.sum(dim=1, keepdim=True)
            fractions = scaled / torch.where(denominator > 0, denominator, 1.)
        return fractions, slopes, eligible

    def accumulation(self, fractions, valid):
        """Masked local accumulation with one unit rainfall per valid cell.

        Fractions on invalid/out-of-bounds pairs are ignored before arithmetic.
        On valid pairs they must be finite/nonnegative and outgoing totals <=1.
        The method does not reconstruct downslope eligibility from fractions.
        """
        if not isinstance(fractions, torch.Tensor) or not fractions.is_floating_point():
            raise TypeError('Fractions must be a floating point tensor')
        if fractions.ndim != 4 or fractions.shape[1] != 8:
            raise ValueError('Expected fractions with shape (B,8,H,W)')
        valid = validate_mask(valid, fractions[:, :1])
        dtype = torch.float64 if fractions.dtype == torch.float64 else torch.float32
        with torch.autocast(device_type=fractions.device.type, enabled=False):
            fractions = torch.where(d8_pair_mask(valid), fractions.to(dtype=dtype), 0.)
            if not torch.isfinite(fractions).all() or (fractions < 0).any():
                raise ValueError('Valid-pair fractions must be finite and nonnegative')
            if (fractions.sum(dim=1, keepdim=True) > 1.+1e-5).any():
                raise ValueError('Outgoing fractions exceed one')
            rain = valid.to(dtype=dtype)
            # Connect even a zero-pass/all-invalid result to the routing graph.
            accumulated = rain + fractions.sum(dim=1, keepdim=True) * 0.
            kernels = self.gather_kernels.to(device=fractions.device, dtype=dtype)
            for _ in range(self.num_iters):
                outflow = accumulated * fractions
                inflow = F.conv2d(outflow, kernels, padding=1, groups=8).sum(dim=1, keepdim=True)
                accumulated = torch.where(valid, rain + inflow, 0.)
        return accumulated

    def loss_support(self, valid):
        """Complete radius K+1 support excludes all possible routing dependencies."""
        validate_mask(valid, valid)
        return square_mask(valid, self.num_iters+1)


class MaskedMetricFlowObjective(nn.Module):
    """Normalized non-flow terms plus metric downhill-only local flow comparison.

    ``loss_terms`` returns differentiable supported means. For pooled effective
    groups, weight each term by its own support count (not microbatch count).
    ``support_counts`` exposes those counts without selecting hyperparameters.
    """
    requires_valid_mask = True
    loss_version = LOSS_VERSION

    def __init__(self, lambda_l1=1., lambda_slope=.1, lambda_curvature=.05,
                 lambda_flow=.01, flow_iters=2, cell_size=10.,
                 elevation_min=-150., elevation_max=500.):
        super().__init__()
        self.elevation_min = _finite_parameter(elevation_min, 'elevation_min')
        self.elevation_max = _finite_parameter(elevation_max, 'elevation_max')
        self.elevation_range = self.elevation_max-self.elevation_min
        if not math.isfinite(self.elevation_range) or self.elevation_range <= 0:
            raise ValueError('Elevation bounds must define a finite positive range')
        self.weights = dict(zip(TERM_NAMES, [lambda_l1, lambda_slope, lambda_curvature, lambda_flow]))
        for name, weight in self.weights.items():
            self.weights[name] = _finite_parameter(weight, f'lambda_{name}')
            if weight < 0:
                raise ValueError('Loss weights must be nonnegative')
        self.flow = MetricFlowRouting(cell_size=cell_size, num_iters=flow_iters)
        self.flow_iters = flow_iters
        self.register_buffer('sobel_x', _make_sobel_x())
        self.register_buffer('sobel_y', _make_sobel_y())
        self.register_buffer('laplacian', _make_laplacian())

    def support_masks(self, valid):
        validate_mask(valid, valid)
        return dict(l1=valid,
                    slope=valid & stencil_mask(valid, self.sobel_x) & stencil_mask(valid, self.sobel_y),
                    curvature=stencil_mask(valid, self.laplacian),
                    flow_pairs=d8_pair_mask(valid), flow=self.flow.loss_support(valid))

    def support_counts(self, valid):
        masks = self.support_masks(valid)
        return {name: int(masks[name].sum().item()) for name in TERM_NAMES}

    def loss_terms(self, pred_normalized, target_normalized, valid):
        if not pred_normalized.is_floating_point() or not target_normalized.is_floating_point():
            raise TypeError('Normalized elevations must be floating point tensors')
        if pred_normalized.device != target_normalized.device:
            raise ValueError('Prediction and target must share a device')
        pred, target, valid = clean_surfaces(pred_normalized.float(), target_normalized.float(), valid)
        masks = self.support_masks(valid)
        with torch.autocast(device_type=pred.device.type, enabled=False):
            terms = dict(l1=masked_mean((pred-target).abs(), masks['l1']))
            terms['slope'] = sum(masked_mean((F.conv2d(pred, k.to(pred), padding=1)-
                                             F.conv2d(target, k.to(target), padding=1)).abs(), masks['slope'])
                                 for k in (self.sobel_x, self.sobel_y))
            terms['curvature'] = masked_mean((F.conv2d(pred, self.laplacian.to(pred), padding=1)-
                                              F.conv2d(target, self.laplacian.to(target), padding=1)).abs(),
                                             masks['curvature'])
            pred_m = pred*self.elevation_range+self.elevation_min
            target_m = target*self.elevation_range+self.elevation_min
            fp, _, _ = self.flow.routing(pred_m, valid)
            ft, _, _ = self.flow.routing(target_m, valid)
            ap = self.flow.accumulation(fp, valid)
            at = self.flow.accumulation(ft, valid)
            terms['flow'] = masked_mean((torch.log1p(ap)-torch.log1p(at)).abs(), masks['flow'])
        return terms

    def forward(self, pred_normalized, target_normalized, valid):
        terms = self.loss_terms(pred_normalized, target_normalized, valid)
        total = sum(self.weights[name]*terms[name] for name in TERM_NAMES)
        components = {name: float(value.detach().item()) for name, value in terms.items()}
        components['total'] = float(total.detach().item())
        return total, components
