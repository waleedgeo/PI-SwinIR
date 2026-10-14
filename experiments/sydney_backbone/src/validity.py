"""Versioned validity operators for revision v1 (True means valid).

These operators do not infer invalidity from elevation zero. Invalid placeholders
are replaced before stencils, and only complete in-bounds supports enter losses.
"""
import torch
import torch.nn.functional as F

MASK_VERSION = 'cageo-r1-mask-v1'
D8_OFFSETS = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]


def validate_mask(mask, reference):
    if mask.shape != reference.shape or reference.ndim != 4 or reference.shape[1] != 1:
        raise ValueError('Expected matching (B,1,H,W) target and validity mask')
    if mask.dtype != torch.bool:
        raise TypeError('Validity must be boolean; never reconstruct it from filled targets')
    return mask.to(device=reference.device)


def stencil_mask(valid, kernel):
    """Every nonzero kernel location must be valid; padded cells are invalid."""
    weights = (kernel != 0).to(device=valid.device, dtype=torch.float32)
    return F.conv2d(valid.float(), weights, padding=kernel.shape[-1]//2) == weights.sum()


def square_mask(valid, radius=1):
    kernel = torch.ones(1,1,2*radius+1,2*radius+1,device=valid.device)
    return stencil_mask(valid, kernel)


def d8_pair_mask(valid):
    kernels = torch.zeros(8,1,3,3,device=valid.device)
    for i,(dr,dc) in enumerate(D8_OFFSETS):
        kernels[i,0,dr+1,dc+1] = 1
    neighbors = F.conv2d(valid.float(),kernels,padding=1) == 1
    return neighbors & valid


def masked_mean(values, valid):
    """Finite supported mean; all-invalid batches give connected differentiable zero."""
    supported = torch.where(valid, values, torch.zeros_like(values))
    if not torch.isfinite(supported).all():
        raise ValueError('Nonfinite value on valid support')
    return supported.sum()/valid.sum().clamp_min(1)


def clean_surfaces(pred, target, valid):
    valid = validate_mask(valid,target)
    if pred.shape != target.shape:
        raise ValueError('Prediction and target shapes must match')
    if not torch.isfinite(target[valid]).all() or not torch.isfinite(pred[valid]).all():
        raise ValueError('Nonfinite elevation on valid support')
    return torch.where(valid,pred,0.), torch.where(valid,target,0.), valid
