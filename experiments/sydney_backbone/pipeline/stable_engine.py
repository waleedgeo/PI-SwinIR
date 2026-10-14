"""Isolated support-pooled engine with overflow-safe gradient-norm arithmetic.

Copied from the existing shared engine; only norm calculation/clipping changes.
No nonfinite gradient is suppressed. Original source remains untouched.
"""
from contextlib import nullcontext
from itertools import islice
import math
import torch
from src.metric_flow_loss import TERM_NAMES


def groups(loader, accumulation):
    if type(accumulation)!=int or accumulation<1:raise ValueError('Positive accumulation required')
    iterator=iter(loader)
    while True:
        group=list(islice(iterator,accumulation))
        if not group:return
        yield group


class SupportedTotals:
    def __init__(self,weights):
        self.weights=weights; self.counts={k:0 for k in TERM_NAMES};self.sums={k:0. for k in TERM_NAMES}
        self.samples=0;self.microbatches=0;self.empty_microbatches=0
    def add(self,terms,counts,n):
        for k in TERM_NAMES:
            value=float(terms[k].detach())
            if not math.isfinite(value):raise ValueError('Nonfinite supported loss')
            self.sums[k]+=value*counts[k];self.counts[k]+=counts[k]
        self.samples+=n;self.microbatches+=1;self.empty_microbatches+=int(counts['l1']==0)
    def result(self):
        means={k:self.sums[k]/self.counts[k] if self.counts[k] else 0. for k in TERM_NAMES}
        return dict(terms=means,term_support_counts=self.counts.copy(),
                    weighted_total=sum(self.weights[k]*means[k] for k in TERM_NAMES),
                    samples=self.samples,microbatches=self.microbatches,empty_microbatches=self.empty_microbatches,
                    reduction='pooled_supported_cells_per_term')


def _batch(batch,device):
    if not {'x','y','valid','sample_id'}.issubset(batch):raise ValueError('Mask and geographic IDs mandatory')
    x,y,valid=(batch[k].to(device) for k in ('x','y','valid'))
    if valid.dtype!=torch.bool or valid.shape!=y.shape or y.ndim!=4 or y.shape[1]!=1:
        raise ValueError('Explicit aligned boolean NCHW mask required')
    if x.ndim!=4 or x.shape[0]!=len(y) or x.shape[2:]!=y.shape[2:] or len(batch['sample_id'])!=len(x):raise ValueError('Batch identity/shape mismatch')
    if not torch.isfinite(x).all() or not torch.isfinite(y[valid]).all():raise ValueError('Nonfinite numeric input/supported target')
    return x,y,valid


def autocast(device,amp):
    return torch.amp.autocast('cuda',dtype=torch.float16) if amp and torch.device(device).type=='cuda' else nullcontext()

def clip_grad_norm_float64(parameters,max_norm):
    """Same global L2 clipping, avoiding float32 sum-of-squares overflow."""
    if not math.isfinite(max_norm) or max_norm<=0:raise ValueError('Positive finite clipping threshold required')
    norms=[torch.linalg.vector_norm(p.grad.to(torch.float64)) for p in parameters if p.grad is not None]
    if not norms:return torch.tensor(0.,dtype=torch.float64)
    norm=torch.linalg.vector_norm(torch.stack(norms))
    if not torch.isfinite(norm):raise ValueError('Nonfinite gradient norm')
    coefficient=min(1.,max_norm/(float(norm)+1e-6))
    with torch.no_grad():
        for p in parameters:
            if p.grad is not None:p.grad.mul_(coefficient)
    return norm


def train_epoch(model,loader,criterion,optimizer,*,device='cpu',accumulation=1,
                scaler=None,amp=False,grad_clip=1.,max_updates=None,guard=lambda:None):
    model.train();totals=SupportedTotals(criterion.weights);updates=0;skipped=0;norms=[]
    for group in groups(loader,accumulation):
        if max_updates is not None and updates>=max_updates:break
        guard()
        # Compute support on CPU. Do not retain an effective batch of GPU inputs/activation graphs.
        counts=[criterion.support_counts(b['valid']) for b in group]
        pooled={k:sum(c[k] for c in counts) for k in TERM_NAMES}
        optimizer.zero_grad(set_to_none=True)
        if not any(pooled[k] and criterion.weights[k] for k in TERM_NAMES):
            skipped+=1
            for b,c in zip(group,counts):totals.add({k:torch.tensor(0.) for k in TERM_NAMES},c,len(b['x']))
            continue # AdamW weight decay must not update parameters on empty supervision.
        for b,c in zip(group,counts):
            guard();x,y,valid=_batch(b,device)
            with autocast(device,amp):pred=model(x)
            if not torch.isfinite(pred).all():raise ValueError('Nonfinite model prediction')
            terms=criterion.loss_terms(pred,y,valid)
            loss=sum(criterion.weights[k]*terms[k]*(c[k]/pooled[k] if pooled[k] else 0.) for k in TERM_NAMES)
            if not torch.isfinite(loss):raise ValueError('Nonfinite objective')
            (scaler.scale(loss) if scaler is not None else loss).backward()
            totals.add(terms,c,len(x))
        if scaler is not None:scaler.unscale_(optimizer)
        parameters=[p for p in model.parameters() if p.grad is not None]
        if any(not torch.isfinite(p.grad).all() for p in parameters):raise ValueError('Nonfinite gradient; optimizer not stepped')
        norm=clip_grad_norm_float64(parameters,grad_clip)
        if not torch.isfinite(norm):raise ValueError('Nonfinite gradient norm')
        if scaler is None:optimizer.step()
        else:scaler.step(optimizer);scaler.update()
        norms.append(float(norm));updates+=1
    result=totals.result();result.update(optimizer_updates=updates,empty_groups_skipped=skipped,gradient_norms=norms)
    return result


@torch.no_grad()
def validate(model,loader,criterion,*,device='cpu',amp=False,guard=lambda:None):
    model.eval();totals=SupportedTotals(criterion.weights);n=0;absolute=square=0.
    for b in loader:
        guard();x,y,valid=_batch(b,device)
        with autocast(device,amp):pred=model(x)
        if not torch.isfinite(pred).all():raise ValueError('Nonfinite validation prediction')
        terms=criterion.loss_terms(pred,y,valid);counts=criterion.support_counts(valid)
        totals.add(terms,counts,len(x))
        # Select before arithmetic so NaN placeholders never contaminate reductions.
        e=(pred.float()[valid]-y[valid])*650.;n+=e.numel();absolute+=float(e.abs().double().sum());square+=float(e.double().square().sum())
    result=totals.result();result.update(valid_pixels=n,mae_m=absolute/n if n else None,rmse_m=math.sqrt(square/n) if n else None)
    return result
