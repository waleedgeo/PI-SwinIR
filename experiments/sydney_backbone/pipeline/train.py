"""Fixed fresh Sydney 50-epoch pair, using the established stable FP32 trainer."""
import argparse
import csv
import gc
import json
import math
from pathlib import Path
import shutil
import time
import traceback
import torch
from torch.utils.data import DataLoader
from .data import PatchDataset,sha256
from .models import build_model,parameter_count,source_identity
from .settings import ROOT,WEIGHTS,seed_all,environment,guard
from .stable_engine import train_epoch,validate
from .settings import write
from src.metric_flow_loss import MaskedMetricFlowObjective,LOSS_VERSION

EPOCHS=50
SEED=42

@torch.no_grad()
def scores(model,loader):
    n=0;absolute=square=signed=0.
    model.eval()
    for b in loader:
        x,y,valid=(b[k].cuda() for k in ('x','y','valid'))
        pred=model(x)
        if not torch.isfinite(pred[valid]).all():raise ValueError('Nonfinite supported test prediction')
        e=(pred[valid]-y[valid]).double()*650;n+=e.numel()
        absolute+=float(e.abs().sum());square+=float(e.square().sum());signed+=float(e.sum())
    if not n:raise ValueError('Empty evaluation support')
    return dict(valid_pixels=n,mae_m=absolute/n,rmse_m=math.sqrt(square/n),bias_m=signed/n)

def paired_fp32(data,out):
    out.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    datasets={role:PatchDataset(data,role) for role in ('train','val','test')}
    loader=lambda role,shuffle=False,seed=42:DataLoader(datasets[role],batch_size=1,shuffle=shuffle,num_workers=0,
        generator=torch.Generator().manual_seed(seed),drop_last=False)
    env=environment(SEED);env.update(precision='float32 model and objective; float64 global clipping norm',scaler_initial_scale=None,tf32=False)
    log=dict(kind='Exploratory Sydney fresh 50-epoch paired follow-up',environment=env,loss_version=LOSS_VERSION,
        data_manifest_sha256=sha256(data/'manifest.json'),epochs=EPOCHS,microbatch=1,accumulation=8,
        lr=2e-4,lr_min=1e-6,warmup_epochs=2,selection='minimum support-pooled validation weighted total',
        primary_metric='test RMSE m',numerical_policy='FP32; microbatch1/accumulation8; float64 L2 clipping norm',
        source_datum_compatibility='not independently verified',models=[],status='running')
    write(out/'pilot.json',log)
    for key in ('unet','swin'):
        seed_all(SEED);model=build_model(key,source_root=ROOT).cuda()
        criterion=MaskedMetricFlowObjective(**WEIGHTS,cell_size=10.).cuda()
        optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4)
        row=dict(backbone=key,parameters=parameter_count(model),epochs=[],selected_epoch=None,best_loss=None)
        log['models'].append(row);write(out/'pilot.json',log);best=math.inf;torch.cuda.reset_peak_memory_stats()
        for epoch in range(EPOCHS):
            guard(started,9000)
            lr=2e-4*(epoch+1)/2 if epoch<2 else 1e-6+(2e-4-1e-6)*.5*(1+math.cos(math.pi*(epoch-2)/max(1,EPOCHS-3)))
            for group in optimizer.param_groups:group['lr']=lr
            torch.cuda.synchronize();t=time.perf_counter()
            tr=train_epoch(model,loader('train',True,SEED+epoch),criterion,optimizer,device='cuda',accumulation=8,
                scaler=None,amp=False,grad_clip=1.,guard=lambda:guard(started,9000))
            torch.cuda.synchronize();train_seconds=time.perf_counter()-t;t=time.perf_counter()
            val=validate(model,loader('val'),criterion,device='cuda',amp=False,guard=lambda:guard(started,9000))
            torch.cuda.synchronize();val_seconds=time.perf_counter()-t;improved=val['weighted_total']<best
            if improved:
                best=val['weighted_total'];row['selected_epoch']=epoch+1;row['best_loss']=best
                torch.save(dict(model_state=model.state_dict(),backbone=key,loss_version=LOSS_VERSION,
                    manifest_sha256=log['data_manifest_sha256'],epoch=epoch+1,seed=SEED,source=source_identity(ROOT),
                    numerical_policy=log['numerical_policy']),out/f'{key}_epoch_{epoch+1:02d}.pt')
            row['epochs'].append(dict(epoch=epoch+1,lr=lr,train=tr,validation=val,train_s=train_seconds,validation_s=val_seconds,selected=improved))
            write(out/'pilot.json',log)
            print(f'{key} FP32 epoch {epoch+1}/{EPOCHS}: train {train_seconds:.1f}s; valRMSE {val["rmse_m"]:.4f}m',flush=True)
        row.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
        cp=torch.load(out/f'{key}_epoch_{row["selected_epoch"]:02d}.pt',map_location='cuda',weights_only=False)
        model.load_state_dict(cp['model_state'],strict=True);row['test']=scores(model,loader('test'));row['validation_selected']=scores(model,loader('val'))
        write(out/'pilot.json',log);del model,criterion,optimizer,cp;gc.collect();torch.cuda.empty_cache()
    log.update(status='completed',elapsed_s=time.perf_counter()-started);write(out/'pilot.json',log)
