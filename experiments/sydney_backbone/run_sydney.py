"""Prepare, train or verify the fixed two-backbone Sydney experiment."""
import argparse
import csv
import gc
import json
import math
from pathlib import Path
import numpy as np
import torch
from pipeline.data import sha256
from pipeline.models import build_model,source_identity
from pipeline.settings import ROOT,write

def read(path):return json.loads(path.read_text(encoding='utf-8'))

def verify(data,checkpoints):
    """Independent NumPy physical-unit reduction of selected saved checkpoints."""
    manifest=read(data/'manifest.json');record=read(checkpoints/'pilot.json')
    assert manifest['city']=='Sydney' and record['status']=='completed' and record['epochs']==50
    assert sha256(data/manifest['coordinates']['path'])==manifest['coordinates']['sha256']
    for role in manifest['roles'].values():
        for a in role['arrays'].values():assert sha256(data/a['path'])==a['sha256']
    x=np.load(data/'test_x.npy',mmap_mode='r');y=np.load(data/'test_y.npy',mmap_mode='r');mask=np.load(data/'test_valid.npy',mmap_mode='r')
    rows=[]
    for key in ('unet','swin'):
        row=next(m for m in record['models'] if m['backbone']==key)
        assert len(row['epochs'])==50
        assert min(row['epochs'],key=lambda e:e['validation']['weighted_total'])['epoch']==row['selected_epoch']
        cp=torch.load(checkpoints/f'{key}_epoch_{row["selected_epoch"]:02d}.pt',map_location='cpu',weights_only=False)
        assert cp['backbone']==key and cp['seed']==42 and cp['manifest_sha256']==record['data_manifest_sha256']
        assert cp['source']['sha256']==source_identity(ROOT)['sha256']
        model=build_model(key,source_root=ROOT).cuda().eval();model.load_state_dict(cp['model_state'],strict=True)
        n=0;absolute=square=signed=0.
        with torch.inference_mode():
            for i in range(len(x)):
                m=np.asarray(mask[i:i+1]);pred=model(torch.from_numpy(np.array(x[i:i+1],copy=True)).cuda()).cpu().numpy()
                assert np.isfinite(pred[m]).all()
                e=(pred[m].astype(np.float64)-np.asarray(y[i:i+1])[m].astype(np.float64))*650
                n+=len(e);absolute+=np.abs(e).sum();square+=np.square(e).sum();signed+=e.sum()
        values=dict(valid_pixels=n,mae_m=float(absolute/n),rmse_m=float(math.sqrt(square/n)),bias_m=float(signed/n))
        assert n==row['test']['valid_pixels']==356605
        assert all(abs(values[k]-row['test'][k])<1e-4 for k in ('mae_m','rmse_m','bias_m'))
        rows.append(dict(model=key,**values));print(json.dumps(rows[-1]),flush=True)
        del model,cp;gc.collect();torch.cuda.empty_cache()
    return rows

def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='mode',required=True)
    p=sub.add_parser('prepare');p.add_argument('--root',type=Path,required=True,help='Root containing data/raw/Sydney_* rasters');p.add_argument('--output',type=Path,required=True)
    p=sub.add_parser('train');p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p=sub.add_parser('verify');p.add_argument('--data',type=Path,required=True);p.add_argument('--checkpoints',type=Path,required=True)
    args=parser.parse_args()
    if args.mode=='prepare':
        from pipeline.prepare import prepare_city
        prepare_city('Sydney',args.output,root=args.root,seed=42);return
    torch.set_num_threads(2);torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if args.mode=='verify':verify(args.data,args.checkpoints);return
    from pipeline.train import paired_fp32
    manifest=read(args.data/'manifest.json')
    if manifest['city']!='Sydney':raise ValueError('This release is Sydney only')
    # Both budgets fixed before training. Existing output directories are refused.
    paired_fp32(args.data,args.output)
    p=read(args.output/'pilot.json');rows=[]
    for m in p['models']:
        rows.append(dict(model=m['backbone'],selected_epoch=m['selected_epoch'],optimizer_updates=sum(e['train']['optimizer_updates'] for e in m['epochs']),**m['test']))
    with (args.output/'comparison.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

if __name__=='__main__':main()
