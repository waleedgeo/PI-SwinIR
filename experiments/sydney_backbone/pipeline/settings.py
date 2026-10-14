"""Portable settings matching the executed Sydney protocol."""
from pathlib import Path
import json
import random
import time
import numpy as np
import torch
from .models import source_identity

ROOT=Path(__file__).resolve().parents[1]
WEIGHTS=dict(lambda_l1=1.,lambda_slope=.1,lambda_curvature=.05,lambda_flow=.01,flow_iters=2)

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(2);torch.backends.cudnn.benchmark=False

def environment(seed=42):
    return dict(torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),
                total_vram_bytes=torch.cuda.get_device_properties(0).total_memory,
                model_source=source_identity(ROOT),seed=seed,weights=WEIGHTS)

def guard(start,seconds):
    if time.perf_counter()-start>seconds:raise TimeoutError('Bounded run wall limit reached')

def write(path,obj):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8');temp.replace(path)
