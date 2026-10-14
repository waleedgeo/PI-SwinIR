"""Small per-city masked preparation using geometry-only, buffered windows."""
import json
from pathlib import Path
import numpy as np
import rasterio
from rasterio.windows import Window
from .data import CHANNELS,NORMALIZATION,PATCH,normalize_features,aggregate_target,sha256

DEVELOPMENT_CITIES=('Sydney',)
# Fixed before fitting, based on the read-only mask-geography audit. Coastal
# eastern voids make the east-west roles unusable in Miami and Sydney.
CITY_AXES={'Sydney':'north_south'}

def city_windows(city,height,width,seed=42):
    if city not in DEVELOPMENT_CITIES:raise ValueError('SF is excluded from development training')
    if min(height,width)<4608:raise ValueError('Grid cannot support the predeclared buffered stripes')
    rng=np.random.default_rng(seed)
    transverse=height if CITY_AXES[city]=='west_east' else width
    grids=dict(train=(range(0,transverse-PATCH+1,64),range(0,2433,64),256),
               val=(range(0,transverse-PATCH+1,PATCH),(3072,3200,3328),64),
               test=(range(0,transverse-PATCH+1,PATCH),(4096,4224,4352,4480),64))
    windows={}
    for role,(rows,cols,count) in grids.items():
        candidates=[(r,c) if CITY_AXES[city]=='west_east' else (c,r) for r in rows for c in cols]
        selected=rng.choice(len(candidates),count,replace=False)
        windows[role]=[dict(row=int(candidates[i][0]),col=int(candidates[i][1]),height=PATCH,width=PATCH,
                             sample_id=f'{city}_{role}_r{candidates[i][0]}_c{candidates[i][1]}') for i in selected]
    return windows

def headers(features,reference):
    if features.count!=9 or reference.count!=1 or features.res!=(10.,10.):raise ValueError('Unexpected 9-band/10 m interface')
    if features.crs is None or reference.crs is None or features.crs!=reference.crs or not features.crs.is_projected or features.crs.linear_units!='metre':raise ValueError('Projected metre grids required')
    ratio=features.res[0]/reference.res[0]
    if ratio not in (2.,10.) or reference.res[0]!=reference.res[1]:raise ValueError('Require 1 m or 5 m isotropic reference')
    factor=int(ratio)
    if features.bounds!=reference.bounds or reference.shape!=(features.height*factor,features.width*factor):raise ValueError('Non-nested source grids')
    if (features.transform.b,features.transform.d,reference.transform.b,reference.transform.d)!=(0.,0.,0.,0.):raise ValueError('Rotated grid requires separate preparation')
    if features.transform.a!=10 or features.transform.e!=-10 or reference.transform.a!=10/factor or reference.transform.e!=-10/factor:
        raise ValueError('Unexpected raster orientation/spacing')
    if tuple(features.descriptions)!=CHANNELS[:9] or reference.descriptions!=('Elevation_m',):raise ValueError('Band descriptions do not establish expected physical interface')
    if any(d!='float32' for d in features.dtypes+reference.dtypes):raise ValueError('Unverified source decoding')
    if any(s!=1 for s in features.scales+reference.scales) or any(o!=0 for o in features.offsets+reference.offsets):raise ValueError('Unexpected scaling; no implicit decoding')
    if features.nodata!=-9999 or reference.nodata!=-9999:raise ValueError('Unexpected declared NoData')
    return factor

def prepare_city(city,output,root=None,seed=42):
    root=Path(root);output=Path(output);output.mkdir(parents=True,exist_ok=False)
    paths={'features':root/f'data/raw/{city}_Features_10m.tif'}
    targets=list((root/'data/raw').glob(f'{city}_GroundTruth_*m.tif'))
    if len(targets)!=1:raise ValueError('Unique reference required')
    paths['reference']=targets[0]
    identities={k:dict(path=str(p),bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for k,p in paths.items()}
    with rasterio.open(paths['features']) as features,rasterio.open(paths['reference']) as reference:
        factor=headers(features,reference)
        windows=city_windows(city,features.height,features.width,seed)
        coords=output/'coordinates.json';coords.write_text(json.dumps(dict(city=city,seed=seed,windows=windows),indent=2),encoding='utf-8')
        manifest=dict(schema='r1c1-city-frozen-patches-v1',city=city,seed=seed,channels=CHANNELS,
            normalization=NORMALIZATION,mask_rule='all nine feature masks/finite values AND complete fine-reference coverage',
            reference_aggregation=f'exact aligned {factor}x{factor} supported arithmetic mean; no target clipping/sign filter',
            reference_spacing_m=reference.res[0],source_identity=identities,source_full_hashes_computed=False,
            source_qualification='Physical-unit metadata and nested grids checked; original source lineage/acquisition dates, vertical-datum compatibility and actual LiDAR land/coverage independently unverified.',
            coordinates=dict(path=coords.name,sha256=sha256(coords)),crs=str(features.crs),transform=list(features.transform)[:6],
            bounds=list(features.bounds),split_buffer_m=4400,geometry_gaps_m=[5120,6400],roles={})
        manifest['split_axis']=CITY_AXES[city]
        manifest['orientation_rule']='Sydney north-south geometry frozen before fitting from input-validity geography'
        for role,samples in windows.items():
            shape=(len(samples),1,PATCH,PATCH)
            definitions=dict(x=((len(samples),11,PATCH,PATCH),np.float32),y=(shape,np.float32),valid=(shape,np.bool_),
                feature_valid=((len(samples),9,PATCH,PATCH),np.bool_),target_valid=(shape,np.bool_),coverage=(shape,np.float32))
            arrays={k:np.lib.format.open_memmap(output/f'{role}_{k}.npy',mode='w+',dtype=dtype,shape=sz) for k,(sz,dtype) in definitions.items()}
            counts=dict(valid_pixels=0,partially_covered_target_cells=0,unavailable_target_cells=0,empty_common_support_patches=0)
            target_range=[float('inf'),float('-inf')]
            for i,sample in enumerate(samples):
                r,c=sample['row'],sample['col'];win=Window(c,r,PATCH,PATCH)
                raw=features.read(window=win);fv=(features.read_masks(window=win)>0)&np.isfinite(raw)
                tw=Window(c*factor,r*factor,PATCH*factor,PATCH*factor)
                fine=reference.read(1,window=tw);supported=(reference.read_masks(1,window=tw)>0)&np.isfinite(fine)
                target,coverage,target_valid=aggregate_target(fine,supported,factor)
                valid=fv.all(axis=0,keepdims=True)&(coverage>=1-1e-6)
                values=dict(x=normalize_features(raw,fv),y=np.where(target_valid,(target+150)/650,0).astype(np.float32),
                            valid=valid,feature_valid=fv,target_valid=target_valid,coverage=coverage)
                for key,value in values.items():arrays[key][i]=value
                counts['valid_pixels']+=int(valid.sum());counts['empty_common_support_patches']+=int(not valid.any())
                counts['partially_covered_target_cells']+=int(((coverage>0)&(coverage<1)).sum())
                counts['unavailable_target_cells']+=int((coverage==0).sum())
                if valid.any():
                    target_range[0]=min(target_range[0],float(target[valid].min()));target_range[1]=max(target_range[1],float(target[valid].max()))
                if (i+1)%64==0:print(f'Prepared {city} {role}: {i+1}/{len(samples)} windows',flush=True)
            for array in arrays.values():array.flush()
            arrays.clear()
            if not counts['valid_pixels']:raise ValueError(f'No valid {city} {role} support; no error-based resampling')
            specs={k:dict(path=f'{role}_{k}.npy',shape=list(sz),dtype=np.dtype(dtype).name,sha256=sha256(output/f'{role}_{k}.npy')) for k,(sz,dtype) in definitions.items()}
            manifest['roles'][role]=dict(samples=samples,arrays=specs,target_range_m=target_range,**counts)
    for k,p in paths.items():
        if p.stat().st_size!=identities[k]['bytes'] or p.stat().st_mtime_ns!=identities[k]['mtime_ns']:raise ValueError('Source changed during preparation')
    path=output/'manifest.json';path.write_text(json.dumps(manifest,indent=2,allow_nan=False),encoding='utf-8')
    return path
