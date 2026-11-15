"""Four-way initial-state evaluation at tau=0; no training or checkpoint changes."""
import argparse
import json
import tarfile
from pathlib import Path
import numpy as np
import torch
from .config import Config, INFERENCE_NS, fingerprint
from .data import generate
from .families import SPECS, bounds
from .model import Transformer
from . import train, joint, joint_train
from .utils import atomic, file_hash, source_hash, write_json, lock, setup_device, environment

FAMILIES=('forced_linear','logistic','stuart_landau')
METHODS=('prediction','parameter','joint_parameter','joint_trajectory')

def evaluate(root,family,seed,train_n,ns,count,device,max_batch,token_budget):
    root=Path(root)
    configs={t:(joint.Config if t=='joint' else Config)(family,t,2**19,train_n,seed=seed,data_seed=seed)
             for t in ('parameter','prediction','joint')}
    paths={t:(joint_train if t=='joint' else train).run_path(root,c)/'best.pt' for t,c in configs.items()}
    hashes={t:file_hash(p) for t,p in paths.items()}
    models={}; epochs={}
    lo,hi=bounds(family); m=len(SPECS[family]['scale']); p=len(lo)
    state_scale=torch.tensor(SPECS[family]['scale'],device=device,dtype=torch.float32)
    midpoint=torch.tensor((lo[:m]+hi[:m])/2,device=device,dtype=torch.float32)
    param_scale=torch.tensor((hi[:m]-lo[:m])/1.6,device=device,dtype=torch.float32)
    for n in ns:
        identity=dict(configs={t:c.to_dict() for t,c in configs.items()},checkpoint_sha256=hashes,
                      n=n,count=count,tau=0.,source=source_hash(['initial_state_cli.py','data.py','families.py','model.py','joint.py']))
        folder=root/'initial_state_experiment'/family/f'train_n{train_n}'/f'n{n}_{fingerprint(identity)}'
        with lock(folder/'evaluation.lock'):
            marker=folder/'metrics.json'; cache=folder/'per_example.npz'
            if marker.exists() and cache.exists():
                rec=json.loads(marker.read_text())
                if rec['identity']!=identity or rec['array_sha256']!=file_hash(cache): raise ValueError(f'Invalid cache: {folder}')
                print(f'Loaded {family} seed={seed} train_n={train_n} n={n}',flush=True)
                continue
            if not models:
                for t,c in configs.items():
                    checkpoint=torch.load(paths[t],map_location='cpu',weights_only=False)
                    if checkpoint['spec']!=(joint_train if t=='joint' else train).run_spec(c):
                        raise ValueError(f'Checkpoint/source mismatch: {paths[t]}')
                    model=(joint.Transformer if t=='joint' else Transformer)(c).to(device)
                    model.load_state_dict(checkpoint['model']);model.eval();models[t]=model
                    epochs[t]=checkpoint['epoch']
            pieces={k:[] for k in ('theta','original_tau','true_initial_state')}
            for method in METHODS:
                for suffix in ('all_token_coordinate_mse','mean_output','mean_output_coordinate_mse'):
                    pieces[f'{method}_{suffix}']=[]
            batch=min(max_batch,max(1,token_budget//n))
            with torch.inference_mode():
                for start in range(0,count,batch):
                    b=generate(configs['prediction'],'test',start,min(batch,count-start),n=n)
                    # Same deterministic test observations as the original prediction evaluations.
                    # Initial-condition coordinates are first in theta for these three families.
                    points=b['points'].copy();points[:,:,-1]=0.
                    x=torch.from_numpy(points).to(device)
                    truth=torch.as_tensor(b['theta'][:,:m],device=device,dtype=x.dtype)/state_scale
                    j=models['joint'](x)
                    param=models['parameter'](x[:,:,:-1])
                    outputs=dict(prediction=models['prediction'](x),parameter=(midpoint+param[:,:,:m]*param_scale)/state_scale,
                                 joint_parameter=(midpoint+j[:,:,:m]*param_scale)/state_scale,joint_trajectory=j[:,:,p:])
                    for method,y in outputs.items():
                        if not torch.isfinite(y).all(): raise FloatingPointError('Nonfinite prediction')
                        mean=y.mean(1)
                        pieces[f'{method}_all_token_coordinate_mse'].append((y-truth[:,None]).square().mean(1).cpu().numpy())
                        pieces[f'{method}_mean_output'].append(mean.cpu().numpy())
                        pieces[f'{method}_mean_output_coordinate_mse'].append((mean-truth).square().cpu().numpy())
                    pieces['theta'].append(b['theta']);pieces['original_tau'].append(b['tau'])
                    pieces['true_initial_state'].append(truth.cpu().numpy())
            values={k:np.concatenate(v) for k,v in pieces.items()}
            atomic(cache,lambda f:np.savez_compressed(f,**values))
            metrics={method:{loss:float(values[f'{method}_{loss}_coordinate_mse'].mean()) for loss in ('all_token','mean_output')} for method in METHODS}
            write_json(marker,dict(identity=identity,metrics=metrics,best_epochs=epochs,array_sha256=file_hash(cache),
                                   units='state normalized by existing family state scales',environment=environment(device)))
            print(f'Saved {family} seed={seed} train_n={train_n} n={n}: {metrics}',flush=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('evaluate','export'))
    parser.add_argument('--seeds',type=int,nargs='+',choices=(20260906,20260907),default=[20260906,20260907])
    parser.add_argument('--families',nargs='+',choices=FAMILIES,default=list(FAMILIES))
    parser.add_argument('--train-ns',type=int,nargs='+',choices=(8,128),default=[8,128])
    parser.add_argument('--ns',type=int,nargs='+',default=list(INFERENCE_NS))
    parser.add_argument('--count',type=int,default=8192)
    parser.add_argument('--device',choices=('cuda','mps','cpu'),default='cuda')
    parser.add_argument('--max-batch',type=int,default=256)
    parser.add_argument('--token-budget',type=int,default=16384)
    parser.add_argument('--root-first',type=Path,default=Path('/export/abdullah/ode-results/paper-v1'))
    parser.add_argument('--root-second',type=Path,default=Path('/export/abdullah/ode-results/paper-seed-20260907'))
    a=parser.parse_args()
    if not 2<=a.count<=8192 or min(*a.ns,a.max_batch,a.token_budget)<1: parser.error('Invalid count or batch/context size')
    if a.root_first.resolve()==a.root_second.resolve(): parser.error('Seed roots must differ')
    roots={20260906:a.root_first,20260907:a.root_second}
    device=setup_device(a.device) if a.command=='evaluate' else None
    for seed in a.seeds:
        root=roots[seed]
        if a.command=='export':
            folder=root/'initial_state_experiment'
            files=sorted([*folder.rglob('metrics.json'),*folder.rglob('per_example.npz')])
            if not files: raise FileNotFoundError(f'No results: {folder}')
            dest=root/f'initial_state_seed_{seed}.tar.gz'
            def pack(f):
                with tarfile.open(fileobj=f,mode='w:gz') as tar:
                    for path in files: tar.add(path,arcname=f'seed_{seed}/'+path.relative_to(root).as_posix())
            atomic(dest,pack);print('Exported:',dest,flush=True)
        else:
            for family in a.families:
                for train_n in a.train_ns:evaluate(root,family,seed,train_n,a.ns,a.count,device,a.max_batch,a.token_budget)

if __name__=='__main__':main()
