"""Conditional estimator-variance experiment at fixed true parameters."""
import json
import hashlib
import os
from pathlib import Path
import numpy as np
import torch
from .config import Config
from .families import bounds, decode, encode, solution, SPECS
from .model import Transformer
from .statistical_comparison import fit_mle

DEFAULT_NS=(2,4,8,16,32,64,128,256,512)

def _anchors(family,count):
    lo,hi=bounds(family); out=np.empty((count,len(lo)))
    for j in range(len(lo)):
        perm=np.random.default_rng(4100+j).permutation(count)
        out[:,j]=lo[j]+(hi[j]-lo[j])*(.1+.8*(perm+.5)/count)
    return out

def _datasets(theta,family,n,reps,seed,q):
    rng=np.random.default_rng(seed)
    t=rng.uniform(0,.8,(reps,n))
    scale=np.asarray(SPECS[family]['scale'])
    y=solution(np.broadcast_to(theta,(reps,len(theta))),t,family)/scale
    y+=q*rng.standard_normal(y.shape)
    return np.concatenate((t[...,None],y),axis=-1).astype(np.float64)

def _load_checkpoint(root,seed,family,train_n,device):
    bank='paper-v1' if seed==20260906 else 'paper-seed-20260907'
    cfg=Config(family,'parameter',2**19,train_n,seed=seed,data_seed=seed)
    from .train import run_path
    path=root/'model_weights'/bank/run_path(Path('.'),cfg).relative_to('.')/'best.pt'
    if not path.is_file(): raise FileNotFoundError(f'Missing checkpoint {path}; extract model_weights.tar.gz into the repo first.')
    raw=torch.load(path,map_location='cpu',weights_only=False)
    model=Transformer(cfg).to(device); model.load_state_dict(raw['model']); model.eval()
    return cfg,model


def _joint_variances(root, family, seeds, ns, theta_grid, repetitions, device, tau):
    from .joint import Config as JointConfig, Transformer as JointTransformer
    from .config import fingerprint
    from .local_results import checked_path
    dim = len(bounds(family)[0])
    results = []
    for seed in seeds:
        base = (root/'evaluation_cache_downloads'/f'seed_{seed}').resolve()
        catalog = json.loads((base/'joint_catalog.json').read_text())
        if catalog['seed'] != seed or catalog['format_version'] != 1:
            raise ValueError('Invalid joint catalog')
        for train_n in (8,128):
            entries = [e for e in catalog['evaluations'] if e['group']=='joint_experiment'
                       and e['config']['family']==family and e['config']['task']=='joint'
                       and e['config']['M']==2**19 and e['config']['n']==train_n
                       and e['config']['seed']==seed]
            runs = {e['run'] for e in entries}
            if len(runs)!=1: raise ValueError(f'Missing/ambiguous joint model: {family}, {seed}, {train_n}')
            cfg = JointConfig(**entries[0]['config'])
            if cfg.data_seed != seed or cfg.q != .02:
                raise ValueError('Joint configuration does not match repeated-dataset protocol')
            path = checked_path(base, next(iter(runs))+'/best.pt')
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != catalog['sha256'][path.relative_to(base).as_posix()]:
                raise ValueError(f'Joint checkpoint checksum mismatch: {path}')
            identity = dict(version=1,config=cfg.to_dict(),checkpoint=digest,ns=list(ns),
                            anchors=theta_grid.tolist(),repetitions=repetitions,device=device,
                            normalized_tau=tau,dataset_protocol='legacy_seed_100000_anchor_1000_nindex',
                            data_source=hashlib.sha256(Path(__file__).with_name('families.py').read_bytes()).hexdigest())
            cache = root/'statistical_comparison_cache'/'conditional_variance'/'joint'/f'{family}_{fingerprint(identity)}.npz'
            if cache.exists():
                with np.load(cache,allow_pickle=False) as saved:
                    z=saved['estimates'].copy()
                print(f'Loaded joint conditional variance: {family}, seed={seed}, train n={train_n}',flush=True)
            else:
                raw=torch.load(path,map_location='cpu',weights_only=False)
                model=JointTransformer(cfg).to(device)
                model.load_state_dict(raw['model']);model.eval()
                z=np.empty((len(theta_grid),len(ns),repetitions,dim))
                for ai,theta in enumerate(theta_grid):
                    for ni,n in enumerate(ns):
                        data=_datasets(theta,family,n,repetitions,seed*100000+ai*1000+ni,cfg.q)
                        data=np.concatenate((data,np.full((*data.shape[:2],1),tau)),axis=-1)
                        with torch.inference_mode():
                            # Batch only across datasets; mean over observation tokens.
                            for start in range(0,repetitions,32):
                                x=torch.as_tensor(data[start:start+32],dtype=torch.float32,device=device)
                                z[ai,ni,start:start+32]=model(x)[...,:dim].mean(1).cpu().numpy()
                    print(f'{family} joint seed={seed} train n={train_n}: anchor {ai+1}/{len(theta_grid)}',flush=True)
                cache.parent.mkdir(parents=True,exist_ok=True)
                temp=cache.with_suffix('.building.npz')
                np.savez_compressed(temp,estimates=z)
                os.replace(temp,cache)
                cache.with_suffix('.json').write_text(json.dumps(identity,indent=2))
            if z.shape!=(len(theta_grid),len(ns),repetitions,dim) or not np.isfinite(z).all():
                raise ValueError('Invalid joint repeated estimates')
            ztrue=np.stack([encode(t,family) for t in theta_grid])
            results.append(dict(seed=seed,method=f'Joint parameter train n={train_n}',estimates=z,
                                variance=np.var(z,axis=2,ddof=1).mean(axis=(0,2)),
                                bias2=np.mean((z.mean(axis=2)-ztrue[:,None,:])**2,axis=(0,2))))
    return results


def run_family(root,family,seeds=(20260906,20260907),ns=DEFAULT_NS,anchors=5,repetitions=128,mle_starts=5,device='cpu',include_joint=False,joint_tau=.5):
    import matplotlib.pyplot as plt
    from .paper_style import apply_style, SINGLE_SIZE, PANEL_SIZE, BAND_ALPHA, context_ticks, estimator_label
    apply_style()
    plt.rcParams.update({'font.size':10, 'axes.labelsize':11, 'axes.titlesize':11,
                         'xtick.labelsize':10, 'ytick.labelsize':10, 'legend.fontsize':9})
    root=Path(root);ns=tuple(ns)
    if family not in ('forced_linear','logistic','stuart_landau'):
        raise ValueError('Use forced_linear, logistic, or stuart_landau.')
    if anchors<2 or repetitions<2: raise ValueError('Need at least two anchors and repetitions.')
    theta_grid=_anchors(family,anchors);dim=len(bounds(family)[0])
    cache=root/'statistical_comparison_cache'/'conditional_variance'/f'{family}.npz'
    info=cache.with_suffix('.json')
    settings=dict(family=family,seeds=list(seeds),ns=list(ns),anchors=anchors,repetitions=repetitions,
                  mle_starts=mle_starts,device=device,version=1)
    results=[]
    if cache.is_file() and info.is_file() and json.loads(info.read_text())==settings:
        with np.load(cache,allow_pickle=False) as z:
            theta_grid=z['anchors']
            for seed in seeds:
                for method in ('MLE','Transformer train n=8','Transformer train n=128'):
                    key=f"s{seed}_{method.replace(' ','_')}"
                    results.append(dict(seed=seed,method=method,variance=z[key+'_variance'].copy(),
                                        bias2=z[key+'_bias2'].copy(),estimates=z[key+'_estimates'].copy()))
        print(f'Loaded cached conditional-variance estimates: {family}')
    else:
        payload={'anchors':theta_grid,'ns':np.asarray(ns)}
        for seed in seeds:
            cfg,m8=_load_checkpoint(root,seed,family,8,device)
            _,m128=_load_checkpoint(root,seed,family,128,device)
            methods=('MLE','Transformer train n=8','Transformer train n=128')
            estimates={m:np.empty((anchors,len(ns),repetitions,dim)) for m in methods}
            ztrue=np.stack([encode(t,family) for t in theta_grid])
            for ai,theta in enumerate(theta_grid):
                for ni,n in enumerate(ns):
                    data=_datasets(theta,family,n,repetitions,seed*100000+ai*1000+ni,cfg.q)
                    for ri,points in enumerate(data):
                        fit,_=fit_mle(points[None],family,mle_starts,1000,seed+ai*100000+ni*1000+ri)
                        estimates['MLE'][ai,ni,ri]=fit[0]
                    x=torch.as_tensor(data,dtype=torch.float32,device=device)
                    with torch.inference_mode():
                        estimates['Transformer train n=8'][ai,ni]=m8(x).mean(1).cpu().numpy()
                        estimates['Transformer train n=128'][ai,ni]=m128(x).mean(1).cpu().numpy()
                print(f'{family} seed={seed}: fixed parameter {ai+1}/{anchors} complete',flush=True)
            for method,z in estimates.items():
                var=np.var(z,axis=2,ddof=1).mean(axis=(0,2))
                bias=z.mean(axis=2)-ztrue[:,None,:]
                bias2=np.mean(np.sum(bias**2,axis=-1)/dim,axis=0)
                result=dict(seed=seed,method=method,variance=var,bias2=bias2,estimates=z)
                results.append(result)
                key=f"s{seed}_{method.replace(' ','_')}"
                payload[key+'_estimates']=z;payload[key+'_variance']=var;payload[key+'_bias2']=bias2
        cache.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(cache,**payload)
        info.write_text(json.dumps(settings,indent=2))
    methods=('MLE','Transformer train n=8','Transformer train n=128')
    if include_joint:
        if not 0 <= joint_tau <= 1: raise ValueError('Normalized query time must lie in [0,1]')
        results.extend(_joint_variances(root,family,seeds,ns,theta_grid,repetitions,device,joint_tau))
        methods += ('Joint parameter train n=8','Joint parameter train n=128')
    fig,ax=plt.subplots(figsize=(8.0,3.2))
    for mi,method in enumerate(methods):
        selected=[next(x for x in results if x['seed']==seed and x['method']==method) for seed in seeds]
        # Average within-model variances; do not pool estimates across models.
        mean_variance=np.mean([r['variance'] for r in selected],axis=0)
        ax.loglog(ns,mean_variance,marker='*' if method=='MLE' else 'o',markersize=7 if method=='MLE' else 4,linestyle='--' if method.endswith('n=8') else '-',color=f'C{mi}',linewidth=2.1,zorder=3,label=method.replace(' train ', ', train '))
    ax.set(xlabel='Observations n',ylabel='Mean conditional\nparameter variance',
           title='Variance reduction comparison ('+{'forced_linear': 'FL', 'logistic': 'LG', 'birth_death': 'BD', 'stuart_landau': 'SL'}.get(family, family)+')')
    ax.grid(True,which='both',alpha=.15);
    fig.tight_layout()
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.27),ncol=len(methods),fontsize=8,
              handlelength=1.4,columnspacing=.8,handletextpad=.4,framealpha=.95,borderaxespad=0)
    out=root/'figures'/'conditional_variance';out.mkdir(parents=True,exist_ok=True)
    fig.savefig(out/f'{family}.png',dpi=180,bbox_inches='tight');fig.savefig(out/f'{family}.pdf',bbox_inches='tight')
    plt.show()
    return results
