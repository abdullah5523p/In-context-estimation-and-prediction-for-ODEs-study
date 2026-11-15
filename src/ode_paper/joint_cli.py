"""Joint parameter-estimation and trajectory-prediction models (16 models:
four families x training n=8,128 x seeds 20260906, 20260907)."""
import argparse
import io
import json
import os
import tarfile
from pathlib import Path
from .runs import SEEDS
from .config import INFERENCE_NS, fingerprint
from .joint import configurations as original_configurations
from . import joint_train as training, joint_evaluate as evaluation
from .utils import file_hash

# Keep joint.py/training/evaluation unchanged: their bytes identify old caches.
JOINT_SEEDS = tuple(SEEDS)


def configurations(seed):
    if seed in SEEDS:
        return original_configurations(seed)
    raise ValueError(f'Unsupported joint seed: {seed}')


def export(root,seed,ns):
    files={}; entries=[]
    for cfg in configurations(seed):
        folder=training.run_path(root,cfg)
        best=folder/'best.pt'
        digest=file_hash(best)
        manifest=json.loads((folder/'manifest.json').read_text())
        history=json.loads((folder/'history.json').read_text())
        if manifest['spec']!=training.run_spec(cfg) or history[-1]['epoch']!=cfg.epochs:
            raise ValueError(f'Incomplete/mismatched run: {folder}')
        for name in ('best.pt','manifest.json','history.json','source_snapshot.json'):
            path=folder/name; files[path.relative_to(root).as_posix()]=path
        for n in ns:
            identity=evaluation.evaluation_identity(cfg,digest,n)
            cache=folder/'inference'/f'n{n}_{fingerprint(identity)}'
            rec=json.loads((cache/'metrics.json').read_text())
            if rec['identity']!=identity: raise ValueError('Mismatched evaluation')
            for name in ('metrics.json','per_example.npz'):
                path=cache/name; files[path.relative_to(root).as_posix()]=path
            entries.append(dict(group='joint_experiment',config=cfg.to_dict(),variant='joint',inference_n=n,
                                run=folder.relative_to(root).as_posix(),cache=cache.relative_to(root).as_posix()))
    dest=root/'joint_experiment'/f'joint_seed_{seed}.tar.gz'
    tmp=dest.with_name(f'.{dest.name}.{os.getpid()}.tmp')
    catalog=dict(format_version=1,seed=seed,evaluations=entries,sha256={k:file_hash(v) for k,v in files.items()})
    try:
        with tarfile.open(tmp,'w:gz') as tar:
            for name,path in files.items(): tar.add(path,arcname=f'seed_{seed}/{name}')
            raw=json.dumps(catalog,indent=2).encode(); info=tarfile.TarInfo(f'seed_{seed}/joint_catalog.json'); info.size=len(raw)
            tar.addfile(info,io.BytesIO(raw))
        os.replace(tmp,dest)
    finally:
        if tmp.exists(): tmp.unlink()
    print('Exported best weights, histories and evaluations:',dest,flush=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('list','train','evaluate','export'))
    parser.add_argument('--seeds',type=int,nargs='+',choices=JOINT_SEEDS,default=list(JOINT_SEEDS))
    parser.add_argument('--root-first',type=Path,default=Path('/export/abdullah/ode-results/paper-v1'))
    parser.add_argument('--root-second',type=Path,default=Path('/export/abdullah/ode-results/paper-seed-20260907'))
    parser.add_argument('--device',choices=('cuda','cpu','mps'),default='cuda')
    parser.add_argument('--ns',type=int,nargs='+',default=list(INFERENCE_NS))
    parser.add_argument('--eval-batch-size',type=int,default=256)
    parser.add_argument('--token-budget',type=int,default=16384)
    args=parser.parse_args()
    if args.root_first.resolve()==args.root_second.resolve(): parser.error('Seed roots must differ')
    if len(set(args.seeds))!=len(args.seeds) or len(set(args.ns))!=len(args.ns) or min(args.ns)<=0:
        parser.error('Seeds and positive context sizes must be unique')
    if min(args.eval_batch_size,args.token_budget)<=0: parser.error('Batch budgets must be positive')
    roots=dict(zip(SEEDS,(args.root_first,args.root_second)))
    selected_roots = [roots[seed].resolve() for seed in args.seeds]
    if len(set(selected_roots)) != len(selected_roots): parser.error('Seed roots must differ')
    for seed in args.seeds:
        if args.command=='export': export(roots[seed],seed,args.ns); continue
        configs = configurations(seed)
        for i,cfg in enumerate(configs,1):
            print(f'Joint seed {seed}, configuration {i}/{len(configs)}: {cfg}',flush=True)
            if args.command=='train': training.train(roots[seed],cfg,args.device)
            elif args.command=='evaluate': evaluation.evaluate(roots[seed],cfg,args.device,ns=args.ns,
                token_budget=args.token_budget,max_batch=args.eval_batch_size)

if __name__=='__main__': main()
