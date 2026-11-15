"""Prediction comparisons using existing parameter fits and transformer caches."""
import hashlib
import json
from pathlib import Path
import numpy as np
from .config import Config, fingerprint
from .data import generate
from .families import decode, encode, solution, SPECS
from .local_results import example_arrays, select, checked_path

FIT_SETTINGS = dict(mle_starts=5, mle_max_nfev=1000, sgd_steps=3000, sgd_batch=8, sgd_lr=.05)

def bootstrap_mean_ci(errors, seed, resamples=5000):
    """Pointwise percentile interval; resample whole test trajectories."""
    errors = np.asarray(errors, dtype=float)
    if errors.ndim != 1 or len(errors) < 2 or not np.isfinite(errors).all() or np.any(errors < 0):
        raise ValueError('Expected at least two finite, nonnegative test errors')
    rng = np.random.default_rng(seed)
    means = np.empty(resamples)
    for start in range(0, resamples, 250):
        stop = min(start+250, resamples)
        indices = rng.integers(len(errors), size=(stop-start, len(errors)))
        means[start:stop] = errors[indices].mean(axis=1)
    return np.quantile(means, [.025, .975])

def _cached_fits(root, family, cfg, n, points, count):
    matches = []
    for path in (root/'statistical_comparison_cache'/family).glob('*/settings.json'):
        identity = json.loads(path.read_text())
        settings = identity['settings']
        if (identity['data_seed'] == cfg.data_seed and identity['q'] == cfg.q
                and identity['n'] == n and settings['count'] == count
                and identity['points_hash'] == hashlib.sha256(points.tobytes()).hexdigest()
                and all(settings.get(k) == v for k, v in FIT_SETTINGS.items())
                and (path.parent/'fits.npz').exists()):
            matches.append(path.parent/'fits.npz')
    if len(matches) != 1:
        raise ValueError(f'Expected one matching saved MLE/SGD cache, found {len(matches)}: {family}, seed={cfg.data_seed}, n={n}.')
    with np.load(matches[0], allow_pickle=False) as saved:
        return {k: saved[k].copy() for k in ('mle', 'sgd')}



def _cached_linear_posterior(root, points, q, seed, n):
    """Read only: identical cache identity to forced_linear_posterior.cached_posterior."""
    source = Path(__file__).with_name('forced_linear_posterior.py')
    options = dict(min_power=12, max_power=18, replicates=4, tolerance=.01)
    identity = dict(q=q, seed=seed, n=n, options=options,
                    points=hashlib.sha256(points.tobytes()).hexdigest(),
                    source=hashlib.sha256(source.read_bytes()).hexdigest(),
                    family_source=hashlib.sha256(source.with_name('families.py').read_bytes()).hexdigest())
    folder = root/'statistical_comparison_cache'/'forced_linear_posterior'/fingerprint(identity)
    estimates = []
    for i in range(len(points)):
        path = folder/f'example_{i:04d}.npz'
        if not path.exists():
            raise FileNotFoundError(f'Matching posterior cache missing: {path}. No integration is run here.')
        with np.load(path, allow_pickle=False) as saved:
            mean = saved['mean'].astype(float)
            if (not bool(saved['converged']) or mean.shape != (3,)
                    or not np.isfinite(mean).all()):
                raise ValueError(f'Invalid or unconverged posterior estimate: {path}')
            estimates.append(mean)
    return np.array(estimates)


def _joint_arrays(root, family, seed, train_n, n, prediction_cfg, prediction_bank, parameter_targets, count, return_arrays=False):
    """Load joint outputs and verify their identity and pairing with the direct model."""
    base = root/'evaluation_cache_downloads'/f'seed_{seed}'
    catalog = json.loads((base/'joint_catalog.json').read_text())
    if catalog['format_version'] != 1 or catalog['seed'] != seed:
        raise ValueError('Invalid joint catalog')
    matches = [e for e in catalog['evaluations']
               if e['group'] == 'joint_experiment' and e['config']['task'] == 'joint'
               and e['config']['family'] == family and e['config']['seed'] == seed
               and e['config']['M'] == 2**19 and e['config']['n'] == train_n
               and e['inference_n'] == n]
    if len(matches) != 1:
        raise ValueError(f'Missing or ambiguous joint cache: {family}, {seed}, {train_n}, {n}')
    entry = matches[0]
    cache = checked_path(base, entry['cache'])
    for name in ('metrics.json', 'per_example.npz'):
        path = checked_path(base, (cache/name).relative_to(base).as_posix())
        relative = path.relative_to(base).as_posix()
        if hashlib.sha256(path.read_bytes()).hexdigest() != catalog['sha256'][relative]:
            raise ValueError(f'Corrupted joint cache: {path}')
    identity = json.loads((cache/'metrics.json').read_text())['identity']
    if identity['config'] != entry['config'] or identity['n'] != n:
        raise ValueError('Joint catalog/cache identity mismatch')
    # Only the supervised task differs; data-generation settings must agree.
    cfg = Config(**{**entry['config'], 'task': 'prediction'})
    if cfg != prediction_cfg:
        raise ValueError('Joint and prediction configurations differ beyond task')
    with np.load(cache/'per_example.npz', allow_pickle=False) as saved:
        arrays = {k: saved[k][:count].copy() for k in ('mean_prediction','targets','theta','tau','all_token_coordinate_mse')}
    expected = np.concatenate((encode(prediction_bank['theta'], family).astype(np.float32),
                               prediction_bank['targets']), axis=1)
    for key in ('theta','tau'):
        if not np.array_equal(arrays[key], prediction_bank[key]):
            raise ValueError(f'Joint test-bank mismatch: {key}')
    p = len(SPECS[family]['low'])
    if (not np.array_equal(arrays['targets'], expected)
            or not np.array_equal(arrays['targets'][:,:p], parameter_targets)
            or arrays['mean_prediction'].shape != expected.shape
            or not np.isfinite(arrays['mean_prediction']).all()):
        raise ValueError('Joint output dimensions or test targets mismatch')
    if return_arrays:
        return arrays
    return arrays['mean_prediction'].astype(float)

PREDICTION_LABELS = {
    'Parameter transformer + solution': 'Parameter-only transformer + solution',
    'Prediction transformer': 'Prediction-only transformer',
    'Joint parameter + solution': 'Joint transformer + solution',
    'Joint direct prediction': 'Joint transformer prediction',
}

def compare_prediction(root, rows, family, seeds=(20260906, 20260907),
                       ns=(2,4,8,16,32,64,128,256,512), count=256):
    """No optimization or network inference: regenerate inputs only to verify pairing."""
    import matplotlib.pyplot as plt
    from .paper_style import apply_style, SINGLE_SIZE, PANEL_SIZE, BAND_ALPHA, context_ticks, estimator_label
    apply_style()
    root = Path(root)
    records = []
    scale = np.asarray(SPECS[family]['scale'])
    for seed in seeds:
        for n in ns:
            reference = None
            for train_n in (128,):
                pair = {}
                for task in ('parameter','prediction'):
                    matches = select(rows, group='main_experiment', family=family, task=task,
                                     seed=seed, M=2**19, train_n=train_n, inference_n=n)
                    if len(matches) != 1:
                        raise ValueError(f'Missing or ambiguous {task} cache: {family}, {seed}, {train_n}, {n}')
                    row = matches[0]
                    cfg = Config(**json.loads((row['cache']/'metrics.json').read_text())['identity']['config'])
                    arrays = example_arrays(row)
                    if not 2 <= count <= len(arrays['targets']):
                        raise ValueError('Invalid test subset size')
                    bank = generate(cfg, 'test', 0, count, n=n)
                    for key in ('theta','tau','targets'):
                        if not np.array_equal(bank[key], arrays[key][:count]):
                            raise ValueError(f'Test-bank mismatch for {key}: {row["cache"]}')
                    pair[task] = (cfg, arrays, bank)
                pcfg, params, pb = pair['parameter']
                dcfg, prediction, db = pair['prediction']
                if not (np.array_equal(pb['points'], db['points'][:,:,:-1])
                        and np.array_equal(pb['theta'],db['theta'])
                        and np.array_equal(pb['tau'],db['tau'])):
                    raise ValueError('Parameter/prediction tasks are not paired')
                if reference is not None and not np.array_equal(reference['points'], pb['points']):
                    raise ValueError('Training contexts do not share test observations')
                reference = pb
                # Clean query target in normalized state coordinates, evaluated in float64.
                truth = solution(pb['theta'], pb['tau'][:,None], family)[:,0,:]/scale
                def add(method, z):
                    state = solution(decode(np.asarray(z,dtype=float),family), pb['tau'][:,None], family)[:,0,:]/scale
                    if not np.isfinite(state).all():
                        raise FloatingPointError(f'Non-finite exact trajectory for {method}; no clipping is applied.')
                    records.append(dict(seed=seed,n=n,method=method,errors=((state-truth)**2).mean(1)))
                add(f'Parameter transformer + solution (train n={train_n})', params['mean_prediction'][:count])
                records.append(dict(seed=seed,n=n,method=f'Prediction transformer (train n={train_n})',
                                    errors=((prediction['mean_prediction'][:count].astype(float)-truth)**2).mean(1)))
                joint = _joint_arrays(root, family, seed, train_n, n, dcfg, db,
                                      params['targets'][:count], count)
                p = len(SPECS[family]['low'])
                add(f'Joint parameter + solution (train n={train_n})', joint[:,:p])
                records.append(dict(seed=seed,n=n,method=f'Joint direct prediction (train n={train_n})',
                                    errors=((joint[:,p:]-truth)**2).mean(1)))
                if train_n == 128:
                    fits = _cached_fits(root,family,pcfg,n,pb['points'],count)
                    for key,label in (('mle','Bounded MLE + solution'),('sgd','Projected SGD + solution')):
                        if fits[key].shape != pb['targets'].shape:
                            raise ValueError('Saved parameter estimate shape mismatch')
                        add(label,fits[key])
                    if family == 'forced_linear':
                        posterior = _cached_linear_posterior(root, pb['points'], pcfg.q, seed, n)
                        add('Posterior mean + solution', posterior)
            print(f'Loaded and composed {family}, seed={seed}, n={n}',flush=True)
    methods = list(dict.fromkeys(r['method'] for r in records))
    fig,ax = plt.subplots(figsize=(8.0,3.2))
    from .seed_average import record_curve
    summary = []
    for j, method in enumerate(methods):
        mean, low, high = record_curve(records, method, ns, seeds)
        statistical = method.startswith(('Bounded MLE', 'Projected SGD', 'Posterior mean'))
        direct = 'direct prediction' in method.lower() or method.startswith('Prediction transformer')
        marker = '*' if statistical else '^' if direct else 'o'
        ax.plot(ns, mean, color=f'C{j}', linestyle='-', marker=marker,
                markersize=7 if statistical else 4, linewidth=2.1, zorder=3, label=PREDICTION_LABELS.get(method.replace(f' (train n={128})', ''), estimator_label(method)))
        valid = low > 0
        ax.fill_between(ns, np.where(valid, low, np.nan), high, where=valid, color=f'C{j}', alpha=BAND_ALPHA, linewidth=0, zorder=1)
        for n, value, lo, hi in zip(ns, mean, low, high):
            summary.append(dict(seeds=list(seeds), n=int(n), method=method, mean=float(value),
                                ci95_lower=float(lo), ci95_upper=float(hi),
                                ci_method='within_seed_percentile_bootstrap_of_seed_mean', bootstrap_resamples=5000))
    ax.set(xscale='log',yscale='log',xlabel='Inference observations n',ylabel='Normalized MSE',
           title='Statistical comparison for prediction ('+{'forced_linear': 'FL', 'logistic': 'LG', 'birth_death': 'BD', 'stuart_landau': 'SL'}.get(family, family)+')')
    context_ticks(ax, ns)
    ax.grid(True,which='both',alpha=.15)
    fig.tight_layout()
    ax.legend(loc='upper center', bbox_to_anchor=(.5,-.27), ncol=len(methods) if len(methods) <= 5 else (len(methods)+1)//2, fontsize=8,
              handlelength=1.4, columnspacing=.8, handletextpad=.4, framealpha=.95, borderaxespad=0)
    output=root/'figures'/'prediction_statistical_comparisons'
    output.mkdir(parents=True,exist_ok=True)
    for ext in ('png','pdf'):
        fig.savefig(output/f'{family}.{ext}',dpi=180,bbox_inches='tight')
    (output/f'{family}_summary.json').write_text(json.dumps(summary,indent=2))
    plt.show()
    return records
