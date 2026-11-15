"""Paired, cached statistical baselines on exported parameter-estimation tests."""
import hashlib
import inspect
import json
import os
import time
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares, lsq_linear
from .config import Config, fingerprint
from .data import generate
from .families import bounds, decode, encode, solution, SPECS
from .local_results import example_arrays, select
from .prediction_comparison import bootstrap_mean_ci


def trajectory(z, times, family):
    """Differentiable physical trajectory; times are normalized by 4."""
    import torch
    lo, hi = bounds(family)
    lo = torch.as_tensor(lo, dtype=z.dtype, device=z.device)
    hi = torch.as_tensor(hi, dtype=z.dtype, device=z.device)
    theta = (lo+hi)/2 + z*(hi-lo)/1.6
    a, b, c = (theta[:, j, None] for j in range(3))
    t = 4*times
    if family == 'forced_linear':
        out = torch.exp(-t)*a - torch.expm1(-t)*b + (t+torch.expm1(-t))*c
    elif family == 'logistic':
        out = c/(1+(c/a-1)*torch.exp(-b*t))
    elif family == 'birth_death':
        r = a-b  # r >= .2 throughout the prior box
        out = .2/(torch.exp(-r*t)+.2*c*(-torch.expm1(-r*t))/r)
    elif family == 'stuart_landau':
        omega = theta[:, 3, None]
        factor = torch.rsqrt(torch.exp(-2*c*t)+(a*a+b*b)/c*(-torch.expm1(-2*c*t)))
        return torch.stack((factor*(a*torch.cos(omega*t)-b*torch.sin(omega*t)),
                            factor*(a*torch.sin(omega*t)+b*torch.cos(omega*t))), dim=-1)
    else:
        raise ValueError(family)
    return out[..., None]


def fit_mle(points, family, starts, max_nfev, seed):
    lo, hi = bounds(family)
    scale = np.asarray(SPECS[family]['scale'])
    rng = np.random.default_rng(seed)
    estimates, diagnostics = [], []
    for point in points:
        times, y = point[:, 0].astype(float), point[:, 1:].astype(float)
        if family == 'forced_linear':
            t = 4*times
            design = np.stack((np.exp(-t), -np.expm1(-t), t+np.expm1(-t)), axis=1)/scale[0]
            fit = lsq_linear(design, y[:, 0], bounds=(lo, hi), tol=1e-11, max_iter=max_nfev)
            estimate = encode(fit.x, family)
            objective = np.mean((design@fit.x-y[:, 0])**2)
            success, optimality = fit.success, fit.optimality
        else:
            def residual(z):
                return (solution(decode(z[None], family), times[None], family)[0]/scale-y).ravel()
            initial = np.vstack((np.zeros(len(lo)), rng.uniform(-.8, .8, (starts-1, len(lo)))))
            fits = [least_squares(residual, z, bounds=(-.8, .8), max_nfev=max_nfev,
                                  ftol=1e-10, xtol=1e-10, gtol=1e-10) for z in initial]
            fit = min(fits, key=lambda f: np.sum(f.fun**2))
            estimate, objective = fit.x, np.mean(fit.fun**2)
            success, optimality = fit.success, fit.optimality
        estimates.append(estimate)
        diagnostics.append((objective, float(success), optimality))
    return np.array(estimates), np.array(diagnostics)


def fit_sgd(points, family, steps, batch, lr, seed):
    """Independent SGD for each task; no batch-of-tasks gradient averaging."""
    import torch
    x = torch.tensor(points, dtype=torch.float64)
    scale = torch.tensor(SPECS[family]['scale'], dtype=torch.float64)
    count, n = x.shape[:2]
    z = torch.zeros((count, len(bounds(family)[0])), dtype=torch.float64, requires_grad=True)
    gen = torch.Generator().manual_seed(seed)
    def loss(p, data):
        return ((trajectory(p, data[:, :, 0], family)/scale-data[:, :, 1:])**2).mean((1, 2))
    initial = loss(z, x).detach()
    for step in range(steps):
        indices = torch.randint(n, (count, min(batch, n)), generator=gen)
        minibatch = x[torch.arange(count)[:, None], indices]
        gradient, = torch.autograd.grad(loss(z, minibatch).sum(), z)
        rate = lr/(1+step/200)**0.6
        with torch.no_grad():
            z.sub_(rate*gradient).clamp_(-.8, .8)
        if not torch.isfinite(z).all():
            raise FloatingPointError('Non-finite SGD parameters')
    final = loss(z, x)
    gradient, = torch.autograd.grad(final.sum(), z)
    projected = (z-(z-gradient).clamp(-.8, .8)).norm(dim=1)
    diagnostics = torch.stack((initial, final.detach(), projected.detach()), dim=1)
    return z.detach().numpy(), diagnostics.numpy()


def losses(z, targets, theta, family):
    if family == 'birth_death':
        p = decode(z, family)
        delta = np.stack(((p[:, 0]-p[:, 1])-(theta[:, 0]-theta[:, 1]),
                          p[:, 2]-theta[:, 2]), axis=1)
        return (delta**2).mean(1)
    return ((z-targets)**2).mean(1)


def compare_family(root, rows, family, seeds=(20260906, 20260907),
                   ns=(2, 4, 8, 16, 32, 64, 128, 256, 512), count=256,
                   mle_starts=5, mle_max_nfev=1000, sgd_steps=3000, sgd_batch=8, sgd_lr=.05, posterior=False, posterior_settings=None,
                   train_ns=(8,128), include_joint=False):
    import matplotlib.pyplot as plt
    from .paper_style import apply_style, SINGLE_SIZE, PANEL_SIZE, BAND_ALPHA, context_ticks, estimator_label
    apply_style()
    plt.rcParams.update({'font.size':10, 'axes.labelsize':11, 'axes.titlesize':11,
                         'xtick.labelsize':10, 'ytick.labelsize':10, 'legend.fontsize':9})
    if min(count, mle_starts, mle_max_nfev, sgd_steps, sgd_batch) < 1 or count < 2 or sgd_lr <= 0:
        raise ValueError('Positive settings and at least two test examples are required')
    root = Path(root)
    digest = hashlib.sha256()
    for name in ('statistical_comparison.py', 'data.py', 'families.py', 'config.py'):
        digest.update(Path(__file__).with_name(name).read_bytes())
    settings = dict(count=count, mle_starts=mle_starts, mle_max_nfev=mle_max_nfev,
                    sgd_steps=sgd_steps, sgd_batch=sgd_batch, sgd_lr=sgd_lr, source=digest.hexdigest())
    # Compatibility is restricted to the exact previous fitting implementations
    # and data sources. Plot/posterior changes do not invalidate those fits.
    fit_digest = hashlib.sha256()
    for fun in (fit_mle, fit_sgd, trajectory):
        fit_digest.update(inspect.getsource(fun).encode())
    for name in ('data.py', 'families.py', 'config.py'):
        fit_digest.update(Path(__file__).with_name(name).read_bytes())
    if fit_digest.hexdigest() == 'd9cf6e1e7a550801edb30f949477dcc57f037c2ed30ef0457b129b924d215c66':
        settings['source'] = 'ad8c66cdfcd80c555a300c72fad5b6e81f3f9f32754b6b4d84fb9822477ab30f'
    records = []
    for seed in seeds:
        for n in ns:
            pair = []
            for train_n in train_ns:
                matches = select(rows, group='main_experiment', family=family, task='parameter',
                                 seed=seed, M=2**19, train_n=train_n, inference_n=n)
                if len(matches) != 1:
                    raise ValueError(f'Expected one cache: {family}, seed={seed}, train_n={train_n}, n={n}')
                row = matches[0]
                record = json.loads((row['cache']/'metrics.json').read_text())
                cfg = Config(**record['identity']['config'])
                arrays = example_arrays(row)
                if count > len(arrays['targets']):
                    raise ValueError('Requested subset exceeds test bank')
                bank = generate(cfg, 'test', 0, count, n=n)
                for key in ('theta', 'targets'):
                    if not np.array_equal(bank[key], arrays[key][:count]):
                        raise ValueError(f'Regenerated {key} differs from cached data: {row["cache"]}')
                if pair and not np.array_equal(bank['points'], pair[0][2]['points']):
                    raise ValueError('The two transformers do not share the same test inputs')
                pair.append((cfg, arrays, bank))
                errors = losses(arrays['mean_prediction'][:count].astype(float), bank['targets'], bank['theta'], family)
                records.append(dict(seed=seed, n=n, method=f'Transformer train n={train_n}', errors=errors))
                if include_joint:
                    from .prediction_comparison import _joint_arrays
                    pcfg = Config(**{**cfg.to_dict(), 'task': 'prediction'})
                    prediction_bank = generate(pcfg, 'test', 0, count, n=n)
                    if not np.array_equal(prediction_bank['points'][:,:,:-1], bank['points']):
                        raise ValueError('Joint and parameter test observations differ')
                    joint = _joint_arrays(root, family, seed, train_n, n, pcfg,
                                          prediction_bank, bank['targets'], count, return_arrays=True)
                    p = len(SPECS[family]['low'])
                    errors = losses(joint['mean_prediction'][:,:p].astype(float), bank['targets'], bank['theta'], family)
                    if errors.shape != (count,) or not np.isfinite(errors).all() or np.any(errors < 0):
                        raise ValueError('Invalid joint parameter errors')
                    records.append(dict(seed=seed, n=n, method=f'Joint parameter train n={train_n}', errors=errors))
            cfg, arrays, bank = pair[0]
            identity = dict(settings=settings, family=family, data_seed=cfg.data_seed, q=cfg.q, n=n,
                            points_hash=hashlib.sha256(bank['points'].tobytes()).hexdigest())
            folder = root/'statistical_comparison_cache'/family/fingerprint(identity)
            folder.mkdir(parents=True, exist_ok=True)
            cache = folder/'fits.npz'
            if cache.exists():
                with np.load(cache, allow_pickle=False) as saved:
                    result = {k: saved[k].copy() for k in saved.files}
                print(f'Loaded {family} seed={seed} n={n}', flush=True)
            else:
                print(f'Fitting {family} seed={seed} n={n}, {count} tasks...', flush=True)
                start = time.perf_counter()
                mle, md = fit_mle(bank['points'], family, mle_starts, mle_max_nfev, seed+n)
                mle_seconds = time.perf_counter()-start
                start = time.perf_counter()
                sgd, sd = fit_sgd(bank['points'], family, sgd_steps, sgd_batch, sgd_lr, seed+n)
                result = dict(mle=mle, sgd=sgd, mle_diagnostics=md, sgd_diagnostics=sd,
                              mle_seconds=mle_seconds, sgd_seconds=time.perf_counter()-start)
                temp = folder/'fits.building.npz'
                np.savez_compressed(temp, **result)
                os.replace(temp, cache)
                (folder/'settings.json').write_text(json.dumps(identity, indent=2))
            md, sd = result['mle_diagnostics'], result['sgd_diagnostics']
            print(f'  MLE solver success {int(md[:,1].sum())}/{count}; median optimality={np.median(md[:,2]):.2g}; '
                  f'SGD median projected gradient={np.median(sd[:,2]):.2g}, '
                  f'objective increased={int(np.sum(sd[:,1]>sd[:,0]+1e-12))}/{count}; '
                  f'fit seconds MLE/SGD={float(result["mle_seconds"]):.1f}/{float(result["sgd_seconds"]):.1f}', flush=True)
            for key, label in (('mle', 'Bounded MLE'), ('sgd', 'Projected SGD')):
                records.append(dict(seed=seed, n=n, method=label,
                                    errors=losses(result[key], bank['targets'], bank['theta'], family)))
            if posterior:
                if family == 'forced_linear':
                    from .forced_linear_posterior import cached_posterior
                    linear_options = {k:v for k,v in (posterior_settings or {}).items()
                                      if k in ('min_power','max_power','replicates','tolerance')}
                    pm = cached_posterior(root, bank['points'], cfg.q, seed, n, **linear_options)
                else:
                    from .nonlinear_posterior import cached_posterior
                    pm = cached_posterior(root, bank['points'], family, cfg.q, seed, n,
                                          result['mle'], **(posterior_settings or {}))
                records.append(dict(seed=seed, n=n, method='Posterior mean',
                                    errors=losses(pm, bank['targets'], bank['theta'], family)))
    fig, ax = plt.subplots(figsize=(8.0,3.2))
    methods = tuple(f'Transformer train n={n}' for n in train_ns)
    if include_joint:
        methods += tuple(f'Joint parameter train n={n}' for n in train_ns)
    methods += ('Bounded MLE', 'Projected SGD')
    if posterior:
        methods += ('Posterior mean',)
    from .seed_average import record_curve
    summary = []
    for j, method in enumerate(methods):
        mean, low, high = record_curve(records, method, ns, seeds)
        statistical = method in ('Bounded MLE', 'Projected SGD', 'Posterior mean')
        ax.plot(ns, mean, color=f'C{j}', linestyle='-', marker='*' if statistical else 'o',
                markersize=7 if statistical else 4, linewidth=2.1, zorder=3, label=(method.replace('Transformer train n=', 'Parameter-only transformer, n=').replace('Joint parameter train n=', 'Joint transformer, n=').replace(', n=128', '')))
        valid = low > 0
        ax.fill_between(ns, np.where(valid, low, np.nan), high, where=valid, color=f'C{j}', alpha=BAND_ALPHA, linewidth=0, zorder=1)
        for n, value, lo, hi in zip(ns, mean, low, high):
            summary.append(dict(seeds=list(seeds), n=int(n), method=method, mean=float(value),
                                ci95_lower=float(lo), ci95_upper=float(hi),
                                ci_method='within_seed_percentile_bootstrap_of_seed_mean', bootstrap_resamples=5000))
    ax.set(xscale='log', yscale='log', xlabel='Inference observations n',
           ylabel='Identifiable-combination MSE\n(physical units)' if family=='birth_death' else 'Normalized MSE',
           title='Statistical comparison for parameter estimation ('+{'forced_linear': 'FL', 'logistic': 'LG', 'birth_death': 'BD', 'stuart_landau': 'SL'}.get(family, family)+')')
    context_ticks(ax, ns)
    ax.grid(True, which='both', alpha=.15)
    fig.tight_layout()
    ax.legend(loc='upper center', bbox_to_anchor=(.5,-.27), ncol=len(methods), fontsize=8,
              handlelength=1.4, columnspacing=.8, handletextpad=.4, framealpha=.95, borderaxespad=0)
    output = root/'figures'/'statistical_comparisons'
    output.mkdir(parents=True, exist_ok=True)
    for extension in ('png', 'pdf'):
        fig.savefig(output/f'{family}.{extension}', dpi=180, bbox_inches='tight')
    (output/f'{family}_summary.json').write_text(json.dumps(dict(settings=settings, posterior=posterior, posterior_settings=posterior_settings, curves=summary), indent=2))
    plt.show()
    return records
