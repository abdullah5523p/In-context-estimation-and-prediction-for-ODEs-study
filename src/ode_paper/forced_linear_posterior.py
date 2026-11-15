"""Uniform-box posterior means by randomized, importance-weighted quadrature.

The likelihood is quadratic. A Gaussian reference with precision X'X/q² + I
is sampled sequentially within the box. Importance weights cancel the added
Gaussian factor exactly, leaving the requested uniform prior, including when
the design is rank deficient. Independent scrambled Sobol rules monitor error.
"""
import hashlib
import os
from pathlib import Path
import numpy as np
from scipy.special import log_ndtr, logsumexp
from scipy.stats import qmc, truncnorm
from .config import fingerprint
from .families import SPECS, bounds, decode


def _log_interval(a, b):
    # For positive intervals use survival probabilities to avoid cancellation.
    upper = np.where(a > 0, log_ndtr(-a), log_ndtr(b))
    lower = np.where(a > 0, log_ndtr(-b), log_ndtr(a))
    return upper + np.log(-np.expm1(lower-upper))


def _rule(mean, chol, lower, upper, power, seed):
    u = qmc.Sobol(3, scramble=True, seed=seed).random_base2(power)
    e = np.zeros_like(u)
    logweight = np.zeros(len(u))
    z = np.empty_like(u)
    for j in range(3):
        center = mean[j] + e[:, :j] @ chol[j, :j]
        a, b = (lower[j]-center)/chol[j,j], (upper[j]-center)/chol[j,j]
        logweight += _log_interval(a, b)
        e[:, j] = truncnorm.ppf(np.clip(u[:,j], 1e-15, 1-1e-15), a, b)
        z[:, j] = center + chol[j,j]*e[:,j]
    # Cancels the reference's N(0,I) factor, restoring the uniform-box prior.
    logweight += .5*np.sum(z*z, axis=1)
    w = np.exp(logweight-logsumexp(logweight))
    estimate = w @ z
    centered = z-estimate
    cov = (centered*w[:,None]).T @ centered
    return estimate, cov, 1/np.sum(w*w)


def posterior_one(point, q, seed, min_power=12, max_power=18, replicates=4, tolerance=.01):
    if q <= 0 or replicates < 2 or min_power < 2 or max_power <= min_power or tolerance <= 0:
        raise ValueError('Require q>0, replicates>=2, max_power>min_power>=2 and tolerance>0')
    # Work in normalized parameter coordinates, matching the plotted MSE.
    lo, hi = bounds('forced_linear')
    transform = (hi-lo)/1.6
    t = 4*point[:,0].astype(float)
    design = np.stack((np.exp(-t), -np.expm1(-t), t+np.expm1(-t)), axis=1)
    design /= SPECS['forced_linear']['scale'][0]
    y = point[:,1].astype(float)-design@((lo+hi)/2)
    design = design*transform
    precision = design.T@design/q**2 + np.eye(3)
    covariance = np.linalg.solve(precision, np.eye(3))
    mean = np.linalg.solve(precision, design.T@y/q**2)
    # Condition on the most restricted marginal first.
    order = np.argsort(np.diag(covariance))
    undo = np.argsort(order)
    chol = np.linalg.cholesky(covariance[np.ix_(order, order)])
    previous = None
    for power in range(min_power, max_power+1):
        rules = [_rule(mean[order], chol, np.full(3,-.8), np.full(3,.8), power,
                       int(seed)+104729*r) for r in range(replicates)]
        estimates = np.array([r[0] for r in rules])
        estimate = estimates.mean(0)
        cov = np.mean([r[1]+np.outer(r[0]-estimate,r[0]-estimate) for r in rules],axis=0)
        se = estimates.std(0,ddof=1)/np.sqrt(replicates)
        sd = np.sqrt(np.maximum(np.diag(cov),1e-24))
        shift = np.inf if previous is None else np.max(np.abs(estimate-previous)/sd)
        ess = min(r[2] for r in rules)
        converged = np.max(se/sd) <= tolerance and shift <= 3*tolerance and ess >= 128
        if converged or power == max_power:
            return dict(mean=estimate[undo], covariance=cov[np.ix_(undo,undo)],
                        integration_se=se[undo], min_ess=ess, power=power,
                        converged=converged, relative_se=float(np.max(se/sd)), relative_shift=shift)
        previous = estimate


def cached_posterior(root, points, q, seed, n, **options):
    defaults = dict(min_power=12, max_power=18, replicates=4, tolerance=.01)
    defaults.update(options)
    identity = dict(q=q, seed=seed, n=n, options=defaults,
                    points=hashlib.sha256(points.tobytes()).hexdigest(),
                    source=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    family_source=hashlib.sha256(Path(__file__).with_name('families.py').read_bytes()).hexdigest())
    folder = Path(root)/'statistical_comparison_cache'/'forced_linear_posterior'/fingerprint(identity)
    folder.mkdir(parents=True, exist_ok=True)
    estimates=[]
    worst=0.
    for i, point in enumerate(points):
        cache = folder/f'example_{i:04d}.npz'
        if cache.exists():
            with np.load(cache, allow_pickle=False) as saved:
                result={k:saved[k].copy() for k in saved.files}
        else:
            result=posterior_one(point,q,seed+1009*n+7919*i,**defaults)
            temp=folder/f'example_{i:04d}.building.npz'
            np.savez_compressed(temp,**result)
            os.replace(temp,cache)
        if not bool(result['converged']):
            raise RuntimeError(f'Posterior quadrature did not stabilize for example {i}, n={n}. '
                               'Increase POSTERIOR_SETTINGS max_power and rerun. '
                               f'Diagnostics are saved in {cache}')
        estimates.append(result['mean'])
        worst=max(worst,float(result['relative_se']))
        if (i+1)%32==0 or i+1==len(points):
            print(f'  Posterior seed={seed} n={n}: {i+1}/{len(points)}, '
                  f'max numerical SE/posterior SD={worst:.3g}',flush=True)
    return np.array(estimates)
