"""Truncated-Gaussian mixture posterior via adaptive scrambled Sobol quadrature.
Promoted from the isolated Bayes/MLE diagnostic; no training dependencies.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import lsq_linear
from scipy.special import log_ndtr, logsumexp
from scipy.stats import qmc, truncnorm, norm

VERSION = 1
BOUND = 0.8
STATE_SCALE = 0.6532414253078556


def design(t):
    return np.stack((np.exp(-t), -np.expm1(-t), t+np.expm1(-t)), axis=-1)


def components(prior):
    if prior == 'uniform':
        return [np.zeros(3)]
    if prior == 'one_mode':
        return [np.full(3, .2)]
    if prior == 'two_mode':
        return [np.full(3, -.4), np.full(3, .4)]
    raise ValueError(prior)


def log_interval(a, b):
    """Stable log(Phi(b)-Phi(a)), using survival probabilities in right tail."""
    left = np.where(a > 0, -b, a)
    right = np.where(a > 0, -a, b)
    hi, lo = log_ndtr(right), log_ndtr(left)
    return hi + np.log(-np.expm1(np.minimum(lo-hi, -np.finfo(float).eps)))


def gaussian_box(mean, covariance, u):
    """Sequential truncated-normal importance quadrature (GHK)."""
    # Permute narrow marginal intervals first; the permutation is deterministic.
    perm = np.argsort(np.diag(covariance))[::-1]
    mu = mean[perm]
    L = np.linalg.cholesky(covariance[np.ix_(perm, perm)])
    z = np.zeros_like(u)
    logw = np.zeros(len(u))
    for j in range(3):
        shift = mu[j] + z[:, :j] @ L[j, :j]
        a, b = (-BOUND-shift)/L[j,j], (BOUND-shift)/L[j,j]
        logw += log_interval(a, b)
        z[:,j] = truncnorm.ppf(u[:,j], a, b)
    x = mu + z @ L.T
    norm = logsumexp(logw)
    w = np.exp(logw-norm)
    avg = np.empty(3)
    avg[perm] = w @ x
    ess = 1/np.sum(w*w)
    return avg, norm-np.log(len(u)), ess


def posterior_components(A, y, sigma, prior, s):
    precision = A.T@A / sigma**2
    h = A.T@y / sigma**2
    out = []
    for mu in components(prior):
        alpha = 0. if prior == 'uniform' else 1/s**2
        P = precision + alpha*np.eye(3)
        cf = cho_factor(P)
        cov = cho_solve(cf, np.eye(3))
        mean = cho_solve(cf, h+alpha*mu)
        # Integrate the completed square; constants common to components cancel.
        logbase = -.5*(np.sum((y-A@mean)**2)/sigma**2
                       + alpha*np.sum((mean-mu)**2))
        logbase -= .5*np.linalg.slogdet(P)[1]
        if prior != 'uniform':
            logbase -= 3*np.log(s)
            logbase -= np.sum(log_interval((-BOUND-mu)/s, (BOUND-mu)/s))
        logbase -= np.log(len(components(prior)))
        out.append((mean,cov,logbase))
    return out


def posterior(A, y, sigma, prior, s, seed, powers=(10,12,14,16),
              scrambles=2, atol=1e-4, rtol=.005, min_ess=128):
    if scrambles < 2 or len(powers) < 2:
        raise ValueError('At least two scrambles and two increasing powers are required.')
    if list(powers) != sorted(set(powers)):
        raise ValueError('powers must be strictly increasing')
    comps = posterior_components(A,y,sigma,prior,s)
    previous = None
    converged = False
    for power in powers:
        estimates, weights_all, ess_all = [], [], []
        for r in range(scrambles):
            u = qmc.Sobol(3, scramble=True, seed=seed+104729*r).random_base2(power)
            u = np.clip(u, np.finfo(float).eps, 1-np.finfo(float).eps)
            vals = [gaussian_box(mu,cov,u) for mu,cov,_ in comps]
            logs = np.array([v[1]+c[2] for v,c in zip(vals,comps)])
            weights = np.exp(logs-logsumexp(logs))
            estimates.append(weights @ np.array([v[0] for v in vals]))
            weights_all.append(weights)
            ess_all.append(min(v[2] for v,w in zip(vals,weights) if w > 1e-8))
        estimate = np.mean(estimates,axis=0)
        weights = np.mean(weights_all,axis=0)
        spread = np.max(np.abs(np.asarray(estimates)-estimate))
        weight_spread = np.max(np.abs(np.asarray(weights_all)-weights))
        change = np.inf if previous is None else np.max(np.abs(estimate-previous[0]))
        weight_change = np.inf if previous is None else np.max(np.abs(weights-previous[1]))
        tolerance = atol+rtol*np.max(np.abs(estimate))
        converged = (max(change,spread) <= tolerance and
                     max(weight_change,weight_spread) <= atol+rtol and
                     min(ess_all) >= min_ess)
        if converged:
            break
        previous = estimate,weights
    if not np.all(np.isfinite(estimate)):
        raise FloatingPointError('Nonfinite posterior estimate')
    return estimate, dict(converged=converged, quad_change=max(change,spread),
                          ess=min(ess_all), power=power)


