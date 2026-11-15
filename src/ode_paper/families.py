"""Exact trajectories in physical units; public time arguments are normalized t/4."""
import numpy as np

SPECS = {
    "forced_linear": dict(names=["x0", "a", "b"], low=[-.8]*3, high=[.8]*3,
                          scale=[0.6532414253078556]),
    "logistic": dict(names=["x0", "r", "K"], low=[.1, .6, .8], high=[.3, 1.8, 1.6], scale=[0.7160787998680879]),
    "birth_death": dict(names=["birth", "death", "competition"], low=[.8, .1, .2], high=[1.8, .6, .8], scale=[1.0066636624210321]),
    "stuart_landau": dict(names=["x0", "y0", "a", "omega"], low=[.4, .4, .8, 2.], high=[.6, .6, 1.2, 3.], scale=[0.6564067052579569, 0.6798484425369343]),
}


def bounds(family):
    s = SPECS[family]
    return np.array(s["low"], dtype=float), np.array(s["high"], dtype=float)


def encode(theta, family):
    lo, hi = bounds(family)
    return 1.6*(theta-(lo+hi)/2)/(hi-lo)


def decode(z, family):
    lo, hi = bounds(family)
    return (lo+hi)/2 + z*(hi-lo)/1.6


def solution(theta, times, family):
    """theta: (batch,p), normalized times: (batch,n); return physical (batch,n,m)."""
    theta = np.asarray(theta, dtype=np.float64)
    t = 4*np.asarray(times, dtype=np.float64)
    a, b, c = [theta[:, j, None] for j in range(3)]
    if family == "forced_linear":
        e = np.exp(-t)
        out = e*a + (-np.expm1(-t))*b + (t+np.expm1(-t))*c
    elif family == "logistic":
        out = c/(1+(c/a-1)*np.exp(-b*t))
    elif family == "birth_death":
        r = a-b
        # Valid also at r=0 by continuous extension, useful for baseline solvers.
        phi = np.empty_like(r*t)
        np.divide(-np.expm1(-r*t), r, out=phi, where=np.broadcast_to(r != 0, phi.shape))
        phi = np.where(r == 0, t, phi)
        out = .2/(np.exp(-r*t)+.2*c*phi)
    elif family == "stuart_landau":
        omega = theta[:, 3, None]
        r0sq = a*a+b*b
        # a,b are initial coordinates; c is the radial growth parameter.
        factor = 1/np.sqrt(np.exp(-2*c*t)+(r0sq/c)*(-np.expm1(-2*c*t)))
        co, si = np.cos(omega*t), np.sin(omega*t)
        return np.stack((factor*(a*co-b*si), factor*(a*si+b*co)), axis=-1)
    else:
        raise ValueError(family)
    return out[..., None]
