"""Equal-weight averages across fixed trained seeds; test-bank uncertainty only."""
import numpy as np


def seed_average(errors_by_seed, *, bands=True, resamples=5000):
    """Bootstrap examples independently within each seed, then average means.

    Does not ensemble predictions or resample training seeds. Each input array
    contains one loss per independent test trajectory for one trained model.
    """
    items = sorted(errors_by_seed.items())
    if len(items) < 2:
        raise ValueError('Expected at least two trained seeds')
    arrays = [np.asarray(a, dtype=float) for _, a in items]
    if any(a.ndim != 1 or len(a) < 2 or not np.isfinite(a).all() or (a < 0).any() for a in arrays):
        raise ValueError('Expected finite nonnegative per-trajectory errors')
    mean = float(np.mean([a.mean() for a in arrays]))
    if not bands:
        return mean, mean, mean
    draws = np.zeros(resamples)
    for (seed, _), a in zip(items, arrays):
        rng = np.random.default_rng(seed)
        for start in range(0, resamples, 250):
            stop = min(start + 250, resamples)
            indices = rng.integers(len(a), size=(stop-start, len(a)))
            draws[start:stop] += a[indices].mean(axis=1)/len(items)
    low, high = np.quantile(draws, [.025, .975])
    return mean, float(low), float(high)


def record_curve(records, method, ns, seeds):
    values = []
    for n in ns:
        errors = {}
        for seed in seeds:
            matches = [r for r in records if r['method']==method and r['seed']==seed and r['n']==n]
            if len(matches) != 1:
                raise ValueError(f'Missing/duplicate result: {method}, {seed}, {n}')
            errors[seed] = matches[0]['errors']
        values.append(seed_average(errors))
    return np.asarray(values).T
