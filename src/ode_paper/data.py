"""Index-addressed data: prefixes are identical across M, n, chunks, and tasks."""
import json
import os
from pathlib import Path
import numpy as np
from .config import fingerprint
from .families import SPECS, bounds, encode, solution
from .utils import lock, source_hash, write_json

SPLITS = {"train": 1001, "validation": 2002, "test": 3003}


def uniform(rows, cols, seed, stream):
    """Counter-based SplitMix64 transform; independent coordinates/streams, no mutable RNG.

    Index-addressed generation ensures n=2 is exactly the first two draws at n=512,
    including observation noise. Values lie strictly inside (0,1).
    """
    r = np.asarray(rows, dtype=np.uint64)[:, None]
    c = np.asarray(cols, dtype=np.uint64)[None, :]
    with np.errstate(over="ignore"):
        z = (r*np.uint64(0xD2B74407B1CE6E93) + c*np.uint64(0xCA5A826395121157)
             + np.uint64(seed) + np.uint64(stream)*np.uint64(0x9E3779B97F4A7C15))
        z += np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> 30))*np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> 27))*np.uint64(0x94D049BB133111EB)
        z ^= z >> 31
    return ((z >> 11).astype(np.float64)+0.5)*(1.0/2**53)


def generate(cfg, split, start, count, n=None):
    n = cfg.n if n is None else n
    ids = np.arange(start, start+count, dtype=np.uint64)
    base = SPLITS[split]
    spec = SPECS[cfg.family]
    lo, hi = bounds(cfg.family)
    theta = lo+(hi-lo)*uniform(ids, np.arange(len(lo)), cfg.data_seed, base)
    # Deliberately unsorted: column positions are sample indices, not time order.
    times = .8*uniform(ids, np.arange(n), cfg.data_seed, base+1)
    tau = uniform(ids, [0], cfg.data_seed, base+2)
    m = len(spec["scale"])
    u = uniform(ids, np.arange(n*m), cfg.data_seed, base+3)
    v = uniform(ids, np.arange(n*m), cfg.data_seed, base+4)
    noise = (np.sqrt(-2*np.log(u))*np.cos(2*np.pi*v)).reshape(count, n, m)
    scale = np.asarray(spec["scale"])
    states = solution(theta, times, cfg.family)/scale + cfg.q*noise
    points = np.concatenate((times[..., None], states), axis=-1)
    if cfg.task == "prediction":
        points = np.concatenate((points, np.broadcast_to(tau[:, :, None], (count, n, 1))), axis=-1)
        targets = solution(theta, tau, cfg.family)[:, 0, :]/scale
    else:
        targets = encode(theta, cfg.family)
    if not (np.isfinite(points).all() and np.isfinite(targets).all()):
        raise FloatingPointError("Non-finite generated data")
    return dict(points=points.astype(np.float32), targets=targets.astype(np.float32),
                theta=theta, tau=tau[:, 0])


def bank(root, cfg, split, chunk_size=2048):
    count = {"train": cfg.M, "validation": cfg.validation_size, "test": cfg.test_size}[split]
    spec = dict(family=cfg.family, task=cfg.task, n=cfg.n, count=count, split=split,
                data_seed=cfg.data_seed, q=cfg.q,
                generator=source_hash(["data.py", "families.py"]))
    folder = Path(root)/"data"/fingerprint(spec)
    with lock(folder/"build.lock"):
        marker = folder/"complete.json"
        if marker.exists():
            if json.loads(marker.read_text()) != spec:
                raise ValueError("Data cache mismatch")
        else:
            sample = generate(cfg, split, 0, 1)
            arrays = {}
            for key in ("points", "targets"):
                arrays[key] = np.lib.format.open_memmap(folder/f"{key}.building.npy", mode="w+",
                    dtype=np.float32, shape=(count, *sample[key].shape[1:]))
            for start in range(0, count, chunk_size):
                values = generate(cfg, split, start, min(chunk_size, count-start))
                for key, array in arrays.items():
                    array[start:start+len(values[key])] = values[key]
            for key in tuple(arrays):
                arrays[key].flush()
                del arrays[key]
                os.replace(folder/f"{key}.building.npy", folder/f"{key}.npy")
            write_json(marker, spec)
    return {key: np.load(folder/f"{key}.npy", mmap_mode="r") for key in ("points", "targets")}
