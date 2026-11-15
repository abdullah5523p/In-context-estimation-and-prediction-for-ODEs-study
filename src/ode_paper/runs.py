"""Complete 52-run grid shared by training, evaluation and plotting.

Keep config.py unchanged: its bytes are part of existing checkpoint identities.
The original 46 configurations retain their indices; the six n=128 runs follow.
"""
from dataclasses import replace
from .config import Config, grid as original_grid

SEEDS = (20260906, 20260907)


def configurations(seed=SEEDS[0]):
    if seed not in SEEDS:
        raise ValueError(f"Choose a seed from {SEEDS}")
    extra = [Config(family, task, M=2**19, n=128)
             for family in ("logistic", "birth_death", "stuart_landau")
             for task in ("parameter", "prediction")]
    return [replace(c, seed=seed, data_seed=seed) for c in original_grid() + extra]
