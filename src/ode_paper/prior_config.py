"""Isolated concentrated-prior sweep; original configurations are unchanged."""
from dataclasses import dataclass
from .config import Config

SEEDS = (20260906, 20260907)
QS = (.05, .10, .20)
SCALES = (.2, .1, .05)
PRIORS = ("one_mode", "two_mode")

@dataclass(frozen=True)
class PriorConfig(Config):
    prior: str = "one_mode"
    s: float = .2

    def __post_init__(self):
        super().__post_init__()
        if self.family != "forced_linear" or self.task != "parameter":
            raise ValueError("This sweep is forced-linear parameter estimation only.")
        if self.prior not in PRIORS or self.s <= 0:
            raise ValueError("Invalid prior or pre-truncation standard deviation.")

def configurations(seeds=SEEDS):
    return [PriorConfig(family="forced_linear", task="parameter",
                        M=2**19, n=8, seed=seed, data_seed=seed,
                        prior=prior, s=s, q=q)
            for seed in seeds for prior in PRIORS for s in SCALES for q in QS]

