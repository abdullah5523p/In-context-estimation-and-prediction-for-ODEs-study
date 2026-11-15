"""Joint parameter/state supervision, using unchanged prediction data and blocks."""
from dataclasses import dataclass, replace
import numpy as np
from torch import nn
from .config import Config as BaseConfig, FAMILIES
from .runs import SEEDS
from . import data
from .families import SPECS, encode
from .model import Transformer as BaseTransformer, Block, RMSNorm

@dataclass(frozen=True)
class Config(BaseConfig):
    def __post_init__(self):
        if self.task != "joint":
            raise ValueError("Joint configurations require task='joint'")
        BaseConfig(**{**self.to_dict(), "task": "prediction"})

def prediction_config(cfg):
    return BaseConfig(**{**cfg.to_dict(), "task": "prediction"})

def configurations(seed):
    if seed not in SEEDS: raise ValueError("Unsupported seed")
    return [Config(f,"joint",2**19,n,seed=seed,data_seed=seed)
            for f in FAMILIES for n in (8,128)]

def generate(cfg, split, start, count, n=None):
    b=data.generate(prediction_config(cfg),split,start,count,n=n)
    b["targets"]=np.concatenate((encode(b["theta"],cfg.family).astype(np.float32),b["targets"]),axis=1)
    return b

def bank(root,cfg,split):
    # Existing on-disk prediction banks store points and state targets, not theta.
    # Regenerate theta deterministically from the original stream, without changing banks.
    pc=prediction_config(cfg)
    existing=data.bank(root,pc,split)
    count=len(existing["targets"]); p=len(SPECS[cfg.family]["low"])
    targets=np.empty((count,p+existing["targets"].shape[1]),dtype=np.float32)
    for start in range(0,count,2048):
        stop=min(count,start+2048)
        b=generate(cfg,split,start,stop-start)
        if not np.array_equal(b["targets"][:,p:],existing["targets"][start:stop]):
            raise ValueError("Prediction bank/generator mismatch")
        targets[start:stop]=b["targets"]
    return {"points":existing["points"],"targets":targets}

class Transformer(BaseTransformer):
    def __init__(self,cfg):
        nn.Module.__init__(self)
        m,p=len(SPECS[cfg.family]["scale"]),len(SPECS[cfg.family]["low"])
        self.input=nn.Linear(m+2,cfg.width,bias=False)
        self.blocks=nn.ModuleList([Block(cfg) for _ in range(cfg.depth)])
        self.norm=RMSNorm(cfg.width)
        self.output=nn.Linear(cfg.width,p+m,bias=False)
