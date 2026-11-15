"""Architecture screen; original modules remain unchanged for checkpoint compatibility."""
from dataclasses import dataclass, replace
from torch import nn
from torch.nn import functional as F
from .families import SPECS
from .config import Config as BaseConfig
from .model import Block, RMSNorm, Transformer as BaseTransformer
from .runs import SEEDS

@dataclass(frozen=True)
class Config(BaseConfig):
    activation: str = "squared_relu"
    loss: str = "all_token"
    variant: str = "baseline"

    def __post_init__(self):
        super().__post_init__()
        if self.activation not in ("squared_relu", "relu") or self.loss not in ("all_token", "mean_output"):
            raise ValueError("Unknown activation/loss")

def baseline(seed):
    return BaseConfig("forced_linear", "parameter", M=2**19, n=128, seed=seed, data_seed=seed)

def configurations(seed):
    if seed not in SEEDS:
        raise ValueError("Unsupported seed")
    cfg = Config(**baseline(seed).to_dict())
    return [
        replace(cfg, variant="mean_output", loss="mean_output"),
        replace(cfg, variant="relu", activation="relu"),
        *[replace(cfg, variant=f"heads_{h}", heads=h) for h in (1, 4, 8)],
        *[replace(cfg, variant=f"depth_{d}", depth=d) for d in (8, 32)],
        replace(cfg, variant="width_32", width=32, ff_width=128),
        replace(cfg, variant="width_128", width=128, ff_width=512),
    ]

class ReLUBlock(Block):
    def forward(self, x):
        x = x + self.dropout(self.fc2(F.relu(self.fc1(self.ff_norm(x)))))
        return x + self.dropout(self.attn(self.attn_norm(x)))

class Transformer(BaseTransformer):
    def __init__(self, cfg):
        nn.Module.__init__(self)
        spec = SPECS[cfg.family]
        m, p = len(spec["scale"]), len(spec["low"])
        din, dout = 1+m+(cfg.task == "prediction"), p if cfg.task == "parameter" else m
        self.input = nn.Linear(din, cfg.width, bias=False)
        block_type = ReLUBlock if cfg.activation == "relu" else Block
        self.blocks = nn.ModuleList([block_type(cfg) for _ in range(cfg.depth)])
        self.norm = RMSNorm(cfg.width)
        self.output = nn.Linear(cfg.width, dout, bias=False)

def loss_fn(outputs, targets, cfg):
    if cfg.loss == "mean_output":
        return (outputs.mean(dim=1)-targets).square().mean()
    return (outputs-targets[:, None, :]).square().mean()

