from dataclasses import asdict, dataclass
import hashlib
import json

FAMILIES = ("forced_linear", "logistic", "birth_death", "stuart_landau")
TASKS = ("parameter", "prediction")
TRAIN_NS = (2, 8, 32, 128)
SIZES = tuple(2**k for k in (11, 13, 15, 17, 19))
INFERENCE_NS = tuple(2**k for k in range(1, 10))
PROTOCOL = "unsorted_fullwidth_squared_relu_v1"


@dataclass(frozen=True)
class Config:
    family: str
    task: str
    M: int
    n: int
    seed: int = 20260906
    data_seed: int = 20260906
    epochs: int = 30
    batch_size: int = 256
    validation_size: int = 4096
    test_size: int = 8192
    depth: int = 16
    width: int = 64
    ff_width: int = 256
    heads: int = 2
    dropout: float = 0.1
    q: float = 0.02
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    warmup_fraction: float = 0.05
    protocol: str = PROTOCOL

    def __post_init__(self):
        if self.family not in FAMILIES or self.task not in TASKS:
            raise ValueError("Unknown family/task")
        for name in ("M", "n", "epochs", "batch_size", "validation_size", "test_size", "depth", "width", "ff_width", "heads"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not (0 <= self.dropout < 1 and self.q >= 0 and 0 <= self.warmup_fraction < 1):
            raise ValueError("Invalid dropout/noise/warmup")

    def to_dict(self):
        return asdict(self)


def fingerprint(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:20]


def grid():
    return [Config(f, t, M, n) for f in FAMILIES for t in TASKS
            for n in (TRAIN_NS if f == "forced_linear" else (8,))
            for M in (SIZES if f == "forced_linear" else (2**19,))]
