from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import tempfile
import torch


def source_hash(names=None):
    base = Path(__file__).parent
    paths = [base / n for n in names] if names else sorted(base.glob("*.py"))
    h = hashlib.sha256()
    for path in paths:
        h.update(path.name.encode()); h.update(path.read_bytes())
    return h.hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


@contextmanager
def lock(path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another process holds {path}") from exc
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def atomic(path, writer):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name+".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            writer(f)
            f.flush(); os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def write_json(path, obj):
    atomic(path, lambda f: f.write(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False).encode()))


def save_torch(path, obj):
    atomic(path, lambda f: torch.save(obj, f))


def environment(device):
    import numpy as np
    return dict(python=platform.python_version(), torch=str(torch.__version__), numpy=np.__version__,
                device=str(device), gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                cuda=torch.version.cuda, platform=platform.platform(), precision="float32; TF32 disabled")


def setup_device(name):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; no silent CPU fallback")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    return device
