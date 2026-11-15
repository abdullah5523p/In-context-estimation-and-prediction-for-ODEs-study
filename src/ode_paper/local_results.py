"""Read exported evaluation bundles locally. Requires NumPy, no checkpoints/GPU."""
import hashlib
import json
from pathlib import Path
import numpy as np
from .test_dispersion import test_dispersion

def checked_path(root, relative):
    path = (root/relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Path outside bundle: {relative}")
    return path

def load_results(directory, verify=True):
    rows, seen = [], set()
    catalogs = sorted(Path(directory).glob("seed_*/catalog.json"))
    if not catalogs:
        raise FileNotFoundError("No exported evaluation bundles found. Run the download/extraction cells first.")
    for path in catalogs:
        catalog = json.loads(path.read_text())
        root = path.parent
        if catalog["format_version"] != 1:
            raise ValueError(f"Unsupported bundle version: {path}")
        if verify:
            for name, digest in catalog["sha256"].items():
                actual = hashlib.sha256(checked_path(root, name).read_bytes()).hexdigest()
                if actual != digest:
                    raise ValueError(f"Incomplete/corrupted download: {name}")
        for entry in catalog["evaluations"]:
            cfg = entry["config"]
            if cfg["seed"] != catalog["seed"]:
                raise ValueError(f"Seed mismatch in {path}")
            cache = checked_path(root, entry["cache"])
            record = json.loads((cache/"metrics.json").read_text())
            if record["identity"]["config"] != cfg or record["identity"]["n"] != entry["inference_n"]:
                raise ValueError(f"Catalog/cache mismatch: {cache}")
            key = (entry["group"], cfg["seed"], cfg["family"], cfg["task"],
                   cfg["M"], cfg["n"], entry["variant"], entry["inference_n"])
            if key in seen:
                raise ValueError(f"Duplicate evaluation: {key}")
            seen.add(key)
            rows.append(dict(group=entry["group"], variant=entry["variant"],
                family=cfg["family"], task=cfg["task"], seed=cfg["seed"],
                data_seed=cfg["data_seed"], M=cfg["M"], train_n=cfg["n"],
                inference_n=entry["inference_n"], best_epoch=record["best_epoch"],
                cache=cache, run=checked_path(root, entry["run"]),
                **record["metrics"], **test_dispersion(cache, record)))
    return rows

def select(rows, **filters):
    return [r for r in rows if all(r[k] == v for k, v in filters.items())]

def example_arrays(row):
    with np.load(row["cache"]/"per_example.npz", allow_pickle=False) as saved:
        return {k: saved[k].copy() for k in saved.files}

def example_losses(row, metric="all_token_mse"):
    arrays = example_arrays(row)
    if metric == "all_token_mse":
        return arrays["all_token_coordinate_mse"].astype(float).mean(axis=1)
    if metric == "mean_output_mse":
        return ((arrays["mean_prediction"].astype(float)-arrays["targets"])**2).mean(axis=1)
    raise ValueError("Choose all_token_mse or mean_output_mse")
