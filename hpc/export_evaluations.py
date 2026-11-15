"""Export current evaluation caches and metadata, without weights or data banks."""
import argparse
import io
import json
from pathlib import Path
import sys
import tarfile
import os

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from ode_paper.runs import SEEDS, configurations
from ode_paper.architecture import configurations as screen_configurations
from ode_paper import train, evaluate, architecture_train, architecture_evaluate
from ode_paper.config import INFERENCE_NS, fingerprint
from ode_paper.utils import file_hash

def export(root, seed, output):
    root, output = Path(root), Path(output)
    rows, files, missing = [], {}, []
    groups = [
        ("main_experiment", configurations(seed), train, evaluate),
        ("architecture_screen", screen_configurations(seed), architecture_train, architecture_evaluate),
    ]
    for group, configs, trainer, evaluator in groups:
        for cfg in configs:
            folder = trainer.run_path(root, cfg)
            required = [folder/name for name in ("best.pt", "manifest.json", "history.json")]
            if not all(p.exists() for p in required):
                missing.append(str(folder)); continue
            manifest = json.loads((folder/"manifest.json").read_text())
            if manifest["spec"] != trainer.run_spec(cfg):
                raise ValueError(f"Manifest identity mismatch: {folder}")
            history = json.loads((folder/"history.json").read_text())
            if not history or history[-1]["epoch"] != cfg.epochs:
                raise ValueError(f"Training is not complete: {folder}")
            digest = file_hash(folder/"best.pt")
            for name in ("manifest.json", "history.json"):
                p = folder/name
                files[p.relative_to(root).as_posix()] = p
            for n in INFERENCE_NS:
                identity = evaluator.evaluation_identity(cfg, digest, n)
                cache = folder/"inference"/f"n{n}_{fingerprint(identity)}"
                marker, arrays = cache/"metrics.json", cache/"per_example.npz"
                if not marker.exists() or not arrays.exists():
                    missing.append(str(cache)); continue
                record = json.loads(marker.read_text())
                if record["identity"] != identity:
                    raise ValueError(f"Cache identity mismatch: {marker}")
                for p in (marker, arrays):
                    files[p.relative_to(root).as_posix()] = p
                rows.append(dict(group=group, config=cfg.to_dict(), inference_n=n,
                    variant=getattr(cfg, "variant", "baseline"),
                    run=folder.relative_to(root).as_posix(),
                    cache=cache.relative_to(root).as_posix()))
    if missing:
        raise FileNotFoundError("Finish evaluation before export; missing:\n"+"\n".join(missing))
    if len(rows) != 61*len(INFERENCE_NS):
        raise ValueError(f"Expected 549 evaluations for seed {seed}, found {len(rows)}")
    output.mkdir(parents=True, exist_ok=True)
    target = output/f"evaluation_seed_{seed}.tar.gz"
    temporary = output/f".evaluation_seed_{seed}.{os.getpid()}.tar.gz"
    prefix = f"seed_{seed}"
    hashes = {}
    try:
        with tarfile.open(temporary, "w:gz") as tar:
            for name, path in sorted(files.items()):
                hashes[name] = file_hash(path)
                tar.add(path, arcname=f"{prefix}/{name}", recursive=False)
            catalog = dict(format_version=1, seed=seed, evaluations=rows, sha256=hashes)
            payload = json.dumps(catalog, indent=2).encode()
            info = tarfile.TarInfo(f"{prefix}/catalog.json")
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Saved {target}: {len(rows)} evaluations, {target.stat().st_size/1024**2:.1f} MiB", flush=True)
    return target

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root-first", type=Path, default=Path("/export/abdullah/ode-results/paper-v1"))
    p.add_argument("--root-second", type=Path, default=Path("/export/abdullah/ode-results/paper-seed-20260907"))
    p.add_argument("--output", type=Path, default=Path("/export/abdullah/ode-results/evaluation-downloads"))
    args = p.parse_args()
    for seed, root in zip(SEEDS, (args.root_first, args.root_second)):
        export(root, seed, args.output)

if __name__ == "__main__":
    main()
