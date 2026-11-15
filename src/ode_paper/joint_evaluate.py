import json
from pathlib import Path
import numpy as np
import torch
from .config import INFERENCE_NS, fingerprint
from .joint import generate, Transformer
from .families import SPECS, bounds, decode
from .joint_train import run_path, run_spec
from .utils import atomic, environment, file_hash, lock, setup_device, source_hash, write_json


def evaluation_identity(cfg, checkpoint_hash, n):
    return dict(checkpoint_sha256=checkpoint_hash, n=n, count=cfg.test_size,
                config=cfg.to_dict(), source=source_hash(["joint_evaluate.py", "joint.py", "data.py", "families.py", "model.py"]))


def evaluate(root, cfg, device_name="cuda", ns=INFERENCE_NS, token_budget=16384, max_batch=256):
    device = setup_device(device_name)
    folder = run_path(root, cfg)
    with lock(folder/"training.lock"):
        if not (folder/"last.pt").exists():
            raise FileNotFoundError(f"Train first: {folder}")
        last = torch.load(folder/"last.pt", map_location="cpu", weights_only=False)
        if last["spec"] != run_spec(cfg) or last["epoch"] != cfg.epochs:
            raise ValueError("Inference sweep requires the completed, matching training run")
        # Repair best.pt if a process stopped between the recovery and best writes.
        checkpoint = last["best_checkpoint"]
        best_path = folder/"best.pt"
        if not best_path.exists():
            from .utils import save_torch
            save_torch(best_path, checkpoint)
        best = torch.load(best_path, map_location="cpu", weights_only=False)
        if best["epoch"] != checkpoint["epoch"] or best["spec"] != checkpoint["spec"]:
            from .utils import save_torch
            save_torch(best_path, checkpoint)
        digest = file_hash(best_path)
        model = Transformer(cfg).to(device)
        model.load_state_dict(checkpoint["model"]); model.eval()
        for n in ns:
            if n < 1:
                raise ValueError("n must be positive")
            identity = evaluation_identity(cfg, digest, n)
            cache = folder/"inference"/f"n{n}_{fingerprint(identity)}"
            marker = cache/"metrics.json"
            with lock(cache/"evaluation.lock"):
                if marker.exists() and (cache/"per_example.npz").exists():
                    record = json.loads(marker.read_text())
                    if record["identity"] != identity:
                        raise ValueError("Inference cache mismatch")
                    print(f"Loaded inference {cfg.family}/{cfg.task} M={cfg.M} train_n={cfg.n} n={n}", flush=True)
                    continue
                batch = min(max_batch, max(1, token_budget//n))
                pieces = {key: [] for key in ("all_token_coordinate_mse", "mean_prediction", "targets", "theta", "tau")}
                with torch.inference_mode():
                    for start in range(0, cfg.test_size, batch):
                        b = generate(cfg, "test", start, min(batch, cfg.test_size-start), n=n)
                        pred = model(torch.from_numpy(b["points"]).to(device))
                        if not torch.isfinite(pred).all():
                            raise FloatingPointError("Non-finite inference output")
                        target = torch.from_numpy(b["targets"]).to(device)
                        pieces["all_token_coordinate_mse"].append((pred-target[:, None]).square().mean(1).cpu().numpy())
                        pieces["mean_prediction"].append(pred.mean(1).cpu().numpy())
                        for key in ("targets", "theta", "tau"):
                            pieces[key].append(b[key])
                values = {k: np.concatenate(v) for k, v in pieces.items()}
                errors = values["all_token_coordinate_mse"].astype(float)
                avg_errors = (values["mean_prediction"].astype(float)-values["targets"])**2
                p=len(SPECS[cfg.family]["low"])
                metrics={}
                for label,sl in (("parameter",slice(0,p)),("trajectory",slice(p,None)),("combined",slice(None))):
                    per=errors[:,sl].mean(1)
                    mean_per=avg_errors[:,sl].mean(1)
                    values[label+"_all_token_mse"]=per
                    values[label+"_mean_output_mse"]=mean_per
                    metrics[label+"_all_token_mse"]=float(per.mean())
                    metrics[label+"_mean_output_mse"]=float(mean_per.mean())
                metrics["all_token_mse"]=metrics["combined_all_token_mse"]
                metrics["mean_output_mse"]=metrics["combined_mean_output_mse"]
                metrics["coordinate_all_token_mse"]=errors.mean(0).tolist()
                lo,hi=bounds(cfg.family)
                scales=np.r_[(hi-lo)/1.6,SPECS[cfg.family]["scale"]]
                metrics["physical_coordinate_all_token_mse"]=(errors*scales**2).mean(0).tolist()
                for label,mask in (("interpolation",values["tau"]<=.8),("extrapolation",values["tau"]>.8)):
                    metrics[label]={"count":int(mask.sum()),"trajectory_all_token_mse":float(errors[mask,p:].mean()) if mask.any() else None}
                if cfg.family=="birth_death":
                    est=decode(values["mean_prediction"][:,:p],cfg.family); truth=values["theta"]
                    dif=np.stack(((est[:,0]-est[:,1])-(truth[:,0]-truth[:,1]),est[:,2]-truth[:,2]),axis=1)
                    values["parameter_mean_output_physical_combination_coordinate_mse"]=dif**2
                m=errors.shape[1]-p
                assert np.allclose(values["combined_all_token_mse"],
                    (p*values["parameter_all_token_mse"]+m*values["trajectory_all_token_mse"])/(p+m))
                atomic(cache/"per_example.npz", lambda f: np.savez_compressed(f, **values))
                write_json(marker, dict(identity=identity, metrics=metrics, best_epoch=checkpoint["epoch"],
                                        environment=environment(device), evaluation_batch_size=batch))
                print(f"{cfg.family}/{cfg.task} M={cfg.M:,} train_n={cfg.n} n={n}: "
                      f"parameter={metrics['parameter_all_token_mse']:.7g} "
                      f"trajectory={metrics['trajectory_all_token_mse']:.7g} combined={metrics['combined_all_token_mse']:.7g}", flush=True)
    return folder
