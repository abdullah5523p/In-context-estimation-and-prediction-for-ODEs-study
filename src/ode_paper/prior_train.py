import json
import math
from pathlib import Path
import random
import time
import numpy as np
import torch
from .config import fingerprint
from .prior_data import bank
from .model import Transformer, all_token_loss
from .utils import environment, lock, save_torch, setup_device, source_hash, write_json

CORE = ["config.py", "families.py", "data.py", "model.py", "utils.py", "prior_config.py", "prior_data.py", "prior_train.py"]


def run_spec(cfg):
    return dict(config=cfg.to_dict(), training_source=source_hash(CORE))


def run_path(root, cfg):
    return Path(root)/"runs"/cfg.prior/f"s{cfg.s:g}_q{cfg.q:g}"/f"n{cfg.n}_M{cfg.M}_{fingerprint(run_spec(cfg))}"


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def rng_state(device, generator):
    out = dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
               shuffle=generator.get_state())
    if device.type == "cuda":
        out["cuda"] = torch.cuda.get_rng_state_all()
    if device.type == "mps":
        out["mps"] = torch.mps.get_rng_state()
    return out


def restore_rng(state, device, generator):
    random.setstate(state["python"]); np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"]); generator.set_state(state["shuffle"])
    if device.type == "cuda":
        torch.cuda.set_rng_state_all(state["cuda"])
    if device.type == "mps":
        torch.mps.set_rng_state(state["mps"])


def tensor_batch(bank_data, indices, device):
    return [torch.from_numpy(np.array(bank_data[k][indices], copy=True)).to(device)
            for k in ("points", "targets")]


@torch.inference_mode()
def validation_loss(model, data, device, batch_size):
    model.eval()
    total = 0.
    for start in range(0, len(data["targets"]), batch_size):
        x, y = tensor_batch(data, slice(start, start+batch_size), device)
        loss = all_token_loss(model(x), y)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite validation loss")
        total += loss.item()*len(x)
    return total/len(data["targets"])


def train(root, cfg, device_name="cuda", stop_after_epoch=None):
    """Resume only at completed epoch boundaries; fixed original 30-epoch schedule."""
    device = setup_device(device_name)
    folder = run_path(root, cfg)
    spec = run_spec(cfg)
    with lock(folder/"training.lock"):
        last_path = folder/"last.pt"
        if last_path.exists():
            saved = torch.load(last_path, map_location="cpu", weights_only=False)
            if saved["spec"] != spec:
                raise ValueError("Checkpoint does not match this configuration/source")
            # last.pt is the transaction authority; repair best/history after interrupted writes.
            best_path = folder/"best.pt"
            old_best = torch.load(best_path, map_location="cpu", weights_only=False) if best_path.exists() else None
            if old_best is None or old_best["epoch"] != saved["best_checkpoint"]["epoch"] or old_best["spec"] != spec:
                save_torch(best_path, saved["best_checkpoint"])
            history_path = folder/"history.json"
            if not history_path.exists() or json.loads(history_path.read_text()) != saved["history"]:
                write_json(history_path, saved["history"])
            if saved["epoch"] == cfg.epochs:
                print(f"Loaded completed {folder} (best epoch {saved['best_checkpoint']['epoch']})", flush=True)
                return folder
            current_env = environment(device)
            for key in ("device", "torch", "numpy", "gpu"):
                if saved["environment"][key] != current_env[key]:
                    raise ValueError(f"Resume environment changed: {key}. Use original environment.")
        else:
            saved = None
        seed_all(cfg.seed)
        model = Transformer(cfg).to(device)
        train_bank, val_bank = bank(root, cfg, "train"), bank(root, cfg, "validation")
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        batches = math.ceil(cfg.M/cfg.batch_size)
        total_steps = cfg.epochs*batches
        warmup = max(1, int(cfg.warmup_fraction*total_steps))
        def multiplier(step):
            if step < warmup:
                return (step+1)/warmup
            progress = min(1., (step-warmup)/max(1, total_steps-warmup))
            return .5*(1+math.cos(math.pi*progress))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
        generator = torch.Generator().manual_seed(cfg.seed+1)
        history, start_epoch, best = [], 0, None
        env = environment(device)
        source_dir = Path(__file__).parent
        if not (folder/"source_snapshot.json").exists():
            write_json(folder/"source_snapshot.json", {p.name: p.read_text() for p in sorted(source_dir.glob("*.py"))})
        write_json(folder/"manifest.json", dict(spec=spec, environment=env,
            parameters=sum(p.numel() for p in model.parameters()), updates=total_steps,
            checkpoint_policy=f"minimum validation all-token MSE over epochs 1..{cfg.epochs}; last also saved"))
        if saved is not None:
            model.load_state_dict(saved["model"])
            optimizer.load_state_dict(saved["optimizer"])
            scheduler.load_state_dict(saved["scheduler"])
            history, start_epoch, best = saved["history"], saved["epoch"], saved["best_checkpoint"]
            restore_rng(saved["rng"], device, generator)
            print(f"Resuming after epoch {start_epoch}: {folder}", flush=True)
        end = cfg.epochs if stop_after_epoch is None else min(cfg.epochs, stop_after_epoch)
        for epoch in range(start_epoch, end):
            begin = time.perf_counter()
            last_progress = begin
            model.train()
            order = torch.randperm(cfg.M, generator=generator).numpy()
            total, grad_sum, grad_max, clipped, updates = 0., 0., 0., 0, 0
            for offset in range(0, cfg.M, cfg.batch_size):
                x, y = tensor_batch(train_bank, order[offset:offset+cfg.batch_size], device)
                if not (torch.isfinite(x).all() and torch.isfinite(y).all()):
                    raise FloatingPointError("Non-finite training input/target")
                optimizer.zero_grad(set_to_none=True)
                pred = model(x)
                loss = all_token_loss(pred, y)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite training output/loss")
                loss.backward()
                grad = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip, error_if_nonfinite=True)
                optimizer.step(); scheduler.step()
                if not torch.stack([torch.isfinite(p).all() for p in model.parameters()]).all():
                    raise FloatingPointError("Non-finite weights after optimizer update")
                g = grad.item(); grad_sum += g; grad_max = max(grad_max, g)
                clipped += int(g > cfg.gradient_clip); updates += 1
                total += loss.item()*len(x)
                if time.perf_counter()-last_progress >= 60:
                    write_json(folder/"progress.json", dict(epoch=epoch+1, completed_batches=updates,
                        batches_per_epoch=batches, elapsed_seconds=time.perf_counter()-begin))
                    print(f"  epoch {epoch+1}: batch {updates}/{batches}, {(time.perf_counter()-begin)/60:.1f} min", flush=True)
                    last_progress = time.perf_counter()
            val = validation_loss(model, val_bank, device, cfg.batch_size)
            row = dict(epoch=epoch+1, train_all_token_mse=total/cfg.M, validation_all_token_mse=val,
                       grad_norm_mean=grad_sum/updates, grad_norm_max=grad_max,
                       clipping_fraction=clipped/updates, optimizer_steps=(epoch+1)*batches,
                       next_learning_rate=optimizer.param_groups[0]["lr"], seconds=time.perf_counter()-begin)
            history.append(row)
            weights = cpu_state(model)
            if best is None or val < best["validation_all_token_mse"]:
                best = dict(spec=spec, model=weights, epoch=epoch+1, validation_all_token_mse=val)
            checkpoint = dict(spec=spec, model=weights, epoch=epoch+1, optimizer=optimizer.state_dict(),
                              scheduler=scheduler.state_dict(), rng=rng_state(device, generator),
                              history=history, best_checkpoint=best, environment=env)
            # One atomic authoritative recovery file precedes derived best/history files.
            save_torch(last_path, checkpoint)
            save_torch(folder/"best.pt", best)
            write_json(folder/"history.json", history)
            print(f"{cfg.family}/{cfg.task} n={cfg.n} M={cfg.M:,} epoch={epoch+1:02}/{cfg.epochs} "
                  f"train={total/cfg.M:.6g} val={val:.6g} clipped={clipped/updates:.1%} "
                  f"seconds={row['seconds']:.1f}", flush=True)
        return folder

