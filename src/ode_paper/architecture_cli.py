"""Separate architecture screen; both seeds are selected by default."""
import argparse
import csv
import json
from pathlib import Path
from .architecture import baseline, configurations
from .runs import SEEDS
from . import train as base_train, evaluate as base_eval
from . import architecture_train as training, architecture_evaluate as evaluation
from .utils import file_hash
from .config import INFERENCE_NS, fingerprint
from .test_dispersion import test_dispersion, shade_test_ci

def entries(seed):
    return [("baseline", baseline(seed), base_train, base_eval)] + [
        (c.variant, c, training, evaluation) for c in configurations(seed)]

def plot(roots, seeds, ns=INFERENCE_NS):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = []
    output = roots[seeds[-1]]/"architecture_screen"/"figures"
    output.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        for name, cfg, trainer, evaluator in entries(seed):
            folder = trainer.run_path(roots[seed], cfg)
            best = folder/"best.pt"
            if not best.exists():
                raise FileNotFoundError(f"Missing checkpoint: {best}")
            digest = file_hash(best)
            manifest = json.loads((folder/"manifest.json").read_text())
            for n in ns:
                identity = evaluator.evaluation_identity(cfg, digest, n)
                marker = folder/"inference"/f"n{n}_{fingerprint(identity)}"/"metrics.json"
                if not marker.exists():
                    raise FileNotFoundError(f"Evaluate first: {marker}")
                record = json.loads(marker.read_text())
                if record["identity"] != identity:
                    raise ValueError(f"Cache mismatch: {marker}")
                rows.append(dict(seed=seed, variant=name, inference_n=n,
                    parameters=manifest["parameters"], best_epoch=record["best_epoch"],
                    **{k: record["metrics"][k] for k in ("all_token_mse", "mean_output_mse")},
                    **test_dispersion(marker.parent, record)))
            history = json.loads((folder/"history.json").read_text())
            train_key = "train_all_token_mse" if name == "baseline" else "train_objective_mse"
            val_key = "validation_all_token_mse" if name == "baseline" else "validation_objective_mse"
            fig, ax = plt.subplots(figsize=(6, 2))
            ax.plot([r["epoch"] for r in history], [r[train_key] for r in history], label="Train")
            ax.plot([r["epoch"] for r in history], [r[val_key] for r in history], label="Validation")
            ax.set(xlabel="Epoch", ylabel="Mean-output MSE" if name == "mean_output" else "All-token MSE",
                   title=f"{name}, seed {seed}", yscale="log")
            ax.legend(); fig.tight_layout()
            fig.savefig(output/f"{name}_seed{seed}_learning.png", dpi=180)
            plt.close(fig)
    with (output/"summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    groups = {
        "readout": ["baseline", "mean_output"],
        "activation": ["baseline", "relu"],
        "heads": ["heads_1", "baseline", "heads_4", "heads_8"],
        "depth": ["depth_8", "baseline", "depth_32"],
        "width": ["width_32", "baseline", "width_128"],
    }
    for group, names in (groups.items() if 128 in ns else []):
        fig, axes = plt.subplots(1, 2, figsize=(10, 2))
        for ax, metric, label in zip(axes, ("all_token_mse", "mean_output_mse"),
                                     ("All-token test MSE", "Mean-output test MSE")):
            bands = []
            for seed in seeds:
                selected = {r["variant"]: r for r in rows if r["seed"] == seed and r["inference_n"] == 128}
                means = [selected[n][metric] for n in names]
                line, = ax.plot(range(len(names)), means, "o-", label=str(seed))
                bands.append((range(len(names)), means,
                              [selected[n][metric+"_ci95_half_width"] for n in names], line.get_color()))
            ax.set_xticks(range(len(names)), names, rotation=20)
            ax.set(ylabel=label, yscale="log", title=f"{group}, inference n=128")
            shade_test_ci(ax, bands)
            ax.legend(title="Seed")
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(output/f"{group}.{ext}", dpi=180)
        plt.close(fig)
    # Context scaling: one figure per architecture comparison, both metrics.
    for group, names in groups.items():
        fig, axes = plt.subplots(1, 2, figsize=(13, 2.5))
        for ax, metric, label in zip(axes, ("all_token_mse", "mean_output_mse"),
                                     ("All-token test MSE", "Mean-output test MSE")):
            bands = []
            for variant_index, name in enumerate(names):
                for seed_index, seed in enumerate(seeds):
                    selected = sorted([r for r in rows if r["seed"] == seed and r["variant"] == name],
                                      key=lambda r: r["inference_n"])
                    x = [r["inference_n"] for r in selected]
                    means = [r[metric] for r in selected]
                    color = f"C{variant_index}"
                    ax.loglog(x, means, marker="o", color=color,
                              linestyle=("-", "--")[seed_index % 2],
                              label=f"{name}, seed={seed}")
                    bands.append((x, means, [r[metric+"_ci95_half_width"] for r in selected], color))
            ax.set(xlabel="Inference observations n", ylabel=label,
                   title=f"{group}: training n=128, M=524,288")
            shade_test_ci(ax, bands)
            ax.grid(True, which="both", alpha=.2)
            ax.legend(fontsize=7)
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(output/f"{group}_inference_scaling.{ext}", dpi=180)
        plt.close(fig)
    print(f"Saved screen plots and summary: {output}")

def main():
    p = argparse.ArgumentParser(description="18 additional architecture runs, both seeds; no baseline retraining.")
    p.add_argument("command", choices=("list", "train", "evaluate", "plot"))
    p.add_argument("--root-first", type=Path, default=Path("/export/abdullah/ode-results/paper-v1"))
    p.add_argument("--root-second", type=Path, default=Path("/export/abdullah/ode-results/paper-seed-20260907"))
    p.add_argument("--seeds", nargs="+", type=int, choices=SEEDS, default=list(SEEDS))
    p.add_argument("--ns", nargs="+", type=int, default=list(INFERENCE_NS),
                   help="Inference sizes for evaluation/plotting (default: 2 through 512, powers of two)")
    p.add_argument("--device", choices=("cuda", "mps", "cpu"), default="cuda")
    p.add_argument("--eval-batch-size", type=int, default=256)
    p.add_argument("--token-budget", type=int, default=16384)
    args = p.parse_args()
    roots = dict(zip(SEEDS, (args.root_first, args.root_second)))
    if args.root_first.resolve() == args.root_second.resolve():
        p.error("Keep seed results in separate folders")
    if len(set(args.seeds)) != len(args.seeds):
        p.error("Seeds must not repeat")
    if any(n < 1 for n in args.ns) or len(set(args.ns)) != len(args.ns):
        p.error("Inference sizes must be positive and unique")
    args.ns = sorted(args.ns)
    if args.eval_batch_size <= 0 or args.token_budget <= 0:
        p.error("Evaluation batch/token budget must be positive")
    if args.command == "plot":
        plot(roots, args.seeds, args.ns)
        return
    if args.command == "evaluate":
        # Fail early if an existing baseline is missing; never train it implicitly.
        for seed in args.seeds:
            if not (base_train.run_path(roots[seed], baseline(seed))/"last.pt").exists():
                raise FileNotFoundError(f"Missing existing baseline for seed {seed} in {roots[seed]}")
    for seed in args.seeds:
        selected = entries(seed) if args.command == "evaluate" else entries(seed)[1:]
        for index, (name, cfg, trainer, evaluator) in enumerate(selected, 1):
            if args.command == "list":
                print(json.dumps(dict(root=str(roots[seed]), **cfg.to_dict())))
            elif args.command == "train":
                print(f"Architecture seed {seed}, configuration {index}/9: {name}", flush=True)
                trainer.train(roots[seed], cfg, args.device)
            else:
                print(f"Evaluating seed {seed}, {index}/{len(selected)}: {name}; n={args.ns}", flush=True)
                evaluator.evaluate(roots[seed], cfg, args.device, ns=args.ns,
                    token_budget=args.token_budget, max_batch=args.eval_batch_size)

if __name__ == "__main__":
    main()

