"""Aggregate only current protocol/source/checkpoint results; never silently mix runs."""
import csv
import io
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .config import FAMILIES, TASKS, TRAIN_NS, SIZES
from .runs import configurations
from .evaluate import evaluation_identity
from .train import run_path
from .utils import atomic, file_hash
from .test_dispersion import test_dispersion, shade_test_ci


def comparison_specs(rows):
    """Three requested comparisons for one family/task; always use test all-token risk."""
    def curve(label, predicate, x):
        return label, sorted([r for r in rows if predicate(r)], key=lambda r: r[x])
    return [
        dict(name="M_scaling_by_training_n", x="M", xlabel="Training tasks M",
             subtitle="Inference n = training n for each curve",
             curves=[curve(f"train/test n={n}", lambda r, n=n: r["train_n"] == n and r["inference_n"] == n, "M")
                     for n in TRAIN_NS]),
        dict(name="train_n128_inference_scaling_by_M", x="inference_n", xlabel="Inference observations n",
             subtitle="Training n=128; varying M",
             curves=[curve(f"M={M:,}", lambda r, M=M: r["train_n"] == 128 and r["M"] == M, "inference_n")
                     for M in SIZES]),
        dict(name="M524288_inference_scaling_by_training_n", x="inference_n", xlabel="Inference observations n",
             subtitle="M=524,288; varying training n",
             curves=[curve(f"train n={n}", lambda r, n=n: r["M"] == 2**19 and r["train_n"] == n, "inference_n")
                     for n in TRAIN_NS]),
    ]


def plot(root, configs=None, sources=None):
    root = Path(root)
    destination = root/"figures"
    destination.mkdir(parents=True, exist_ok=True)
    rows = []
    if sources is None:
        sources = [(root, configurations() if configs is None else configs)]
    for source_root, cfg in ((r, c) for r, cs in sources for c in cs):
        folder = run_path(source_root, cfg)
        if not (folder/"best.pt").exists():
            continue
        digest = file_hash(folder/"best.pt")
        for path in sorted((folder/"inference").glob("*/metrics.json")):
            record = json.loads(path.read_text())
            n = record["identity"]["n"]
            if record["identity"] != evaluation_identity(cfg, digest, n):
                continue
            row = dict(family=cfg.family, task=cfg.task, seed=cfg.seed, data_seed=cfg.data_seed,
                       M=cfg.M, train_n=cfg.n, inference_n=n,
                       best_epoch=record["best_epoch"], all_token_mse=record["metrics"]["all_token_mse"],
                       physical_all_token_mse=record["metrics"]["physical_all_token_mse"])
            for subset in ("interpolation", "extrapolation"):
                row[subset+"_mse"] = record["metrics"].get(subset, {}).get("all_token_mse")
            row.update(test_dispersion(path.parent, record))
            rows.append(row)
        history_path = folder/"history.json"
        if history_path.exists():
            hist = json.loads(history_path.read_text())
            fig, ax = plt.subplots(figsize=(6, 2))
            for key, label in (("train_all_token_mse", "Train (dropout active)"), ("validation_all_token_mse", "Validation")):
                ax.semilogy([r["epoch"] for r in hist], [r[key] for r in hist], label=label)
            ax.set(xlabel="Epoch", ylabel="Normalized all-token MSE", title=f"{cfg.family}, {cfg.task}, n={cfg.n}, M={cfg.M:,}")
            ax.legend(); fig.tight_layout(); fig.savefig(folder/"learning_curve.png", dpi=160); plt.close(fig)
    if not rows:
        print("No current-grid inference results yet. Learning curves generated where available.")
        return
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    atomic(destination/"inference_summary.csv", lambda f: f.write(buf.getvalue().encode()))
    for family in FAMILIES:
        for task in TASKS:
            selected = [r for r in rows if r["family"] == family and r["task"] == task]
            if not selected:
                continue
            specs = comparison_specs(selected)
            if family != "forced_linear":
                specs = [dict(specs[-1], subtitle="M=524,288; varying training n")]
            for spec in specs:
                if not any(points for _, points in spec["curves"]):
                    print(f"Skipped {family}/{task}/{spec['name']}: required results unavailable")
                    continue
                fig, ax = plt.subplots(figsize=(7, 2.4))
                bands = []
                for curve_index, (label, points) in enumerate(spec["curves"]):
                    if points:
                        for seed_index, seed in enumerate(sorted({r["seed"] for r in rows})):
                            seed_points = [r for r in points if r["seed"] == seed]
                            if seed_points:
                                bands.append(([r[spec["x"]] for r in seed_points],
                                              [r["all_token_mse"] for r in seed_points],
                                              [r["all_token_mse_ci95_half_width"] for r in seed_points],
                                              f"C{curve_index % 10}"))
                                ax.loglog([r[spec["x"]] for r in seed_points],
                                          [r["all_token_mse"] for r in seed_points],
                                          marker="o", color=f"C{curve_index % 10}",
                                          linestyle=("-", "--")[seed_index % 2],
                                          label=f"{label}, seed={seed}" if len(sources) > 1 else label)
                ax.set(xlabel=spec["xlabel"], ylabel="Normalized all-token MSE",
                       title=f"{family}: {task}\n{spec['subtitle']}")
                shade_test_ci(ax, bands)
                ax.grid(True, which="both", alpha=.2); ax.legend(fontsize="small")
                fig.tight_layout()
                for ext in ("pdf", "png"):
                    fig.savefig(destination/f"{family}_{task}_{spec['name']}.{ext}", dpi=180)
                plt.close(fig)
    print(f"Saved current-protocol results to {destination}")
