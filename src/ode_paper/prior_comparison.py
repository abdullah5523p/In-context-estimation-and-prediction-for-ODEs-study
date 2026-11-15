"""Parameter-only comparisons for the isolated concentrated-prior sweep."""
import hashlib
import inspect
import json
from pathlib import Path
import numpy as np
from .prior_config import PriorConfig
from .prior_data import generate
from .families import SPECS, encode, solution
from .statistical_comparison import fit_mle, fit_sgd, trajectory
from .prediction_comparison import bootstrap_mean_ci
from . import prior_posterior as posterior_module

NS = (2, 4, 8, 16, 32, 64, 128, 256, 512)
METHODS = ("Transformer", "Posterior mean", "Bounded MLE", "Projected SGD")
DEFAULT_QUADRATURE = dict(powers=(10, 12, 14, 16), scrambles=2,
                          atol=1e-4, rtol=.005, min_ess=128)


def digest(value):
    return hashlib.sha256(value).hexdigest()


def cached(path, identity, compute):
    """Atomic, identity-checked caches, independent of plotting choices."""
    serial = json.dumps(identity, sort_keys=True)
    path = Path(path)
    if path.exists():
        with np.load(path, allow_pickle=False) as z:
            if str(z["identity"]) != serial:
                raise ValueError(f"Cache identity mismatch: {path}")
            return {k: z[k].copy() for k in z.files if k != "identity"}
    values = compute()
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp.npz")
    np.savez_compressed(temp, identity=serial, **values)
    temp.replace(path)
    return values


def compare_setting(root, prior, s, q, seed=20260906, ns=NS, count=256,
                    sgd_steps=3000, sgd_batch=8, sgd_lr=.05,
                    quadrature=None):
    root = Path(root)
    quadrature = dict(DEFAULT_QUADRATURE if quadrature is None else quadrature)
    search = root/"evaluation_cache_downloads"/"prior-screen"/f"seed_{seed}"/"runs"/prior/f"s{s:g}_q{q:g}"
    candidates = list(search.glob("*/inference/*/metrics.json"))
    records = []
    fit_source = digest("".join(inspect.getsource(f) for f in
                               (fit_mle, fit_sgd, trajectory)).encode())
    posterior_source = digest(Path(posterior_module.__file__).read_bytes())
    scale = float(SPECS["forced_linear"]["scale"][0])
    for n in ns:
        matches = [(p, json.loads(p.read_text())) for p in candidates]
        matches = [(p, r) for p, r in matches if r["identity"]["n"] == n]
        if len(matches) != 1:
            raise ValueError(f"Expected one inference cache for {prior}, S={s}, q={q}, n={n}; found {len(matches)}")
        marker, record = matches[0]
        snapshot = json.loads((marker.parents[2]/"source_snapshot.json").read_text())
        for name in ("prior_data.py", "data.py", "families.py"):
            if snapshot.get(name) != Path(__file__).with_name(name).read_text():
                raise ValueError(f"Training data source differs from current {name}; do not regenerate blindly")
        cfg = PriorConfig(**record["identity"]["config"])
        if (cfg.prior, cfg.s, cfg.q, cfg.seed, cfg.data_seed, cfg.n, cfg.M) != (prior, s, q, seed, seed, 8, 2**19):
            raise ValueError("Unexpected model configuration")
        with np.load(marker.parent/"per_example.npz", allow_pickle=False) as z:
            arrays = {k: z[k][:count].copy() for k in z.files}
        if len(arrays["targets"]) != count or count < 2:
            raise ValueError("Invalid subset size")
        bank = generate(cfg, "test", 0, count, n=n)
        for key in ("theta", "targets", "tau"):
            # Inverse-normal CDF can differ by a few float64 ulps across HPC/Mac SciPy builds.
            matches_data = (np.allclose(bank[key], arrays[key], rtol=0, atol=1e-13)
                            if key == "theta" else np.array_equal(bank[key], arrays[key]))
            if not matches_data:
                raise ValueError(f"Regenerated {key} does not match exported cache")
        # Cross-check the explicit regression design against the repository solver.
        A = posterior_module.design(4*bank["points"][:, :, 0].astype(float))
        if not np.allclose(np.einsum("bnp,bp->bn", A, bank["theta"]),
                           solution(bank["theta"], bank["points"][:,:,0], cfg.family)[:,:,0],
                           atol=1e-12, rtol=1e-12):
            raise ValueError("Regression design does not match exact trajectory")
        identity = dict(config=cfg.to_dict(), n=n, count=count,
                        points_hash=digest(bank["points"].tobytes()),
                        targets_hash=digest(bank["targets"].tobytes()))
        folder = root/"statistical_comparison_cache"/"prior_screen"/digest(
            json.dumps(identity, sort_keys=True).encode())[:20]
        fit_identity = dict(**identity, fit_source=fit_source, steps=sgd_steps,
                            batch=sgd_batch, lr=sgd_lr, max_nfev=1000)
        fit_key = digest(json.dumps(fit_identity, sort_keys=True).encode())[:20]
        def compute_fits():
            mle, md = fit_mle(bank["points"], cfg.family, 1, 1000, seed+n)
            sgd, sd = fit_sgd(bank["points"], cfg.family, sgd_steps, sgd_batch, sgd_lr, seed+n)
            if not np.all(md[:,1] == 1):
                raise RuntimeError("Bounded least-squares solver failed")
            return dict(mle=mle, sgd=sgd, mle_diagnostics=md, sgd_diagnostics=sd)
        fits = cached(folder/f"fits_{fit_key}.npz", fit_identity, compute_fits)
        estimates, diagnostics = [], []
        # Small blocks preserve progress if interrupted during a posterior sweep.
        for start in range(0, count, 32):
            stop = min(count, start+32)
            post_id = dict(**identity, source=posterior_source, quadrature=quadrature,
                           start=start, stop=stop, algorithm_seed=seed+n)
            post_key = digest(json.dumps(post_id, sort_keys=True).encode())[:20]
            def compute_post():
                vals, ds = [], []
                for i in range(start, stop):
                    val, d = posterior_module.posterior(
                        A[i], bank["points"][i,:,1].astype(float)*scale, q*scale,
                        prior, s, seed+n+100003*i, **quadrature)
                    vals.append(encode(val, cfg.family))
                    ds.append([d["converged"], d["quad_change"], d["ess"], d["power"]])
                return dict(estimate=np.array(vals), diagnostics=np.array(ds))
            post = cached(folder/f"posterior_{post_key}.npz", post_id, compute_post)
            estimates.append(post["estimate"])
            diagnostics.append(post["diagnostics"])
        estimates, diagnostics = np.concatenate(estimates), np.concatenate(diagnostics)
        predictions = dict(zip(METHODS, (arrays["mean_prediction"], estimates, fits["mle"], fits["sgd"])))
        errors = {name: np.mean((pred.astype(float)-bank["targets"])**2, axis=1)
                  for name, pred in predictions.items()}
        if any(not np.isfinite(e).all() for e in errors.values()):
            raise ValueError("Non-finite test errors")
        records.append(dict(n=n, errors=errors, posterior_diagnostics=diagnostics,
                            mle_diagnostics=fits["mle_diagnostics"],
                            sgd_diagnostics=fits["sgd_diagnostics"], folder=str(folder)))
        print(f"{prior} S={s:g} q={q:g} n={n}: " +
              ", ".join(f"{name}={e.mean():.6g}" for name,e in errors.items()) +
              f"; posterior unconverged={int(np.sum(diagnostics[:,0] == 0))}/{count}", flush=True)
    return dict(prior=prior, s=s, q=q, seed=seed, count=count, records=records)


def plot_setting(root, result, resamples=5000):
    import matplotlib.pyplot as plt
    from .paper_style import apply_style
    apply_style()
    fig, ax = plt.subplots(figsize=(6.8, 3))
    ns = np.array([r["n"] for r in result["records"]])
    colors = ("C0", "C3", "C1", "C2")
    for method, color in zip(METHODS, colors):
        errors = [r["errors"][method] for r in result["records"]]
        means = np.array([e.mean() for e in errors])
        ci = np.array([bootstrap_mean_ci(e, 7731+int(n), resamples) for n,e in zip(ns,errors)])
        ax.fill_between(ns, ci[:,0], ci[:,1], color=color, alpha=.22, zorder=1)
        ax.plot(ns, means, color=color, marker="o" if method == "Transformer" else "*",
                markersize=4 if method == "Transformer" else 6, linewidth=1.7,
                label=method, zorder=3)
    ax.axvline(8, color="gray", linestyle=":", linewidth=1)
    ax.set(xscale="log", yscale="log", xlabel="Inference observations n",
           ylabel="Parameter test MSE",
           title=f'{result["prior"].replace("_", " ")}: S={result["s"]:g}, q={result["q"]:g}')
    ax.set_xticks(ns, labels=[str(n) for n in ns])
    ax.grid(True, alpha=.18)
    ax.legend(loc="center left", bbox_to_anchor=(1.01,.5), frameon=False)
    fig.tight_layout()
    folder = Path(root)/"figures"/"prior_screen"
    folder.mkdir(parents=True, exist_ok=True)
    stem = f'{result["prior"]}_s{result["s"]:g}_q{result["q"]:g}_parameter_statistics'
    for suffix in ("pdf", "png"):
        fig.savefig(folder/f"{stem}.{suffix}", bbox_inches="tight", dpi=180)
    return fig


def plot_prior_pair(root, results, prior, resamples=5000):
    """One shared-legend pair: concentration by panel, noise by line style."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from .paper_style import apply_style, PAIR_SIZE, BAND_ALPHA, context_ticks
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(PAIR_SIZE[0], 4.0), sharey=True)
    colors = ("C0", "C3", "C1", "C2")
    for ax, s in zip(axes, (.2, .1)):
        for q, linestyle in ((.1, "-"), (.2, ":")):
            matches = [r for r in results if (r["prior"], r["s"], r["q"]) == (prior, s, q)]
            if len(matches) != 1:
                raise ValueError(f"Expected one result for {prior}, S={s}, q={q}")
            records = matches[0]["records"]
            ns = np.array([r["n"] for r in records])
            for method, color in zip(METHODS, colors):
                errors = [r["errors"][method] for r in records]
                means = np.array([e.mean() for e in errors])
                ci = np.array([bootstrap_mean_ci(e, 7731+int(n), resamples)
                               for n, e in zip(ns, errors)])
                ax.fill_between(ns, ci[:, 0], ci[:, 1], color=color,
                                alpha=BAND_ALPHA, linewidth=0, zorder=1)
                ax.loglog(ns, means, color=color, linestyle=linestyle,
                          marker="o" if method == "Transformer" else "*",
                          markersize=4, linewidth=2.1, zorder=3)
        ax.axvline(8, color="gray", linestyle=":", linewidth=1, alpha=.65)
        ax.set(title="Moderately concentrated prior (FL)" if s == .2 else "Highly concentrated prior (FL)",
               xlabel="Inference observations n")
        context_ticks(ax, ns)
        ax.grid(True, which="both", alpha=.18)
    axes[0].set_ylabel("Normalized MSE")
    handles = [Line2D([], [], color=color, marker="o" if method == "Transformer" else "*",
                      linewidth=2.1, markersize=4, label=method)
               for method, color in zip(METHODS, colors)]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .08),
               ncol=4, fontsize=10, handlelength=3, columnspacing=1.4, framealpha=.95)
    fig.tight_layout(rect=(0, .16, 1, 1))
    folder = Path(root)/"figures"/"prior_screen"
    folder.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(folder/f"{prior}_parameter_statistics_pair.{suffix}", dpi=180)
    return fig, axes


def summary_frame(results, resamples=5000):
    import pandas as pd
    rows = []
    for result in results:
        for record in result["records"]:
            for method, errors in record["errors"].items():
                lo, hi = bootstrap_mean_ci(errors, 7731+record["n"], resamples)
                rows.append(dict(prior=result["prior"], s=result["s"], q=result["q"],
                                 seed=result["seed"], n=record["n"], method=method,
                                 mse=float(errors.mean()), ci_low=lo, ci_high=hi,
                                 posterior_unconverged=int(np.sum(record["posterior_diagnostics"][:,0] == 0))))
    return pd.DataFrame(rows)

