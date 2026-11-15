"""Test-task dispersion from existing per-example inference caches."""
from pathlib import Path
import numpy as np

def test_dispersion(cache, record):
    """Sample SD across independent tasks, not across tokens, coordinates or seeds."""
    path = Path(cache)/"per_example.npz"
    if not path.exists():
        raise FileNotFoundError(f"Per-example cache required for confidence intervals: {path}")
    with np.load(path, allow_pickle=False) as saved:
        coordinate_errors = saved["all_token_coordinate_mse"].astype(np.float64)
        prediction = saved["mean_prediction"].astype(np.float64)
        targets = saved["targets"].astype(np.float64)
    if coordinate_errors.ndim != 2 or prediction.shape != targets.shape or targets.shape != coordinate_errors.shape:
        raise ValueError(f"Invalid per-example shapes: {path}")
    count = coordinate_errors.shape[0]
    if count != record["identity"]["count"] or count < 2:
        raise ValueError(f"Unexpected test-task count: {path}")
    if not all(np.isfinite(a).all() for a in (coordinate_errors, prediction, targets)) or (coordinate_errors < 0).any():
        raise ValueError(f"Invalid cached errors: {path}")
    values = {"all_token_mse": coordinate_errors.mean(axis=1),
              "mean_output_mse": ((prediction-targets)**2).mean(axis=1)}
    result = {"test_count": count}
    for name, errors in values.items():
        mean, sd = float(errors.mean()), float(errors.std(ddof=1))
        if name in record["metrics"] and not np.isclose(mean, record["metrics"][name], rtol=1e-6, atol=1e-12):
            raise ValueError(f"Cached mean disagrees with per-example errors ({name}): {path}")
        se = sd/np.sqrt(count)
        half_width = 1.959963984540054*se
        result[name+"_se"] = float(se)
        result[name+"_ci95_half_width"] = float(half_width)
        result[name+"_ci95_lower"] = float(mean-half_width)
        result[name+"_ci95_upper"] = float(mean+half_width)
        result[name+"_std"] = sd
        result[name+"_minus_std"] = mean-sd
        result[name+"_plus_std"] = mean+sd
    return result

def shade_test_ci(ax, bands):
    """Draw pointwise normal-approximation 95% CIs for the mean; clip nonpositive lower bounds on log axes.

    bands contains (x, means, ci95_half_widths, color) for each separate seed.
    Raw, unclipped bounds are retained in the summary CSV.
    """
    if not bands:
        return
    positive, upper_values = [], []
    for x, mean, half_width, color in bands:
        mean, half_width = np.asarray(mean, dtype=float), np.asarray(half_width, dtype=float)
        positive.extend(mean[mean > 0])
        lower = mean-half_width
        positive.extend(lower[lower > 0])
        upper_values.extend(mean+half_width)
    if not positive:
        raise ValueError("Log-scale MSE plot requires at least one positive error")
    floor = min(positive)/1.3
    ceiling = max(max(upper_values)*1.3, floor*2)
    clipped = False
    for x, mean, half_width, color in bands:
        mean, half_width = np.asarray(mean, dtype=float), np.asarray(half_width, dtype=float)
        lower, upper = mean-half_width, mean+half_width
        clipped |= bool((lower <= 0).any())
        ax.fill_between(x, np.maximum(lower, floor), np.maximum(upper, floor),
                        color=color, alpha=.45, linewidth=0, zorder=1)
        if len(mean) == 1:
            ax.vlines(x, np.maximum(lower, floor), np.maximum(upper, floor),
                      color=color, alpha=.45, linewidth=4)
    ax.set_ylim(floor, ceiling)
    note = "Shading: pointwise 95% CI for mean test MSE"
    if clipped:
        note += "\nNonpositive lower bounds clipped at axis limit"
    ax.text(.02, .02, note, transform=ax.transAxes, fontsize=7, va="bottom",
            bbox=dict(facecolor="white", alpha=.8, edgecolor="none"))
