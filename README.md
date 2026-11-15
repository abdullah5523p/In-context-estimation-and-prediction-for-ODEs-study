# Transformer estimators for ODE inverse and prediction problems

Reproducible experiments for **parameter estimation** and **broadcast-query trajectory prediction**, accompanying the paper *In-Context Estimation and Prediction for Parametric ODEs* (A. Ahmed and Y. Yang).

Training can generate new artifacts, while the local plotting notebooks use the evaluation downloads. Trained model weights and evaluation caches, which are needed to redraw the figures without retraining, are archived on Zenodo; see [ZENODO.md](ZENODO.md). The HPC instructions document the original UMD setup; adjust account-specific paths for another machine.

## Repository layout

```text
src/ode_paper/          # installable package: shared data, model, training, evaluation, plots
  config.py            # hyperparameters; original grid retained for checkpoint compatibility
  runs.py              # shared complete 52-run grid for either seed
  families.py          # exact solutions, priors and adopted RMS state scales
  data.py              # deterministic data generation and disk banks
  model.py             # transformer and all-token loss
  train.py             # training, validation and recovery
  evaluate.py          # cached inference sweeps and metrics
  plots.py             # plots from existing results only
  cli.py, utils.py      # command routing and shared utilities
hpc/                   # remote launch and correctness-check scripts
notebooks/             # calibration, local plots, statistical and prediction comparisons
calibration/           # calibration provenance and numerical results (tracked)
tests/                 # CPU correctness and pipeline tests
docs/                  # data protocol, HPC, architecture-screen, joint and prior-sweep notes
pyproject.toml         # package dependencies and ode-paper command
ZENODO.md              # archived model weights and evaluation caches (Zenodo)
results/               # generated local outputs, if requested (ignored by git)
```

Training, evaluation and plotting have separate entry points. The default HPC output root is outside the source repository. Runtime datasets, checkpoints and figures should not be committed to GitHub; the small calibration record should be retained with the source.

## Fixed experiment

| Setting | Value |
|---|---|
| Families | Forced linear, logistic growth, birth–death, Stuart–Landau |
| Training tasks M | 2^11, 2^13, 2^15, 2^17, 2^19 |
| Training observations n | 2, 8, 32, 128 |
| Inference observations n | 2, 4, 8, 16, 32, 64, 128, 256, 512 |
| Seed / data seed | 20260906 / 20260906 and 20260907 / 20260907 |
| Training | 30 epochs, effective batch size 256, AdamW |
| AdamW | LR 3e-4, weight decay 1e-4, betas (0.9, 0.999), epsilon 1e-8 |
| Schedule | 5% linear warmup, cosine decay, updated after every optimizer step |
| Gradient clipping | Global L2 norm at 1.0; finite inputs, losses, gradients and updated weights checked |
| Selection | Lowest validation mean all-token MSE among epochs 1–30 |
| Validation / test tasks | 4,096 / 8,192 |
| Relative observation noise | q=0.02 in normalized state coordinates |

The M and training-n sweeps above apply only to **forced linear** (40 models across both tasks per seed). Logistic, birth–death and Stuart–Landau each use **training n=8 and 128, M=2^19**, for both tasks (12 models per seed). There are **52 configurations per seed, 104 models across two seeds**, and 936 inference evaluations across nine context sizes. Both seeds use the same configuration list, loaders and output structure. Inference is a separate command and does not retrain models. At the largest M, each run uses 61,440 optimizer updates; smaller M uses proportionally fewer updates. This is fixed-epoch, not fixed-compute scaling.

## Architecture and loss

An additional [architecture screen](docs/architecture.md) compares nine variations
for each seed (18 new runs) at forced-linear parameter estimation, n=128, M=2^19.
It reuses the existing reference models and leaves the main 52-run grid unchanged.
Run both seeds with `bash hpc/run_architecture.sh train`; evaluation and plotting
are separate commands using the same launcher.

Every token is projected to width 64. Sixteen independent blocks apply:

1. Pre-RMSNorm, bias-free 64→256→64 FFN with squared ReLU, output dropout, residual addition.
2. Pre-RMSNorm, two global attention heads, summed output, output dropout, residual addition.

Every head has its own **64×64 Q, K and V**. Attention scores are divided by sqrt(64). There is no concatenation, no learned attention output projection, no positional encoding, no attention-probability dropout, and no sorting of observation times. Dropout is 0.1 at the two residual branches. RMSNorm uses learned gains and epsilon 1e-6. Final RMSNorm precedes the bias-free task output projection. The implementation stores tokens as rows; this is the transpose of the column-token notation in the paper. Square matrices from different heads are packed into linear layers for efficient execution, without sharing their weights.

Main-grid training averages squared error over **examples, tokens, and target coordinates** (L_all). Parameter targets are normalized theta. Prediction targets are normalized clean X_theta(tau), repeated at every output token. This is **not** the squared error of the average prediction. The architecture screen's mean-output variant instead uses L_mean, averaging squared error of the mean token prediction over examples and coordinates. Both metrics are separately retained at evaluation.

This is a practical architecture motivated by the theory, not the exact constructive network: it retains RMSNorm, FFN residuals, dropout, expanded FFN width and two global masks.

## Data and units

See [the protocol](docs/protocol.md) for equations, parameter ranges and assumptions.

Physical observations are drawn on [0,3.2]; physical query times are drawn on **[0,4]**. Input times are divided by 4, giving observation times in [0,0.8] and query times in [0,1]. Exact solutions evaluate physical time 4s. The observation horizon has not been shortened. Prediction therefore covers interpolation and extrapolation; both regions are reported separately in addition to their combined error.

Independent uniform parameter coordinates are encoded into [-0.8,0.8]. States use fixed coordinatewise RMS divisors, without centering, calibrated over the parameter prior and physical observation interval [0,3.2]. Noise is added independently in every observed state coordinate, with physical standard deviation q times that divisor. Query targets are clean and use the same divisors. This approximately equalizes population-averaged signal-power/noise-power ratios per coordinate at 1/q² (up to calibration error); it does not equalize per-trajectory SNR, identifiability or statistical difficulty.

An index-addressed pseudorandom generator makes train/validation/test disjoint streams, M banks nested, and observations at smaller n prefixes of observations at larger n. Parameters and observations are shared across the two tasks. Query times are independent of observations and parameters. Arrays are generated in float64 and stored as float32. Banks are memory-mapped on disk; chunk size does not change samples. A fixed training bank is reused for all 30 epochs, with a new shuffle each epoch.

## Install and run

Python 3.11+ is required. Install the appropriate PyTorch build for your machine first, then:

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
ode-paper list
ode-paper train --index 0 --device cuda --root /path/to/paper-results
ode-paper evaluate --index 0 --device cuda --root /path/to/paper-results
ode-paper plot --root /path/to/paper-results
```

The package declares NumPy, PyTorch, Matplotlib and SciPy dependencies. SciPy is
required by the statistical baselines and posterior integration. For notebook
execution, install `python -m pip install -e '.[notebooks]'` and select that
environment's Python kernel in your notebook editor. Dependency ranges allow
compatible releases; they are not an exact lockfile for the original HPC environment.

Use `--all` instead of `--index 0` to explicitly select all 52 runs, serially. Use the same commands with `--seed 20260907 --root /path/to/seed-20260907` for the second seed; both model and data seed change together. The default seed is 20260906. Training and inference remain separate commands. On a Mac, choose `--device mps`; CPU is also supported. Device failure never silently falls back to CPU. Do not mix backends within an interrupted run.

`list` prints zero-based indices. Forced linear occupies 0–39: the first 20 are parameter estimation and the next 20 prediction, ordered by training n then M. The n=8 runs for logistic occupy 40–41, birth–death 42–43, and Stuart–Landau 44–45. Their n=128 runs occupy 46–47, 48–49, and 50–51 respectively, always parameter estimation then prediction. Indices are identical for both seeds. No training command runs automatically on import.

To process both seeds in one command, select `--seed 20260907 --root /path/to/seed-20260907 --previous-root /path/to/seed-20260906`. This works for training, evaluation and plotting. Training/evaluation also require `--all` or `--index`. Combined plots and CSV are written under the second root; existing runs and inference caches remain in their original folders. The training and evaluation sources used in checkpoint/cache fingerprints are unchanged by this consolidation. `config.py` retains its historical grid solely for compatibility; use `runs.configurations()` for the full experiment list.

For the previously configured UMD A100 host, use [HPC instructions](docs/hpc.md).

### Command-line plots

Test-error curves include a light pointwise 95% confidence interval for the
mean test error, computed from existing `per_example.npz` caches as
mean ± 1.959963984540054 × sample SD / sqrt(number of test trajectories).
Each trajectory contributes one loss averaged over tokens and target coordinates
(or the squared error of its mean prediction for the mean-output metric).
Intervals use the normal approximation and are conditional on each trained
model, separately for each seed. They measure test-bank sampling uncertainty;
they do not quantify retraining variability or simultaneous coverage of a curve.
Summary CSV files retain the SD, standard error, test count and raw CI bounds.
Nonpositive lower bounds are clipped at the log-axis limit and labeled.
Training/validation learning curves retain means only. Rerunning plotting is
sufficient for existing evaluations; inference need not be repeated. Download
per-example caches as well as metrics to regenerate these intervals locally.

Plotting is a separate stage (`ode-paper plot`, or `bash hpc/run.sh plot`); it does not train or run inference. For forced linear and both tasks it produces:

1. Training/validation learning curves for each run (PNG under that run's folder).
2. M scaling with four training-n curves; **inference n equals training n** on each curve.
3. Inference-n scaling at fixed training n=128, with five M curves.
4. At M=2^19, inference-n scaling with four training-n curves.

For each other family/task, it produces learning curves and inference-n scaling curves for the n=8 and n=128, M=2^19 models; no one-point M-scaling plot is generated. Comparison figures are saved as PDF and PNG in `figures/` (12 figures for a complete grid, plus the per-run learning curves). Combined two-seed plots show individual seed curves, with test-bank confidence intervals computed separately for each model, not across seeds. Main-grid curves use normalized test all-token MSE; prediction averages all query times. Interpolation/extrapolation metrics remain in the caches and summary CSV, but are not separate panels in these figures. Missing results are not fabricated; available points are plotted, and wholly unavailable comparisons are skipped.

### Local notebooks and paper figures

The local notebook workflow is separate from the command-line plotting workflow:

| Notebook | Purpose |
|---|---|
| `01_rms_calibration.ipynb` | Calibrate coordinatewise state RMS scales |
| `02_local_evaluation_plots.ipynb` | Architecture screens and main M/context scaling plots from downloaded evaluation caches |
| `03_statistical_comparisons.ipynb` | Parameter estimation versus bounded MLE and projected SGD, optional posterior means, and conditional-variance experiments |
| `04_prediction_statistical_comparisons.ipynb` | Direct prediction versus parameter estimation composed with the exact forward solution |

Notebook 02 uses the saved 8,192-example test banks. Statistical-comparison
notebooks default to the same first 256 examples per seed for the estimators
being compared. Notebook 03's conditional-variance experiments instead generate
repeated datasets at fixed parameters and require downloaded transformer weights.
Existing compatible statistical results are cached; missing results require computation.
Posterior integration is optional and can be expensive.

The test-error bands in notebooks 02–04 use **pointwise 95% percentile-bootstrap
confidence intervals for the mean**, with 5,000 resamples of the per-example
errors, separately for each model/seed. They describe test-bank sampling
uncertainty, not across-training-seed uncertainty or simultaneous coverage.
This differs from the normal-approximation intervals in the command-line
plotters described above. Rerunning cached plotting cells requires neither
retraining nor repeating transformer inference. Conditional variance is a
different statistic from these confidence intervals.

Metrics are explicit in the plotting cells: parameter comparisons generally use
L_all, while birth–death identifiable-combination comparisons use the mean
parameter output. The readout screen compares mean-output errors for both
training losses. Notebook 04 evaluates the direct prediction transformer's
mean output so that each method supplies one prediction per query.

See [local cache download instructions](docs/local_plotting.md) for the export
and download workflow. Local artifacts live in `evaluation_cache_downloads/`,
`model_weights/`, and `statistical_comparison_cache/`; these are not supplied by
installing the Python package. Figure destinations are specified in the notebook
cells under `figures/`. The older plotting guides describe command-line
normal-approximation intervals; use the notebook description above for the
bootstrap-based paper plots.

The executed [RMS calibration notebook](notebooks/01_rms_calibration.ipynb) estimates prior-averaged, coordinatewise clean-state RMS over the observation interval using an independent Monte Carlo bank. Results and sampling diagnostics are in `calibration/rms_observation_interval.json` and its CSV summary. The resulting full-precision RMS values have been adopted in `families.py`. Running the notebook alone does not modify those constants. The saved notebook outputs and calibration record retain their original pre-adoption context; see [calibration provenance](calibration/README.md). To install notebook execution dependencies, use `python -m pip install -e '.[notebooks]'`.

## Outputs and recovery

```text
results/                         # or the explicit --root; excluded from git
  data/<data-identity>/           # reusable memory-mapped banks
  runs/<family>/<task>/<run-id>/
    manifest.json                # config, source hash, environment, parameter count
    source_snapshot.json         # exact source used for training
    last.pt                      # last completed epoch, optimizer, scheduler, RNG, best state
    best.pt                      # best-validation model, epoch and selection metric
    history.json                 # normalized losses, gradient norms, clipping, times
    progress.json                # heartbeat during an epoch; checkpoint is authoritative
    inference/n<n>_<identity>/
      metrics.json
      per_example.npz            # errors, mean predictions, truth and query times
    learning_curve.png
  figures/                       # CSV, M curves and inference-n curves
```

Rerunning a completed training command loads its checkpoint. Interrupted runs restart from the last **completed epoch**, restoring optimizer, LR schedule, shuffle and dropout RNG states. An incomplete epoch is replayed, not counted as completed. `last.pt` also contains the best weights so derived files can be recovered after an interrupted save. The original 30-epoch schedule is not extended or restarted. There is no destructive force-retrain option: use a new output root for an intentional fresh run.

Evaluation uses best.pt and caches each n separately. A cached result is reused only when checkpoint content, data configuration and evaluation implementation match. Changing evaluation batch size does not intentionally invalidate otherwise equivalent results. Changed training source/configuration produces a different run identity. Plotting includes only current-grid runs and current-checkpoint caches. Missing runs remain missing; old results are never silently substituted.

CUDA uses float32 with TF32 disabled; no automatic mixed precision, compilation, or batch-size adaptation. PyTorch scaled-dot-product attention can use available optimized kernels. Deterministic algorithms are requested with warnings for unsupported operations; bitwise equality across hardware/software versions is not claimed. Exact epoch-resume equality is tested on CPU. Record and retain the environment for HPC replication.

## License

The code is released under the MIT License; see [LICENSE](LICENSE).
