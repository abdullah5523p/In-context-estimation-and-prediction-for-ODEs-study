# Archived artifacts (Zenodo)

Trained model checkpoints and evaluation caches are too large for GitHub. They
are archived on Zenodo together with a snapshot of this repository:

**DOI:** `10.5281/zenodo.XXXXXXX` <!-- TODO: replace with the DOI after publishing the Zenodo record -->

The record contains 9 archives (1.2 GB) and `SHA256SUMS.txt`, covering every
experiment reported in the paper and its supplement. Use the archives in place:
the notebooks expect these file names and folders and verify checksums when they
load files, so do not rename archives or move their contents.

## What to download

| Goal | Archives needed |
|---|---|
| Redraw every figure in the paper (no GPU) | all except `model_weights` |
| Obtain every trained model | `model_weights`, `joint_seed_*`, `prior_seed_20260906` |
| Retrain from scratch | none; see `docs/hpc.md` |

## Model checkpoints

All 146 reported models are included as best-validation checkpoints (`best.pt`).

| Experiment | Models | Archive | Location after extraction |
|---|---|---|---|
| Main grid, both tasks, both seeds | 104 | `model_weights.tar.gz` | `model_weights/{paper-v1,paper-seed-20260907}/runs/` |
| Architecture screen, both seeds | 18 | `model_weights.tar.gz` | `model_weights/{paper-v1,paper-seed-20260907}/architecture_screen/` |
| Joint estimation and prediction, both seeds | 16 | `joint_seed_20260906.tar.gz`, `joint_seed_20260907.tar.gz` | `evaluation_cache_downloads/seed_<seed>/joint_experiment/` |
| Truncated-Gaussian prior comparison | 4 | `prior_seed_20260906.tar.gz` | `evaluation_cache_downloads/prior-screen/seed_20260906/runs/one_mode/` |

`paper-v1` is seed 20260906. The initial-state experiment reuses the main-grid
models and has no checkpoints of its own.

## Evaluation outputs

| Archive | Contents |
|---|---|
| `evaluation_seed_20260906.tar.gz`, `evaluation_seed_20260907.tar.gz` | Main-grid and architecture-screen evaluation caches, catalogs, manifests and training histories |
| `joint_seed_20260906.tar.gz`, `joint_seed_20260907.tar.gz` | Joint-experiment evaluation caches (together with the checkpoints above) |
| `initial_state_seed_20260906.tar.gz`, `initial_state_seed_20260907.tar.gz` | Initial-state recovery evaluations |
| `prior_seed_20260906.tar.gz` | Prior-comparison evaluations (together with the checkpoints above) |
| `statistical_comparison_cache.tar.gz` | Cached MLE, projected-SGD and posterior-mean fits, conditional-variance and query-sweep results |

## Figures

| Paper figures | Notebook | Archives |
|---|---|---|
| Forced-linear scaling in M and n; inference-context scaling | `02_local_evaluation_plots` | `evaluation_seed_*` |
| Architecture screen (depth, width, heads, readout, activation) | `02_local_evaluation_plots` | `evaluation_seed_*` |
| Joint versus separate models (all families) | `02_local_evaluation_plots` | `evaluation_seed_*`, `joint_seed_*` |
| Pretraining-risk scaling fits | `05_pretraining_risk_scaling_fits` | `evaluation_seed_*` |
| Statistical comparisons (parameter and prediction panels); conditional variance | `03_statistical_comparisons`, then `04_prediction_statistical_comparisons` | `evaluation_seed_*`, `statistical_comparison_cache` (`model_weights` only to recompute the cache) |
| Truncated-Gaussian prior comparison | `08_prior_statistical_comparisons` | `prior_seed_20260906`, `statistical_comparison_cache` |
| Joint-model parameter error versus query time | `07_joint_parameter_query_sweep` | `joint_seed_*`, `statistical_comparison_cache` |
| Initial-state recovery | `06_initial_state_plots` | `initial_state_seed_*` |

## Where to put the archives

```text
<repo>/
├── statistical_comparison_cache.tar.gz      # extract here: tar -xzf statistical_comparison_cache.tar.gz
├── model_weights/
│   └── model_weights.tar.gz                 # extract here: cd model_weights && tar -xzf model_weights.tar.gz
└── evaluation_cache_downloads/              # notebooks 02 and 06 extract their archives themselves;
    ├── evaluation_seed_20260906.tar.gz      # extract the others with tar -xzf in this folder
    ├── evaluation_seed_20260907.tar.gz
    ├── initial_state_seed_20260906.tar.gz
    ├── initial_state_seed_20260907.tar.gz
    ├── joint_seed_20260906.tar.gz
    ├── joint_seed_20260907.tar.gz
    └── prior-screen/
        └── prior_seed_20260906.tar.gz
```

## Checksums

`SHA256SUMS.txt` in the record lists SHA-256 checksums for all archives. Check
with `shasum -a 256 -c SHA256SUMS.txt` (macOS) or `sha256sum -c SHA256SUMS.txt`
(Linux) in the download folder.

| Archive | Size |
|---|---|
| `model_weights.tar.gz` | 423 MB |
| `evaluation_seed_20260906.tar.gz`, `evaluation_seed_20260907.tar.gz` | 239 MB each |
| `joint_seed_20260906.tar.gz`, `joint_seed_20260907.tar.gz` | 95 MB each |
| `initial_state_seed_20260906.tar.gz`, `initial_state_seed_20260907.tar.gz` | 40 MB each |
| `prior_seed_20260906.tar.gz` | 31 MB |
| `statistical_comparison_cache.tar.gz` | 8.2 MB |
