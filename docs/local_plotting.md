# Download evaluation caches and plot locally

Notebook: notebooks/02_local_evaluation_plots.ipynb.

**Reproducing the paper's figures:** download the archives from Zenodo and place
them as described in [ZENODO.md](../ZENODO.md), then go directly to step 4.
Steps 1–3 document how the archives were exported from the original HPC runs.

Run with Python 3.11.8 or newer, NumPy, Matplotlib and Jupyter.
The notebook can be opened in Jupyter or VS Code with a Python kernel.

## 1. Upload the exporter from a Mac terminal

```bash
sftp <user>@<hpc-host>
```

At sftp>:
```text
lcd "<path-to-local-repo>"
cd "<hpc-root>/ode-work/Paper repo"
put hpc/export_evaluations.py hpc/export_evaluations.py
bye
```

## 2. Package the caches in the HPC terminal

```bash
cd "<hpc-root>/ode-work/Paper repo"
<hpc-root>/ode-env/bin/python hpc/export_evaluations.py
```

It verifies all 549 evaluations per seed and exports the current matching caches,
manifests and training histories. It does not run models. Missing evaluations
produce an explicit error. There are 1098 unique evaluations for 122 models;
architecture references reuse main models. Archives exclude model weights and
training data banks.

## 3. Download into the repo from the Mac terminal

```bash
mkdir -p "<path-to-local-repo>/evaluation_cache_downloads"
sftp <user>@<hpc-host>
```

At sftp>:
```text
lcd "<path-to-local-repo>/evaluation_cache_downloads"
cd "<hpc-root>/ode-results/evaluation-downloads"
get evaluation_seed_20260906.tar.gz
get evaluation_seed_20260907.tar.gz
bye
```

## 4. Open the notebook and run its cells

The first cells extract both archives and verify checksums. Local plotting needs
no model checkpoints, GPU, SSH connection, or inference rerun.
The plotting cells expose styling, metric, model/context selection, and confidence
intervals. Figures save to figures/main_experiment/notebook_plots and
figures/architecture_screen/notebook_plots.

Saved arrays contain per-example coordinate MSE averaged over output tokens,
mean output predictions, true targets, parameters, and query times. They do not
contain individual output-token predictions. All-token MSE is obtained by averaging
the saved coordinate MSE across coordinates; do not square it again.
The CI uses mean +/- 1.959963984540054 * sample_SD / sqrt(test_count).
Both seeds remain separate; these intervals describe test-bank sampling uncertainty
conditional on a trained model.

The bundles are excluded from Git because they are generated experiment outputs.
