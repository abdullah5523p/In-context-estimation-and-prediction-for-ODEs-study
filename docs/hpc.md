# UMD GPU execution

The previously tested host was `<hpc-host>`, with an NVIDIA A100 80 GB and the environment `<hpc-root>/ode-env` (PyTorch 2.8.0+cu128). The instructions below do not assume Slurm or authorize bypassing local allocation rules. Use the GPU access procedure assigned to your account. The wrapper is Bash, so it also works when your interactive login shell is tcsh.

## Transfer the new repository

Upload this entire source folder using SFTP to a new remote folder, for example `<hpc-root>/ode-work/Paper-repo`. Do not upload earlier artifacts or overwrite the old repo. Create the destination parent first if needed. Example in the Mac's SFTP session:

```text
put -r "<path-to-local-repo>" <hpc-root>/ode-work/Paper-repo
```

SFTP destination behavior depends on whether the target already exists: after uploading, verify that `<hpc-root>/ode-work/Paper-repo/src/ode_paper` and `hpc/run.sh` exist (avoid an accidentally nested `Paper repo` directory).

## Check the existing environment (remote SSH terminal)

```bash
cd <hpc-root>/ode-work/Paper-repo
<hpc-root>/ode-env/bin/python -c 'import torch, numpy, matplotlib; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))'
bash hpc/check.sh
bash hpc/run.sh list
```

The wrappers set PYTHONPATH to this repository; installation is not required on the HPC. They use the existing PyTorch installation and do not replace it. `check.sh` runs tiny CPU correctness tests in temporary folders, not production training or a hyperparameter pilot.

## Launch only when ready

```bash
tmux new -s ode-paper
cd <hpc-root>/ode-work/Paper-repo
bash hpc/run.sh train --index 0
```

For the entire grid, replace `--index 0` with `--all`. Runs are serial on one GPU. Detach using Ctrl-b then d; reconnect with `tmux attach -t ode-paper`. Do not start a second overlapping sweep: per-run locks protect checkpoints, but different runs could still compete for the same GPU.

After training completes:

```bash
bash hpc/run.sh evaluate --all
bash hpc/run.sh plot
```

You can evaluate one completed run first with `evaluate --index 0`. The default sweep evaluates all nine inference n values for every selected model, including M below 2^19. To stage a subset, use `--ns 2 8 32 128 512`; omitted values can be added later without recomputing existing caches. Reduce `--eval-batch-size` or `--token-budget` if evaluation memory is constrained; training batch size is never automatically changed.

Default generated outputs: `<hpc-root>/ode-results/paper-v1` for seed 20260906, and `<hpc-root>/ode-results/paper-seed-20260907` for seed 20260907. Override with `--root /another/path` (including logs). To change environment variables in tcsh, use `setenv NAME value`; in Bash/zsh use `export NAME=value`.

Repeat the same training command after interruption. It restores the last completed epoch and the original schedule. A progress heartbeat is not a checkpoint. Resume requires matching backend, GPU model and NumPy/PyTorch versions. Code changes produce new identities; finish a sweep on a frozen source version.

Use SFTP to download `figures` for figures/CSV and `runs` for checkpoints and per-task caches. All outputs initially live on the HPC, not your laptop. Keep checkpoints and provenance backed up before cleaning any data. Exact data banks can be regenerated from source and seeds.

Before publication, retain `environment.txt` from `check.sh`, the source version, and the generated manifests. Performance and CUDA kernel behavior should be verified on the actual A100; local CPU checks cannot establish GPU speed or exact GPU reproducibility.

## Unified two-seed runs

Both seeds use `src/ode_paper/runs.py` as the complete configuration list and the
same training, evaluation, and plotting commands. Each has 52 configurations:
40 forced-linear runs and 12 other-family runs (both tasks at n=8 and n=128).
Both model and data seeds change together. The original 46 indices are unchanged;
the six n=128 configurations are appended. No separate extra-run script is needed.

Run each stage separately in the remote repository:

```bash
bash hpc/run.sh train --all --seed 20260906
bash hpc/run.sh evaluate --all --seed 20260906
bash hpc/run.sh plot --seed 20260906

bash hpc/run.sh train --all --seed 20260907
bash hpc/run.sh evaluate --all --seed 20260907
bash hpc/run.sh plot --seed 20260907
```

Completed training runs load, unfinished runs resume, and valid inference caches
are reused. Existing checkpoint and cache identities and results locations are
unchanged. `config.py` retains the original grid because its bytes participate in
checkpoint identities; `runs.py` supplies the complete shared list.

For combined plots:

```bash
bash hpc/run.sh plot --seed 20260907 --previous-root <hpc-root>/ode-results/paper-v1
```

The same `--previous-root` option works for training and evaluation to process
both seeds sequentially. Combined plots and their CSV go into the second seed's
`figures` directory; learning curves stay with their runs. Overlay lines represent
individual seeds, not confidence intervals. Missing evaluations are omitted.
No results need moving or copying. The old `run_seed_20260907.sh` command remains
as a compatibility shortcut to this same CLI, not a separate implementation.

To update the existing HPC checkout, upload `src/ode_paper/runs.py`,
`src/ode_paper/cli.py`, `src/ode_paper/plots.py`, `hpc/run.sh`, and
`hpc/seed_20260907.py` to their corresponding folders. The historical separate
extra-run scripts can then be removed from the server; no result files should be
removed. The current remote checkout is `<hpc-root>/ode-work/Paper repo`.
