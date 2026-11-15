# Architecture screen (18 additional models)

This is separate from the existing 52-run-per-seed grid. Both seeds (20260906
and 20260907, including their data seeds) use the same nine variations.
Only forced-linear parameter estimation is screened, at n=128, M=2^19,
30 epochs. Training data, validation data, noise, optimizer, learning-rate
schedule, dropout, clipping, and finite-value checks match the original setup.

| Variation | Change from existing reference |
|---|---|
| mean_output | MSE of the mean output token, averaged over parameter coordinates |
| relu | ReLU instead of squared ReLU |
| heads_1, heads_4, heads_8 | Full-width global heads; outputs summed |
| depth_8, depth_32 | Number of independent blocks |
| width_32 | Hidden width 32; FFN width 128; square Q/K/V 32 by 32 |
| width_128 | Hidden width 128; FFN width 512; square Q/K/V 128 by 128 |

The reference uses depth 16, hidden width 64, FFN width 256, two heads,
squared ReLU, and all-token loss. It is loaded from the existing main runs,
not retrained. All variants except mean_output average squared error over
examples, tokens, and parameter coordinates. mean_output averages tokens
BEFORE squaring the error. Validation uses the training objective to choose
best.pt. History keys explicitly say objective_mse rather than mislabeling
mean-output loss as all-token loss.

Head-count changes alter parameter counts (heads remain full-width).
Width changes resize input/output projections and norms, but never change
the physical input or target dimension. This is fixed-epoch, not fixed-compute.
Both test metrics are plotted for every model so comparisons use a common
metric; individual seed curves are not confidence intervals.

Each test curve has a pointwise 95% confidence interval for the mean, computed
separately for each trained model/seed as mean ± 1.959963984540054 × sample SD
/ sqrt(test count). Coordinates are averaged within each trajectory before
taking SD. Mean-output intervals use each trajectory's mean prediction error.
These are normal-approximation intervals for test-bank sampling uncertainty,
not retraining variability or simultaneous curve coverage. The CSV retains
SD, standard error, test count and raw CI bounds. Nonpositive lower bounds are
clipped at the log-axis limit and labeled. Learning curves show means only.
Rerun plotting with existing per-example caches; inference need not be repeated.

## HPC commands

From the existing remote Paper repo directory, inside tmux:

```bash
bash hpc/run_architecture.sh list
bash hpc/run_architecture.sh train
# After training, run these separately:
bash hpc/run_architecture.sh evaluate
bash hpc/run_architecture.sh plot
```

Each command covers BOTH seeds by default. Train runs only the 18 new
configurations. Evaluate uses best-validation checkpoints at inference
n=2,4,8,16,32,64,128,256,512, including the two existing references.
Valid caches, including completed n=128 evaluations, are reused.
Plot reads saved results only. It generates five inference-scaling comparisons
(readout, activation, heads, depth, width), each with all-token and mean-output
MSE panels and 95% confidence intervals for the mean, plus the original n=128 comparisons.
Use --ns with either evaluate or plot to select a subset of context sizes.
Missing references/results produce an error, not a silently incomplete plot.

Results are nested under architecture_screen in each existing seed root:
- <hpc-root>/ode-results/paper-v1/architecture_screen
- <hpc-root>/ode-results/paper-seed-20260907/architecture_screen

Data banks are shared with the main runs in each seed root. Combined figures,
per-model learning curves and summary.csv are written to the second root's
architecture_screen/figures. Logs default to
<hpc-root>/ode-results/architecture-screen/logs.

Use --root-first and --root-second for other result locations, --seeds to
select one seed, and --device cpu or mps for local testing. The default is CUDA.
Interrupted training resumes from the last completed epoch. Best and last
weights are both saved. Original source files participating in existing
checkpoint/cache hashes are not changed. The isolated screen training and
evaluation loops retain the original recovery and safety checks.

Upload these new files into the corresponding existing remote folders:
- src/ode_paper/architecture.py
- src/ode_paper/architecture_train.py
- src/ode_paper/architecture_evaluate.py
- src/ode_paper/architecture_cli.py
- hpc/run_architecture.sh
- src/ode_paper/test_dispersion.py

For an HPC checkout that already has the screen and confidence-interval plotting code, this
inference-scaling update requires uploading only src/ode_paper/architecture_cli.py.
The lower-level evaluation code is unchanged to retain all cache identities;
the launcher explicitly passes the complete inference grid.

The documentation and tests can also be uploaded; no results need relocating.

