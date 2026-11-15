# Joint parameter and trajectory prediction

16 models: four families × training n=8,128 × seeds 20260906,20260907.
All use M=2^19.
Default 16 blocks, width 64, FFN 256, two full-width global heads, squared ReLU,
RMS normalization, bias-free projections, dropout 0.1. Same 30-epoch optimizer,
schedule, clipping and finite checks as the main experiment.

Input: (t_i,y_i,tau), dimension m+2. Target: concatenation of normalized theta
(first p entries) and normalized x(tau) (last m entries). Loss averages squared
error over examples, tokens and all p+m coordinates. Thus parameter/trajectory
block weights are p/(p+m), m/(p+m), not equal task weights. Best checkpoint is
selected by combined validation loss; last checkpoint retains optimizer/RNG state.

Prediction data banks are reused unchanged. Their disk files contain only points
and trajectory targets; theta is regenerated from the same deterministic stream.
Test observations, parameters and tau match the existing separate-task evaluations.

Run from the HPC repository, inside tmux:

```bash
bash hpc/run_joint.sh list
bash hpc/run_joint.sh train
bash hpc/run_joint.sh evaluate
bash hpc/run_joint.sh export
```

Each command defaults to both seeds (16 models total); use `--seeds` to select
one. Training resumes or loads completed runs.
Evaluation defaults to n=2,4,8,16,32,64,128,256,512 and loads matching caches.
No original source files or checkpoint identities are modified. New runs live in
each existing seed result root under joint_experiment/runs/FAMILY/joint/.

Evaluation saves parameter_all_token_mse, trajectory_all_token_mse and
combined_all_token_mse separately, including per-example values. The first two
are normalized by p and m respectively; combined is normalized by p+m. Also saved:
per-coordinate all-token errors, concatenated mean predictions/targets, theta,
tau, mean-output losses for each block, physical coordinate errors in metrics,
and birth-death identifiable-combination errors of mean parameter outputs.
Full per-token predictions are not retained, matching the existing cache policy.
Training history's all_token_mse fields refer to the combined objective.

Export creates joint_seed_SEED.tar.gz under each seed root's joint_experiment/.
It includes best.pt (not last.pt), histories, manifests, source snapshots and all
evaluation caches with checksums. Its joint_catalog.json is separate from old
catalog.json files so extraction into evaluation_cache_downloads does not overwrite
existing catalogs. Plotting is intentionally left to later notebooks.
