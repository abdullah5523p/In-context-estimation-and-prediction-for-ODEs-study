# Concentrated-prior / noise sweep

This forced-linear parameter study (the truncated-Gaussian prior comparison in
the paper) runs 4 models: one seed (20260906), a one-mode truncated Gaussian
prior, two S values (0.2, 0.1), and two q values (0.10, 0.20). Each uses n=8, M=2^19,
and the unchanged default 16-block architecture and 30-epoch optimizer schedule.

Prior: independent Gaussian coordinates centered at (0.2,0.2,0.2),
conditioned on the original [-0.8,0.8]^3 box.
S is the Gaussian SD BEFORE truncation.
Sampling uses inverse CDFs, not clipping Gaussian draws.

Times remain unsorted uniform [0,3.2]. States use the original fixed
RMS scale 0.6532414253078556; physical noise SD is q times that scale.
Training loss is squared error averaged over examples, tokens and all
three normalized parameter coordinates. There is no additional factor of n.

Separate train/validation/test random streams are used, distinct from
the original experiment. Within a seed, base uniforms and noise are
paired across settings; inference contexts share prefixes. Validation
has 4096 examples, test 8192. Test distributions match each training prior
and noise setting. Cross-setting risks thus refer to different populations.

Run on HPC from the repository root:

    bash hpc/run_prior.sh list
    bash hpc/run_prior.sh train
    bash hpc/run_prior.sh evaluate

Training runs all four configurations sequentially. Evaluation is separate, at
n=2,4,8,16,32,64,128,256,512 (override with --ns).
Default output root: <hpc-root>/ode-results/prior-screen,
with a seed_20260906 subfolder. Override with --root.
The CLI (src/ode_paper/prior_cli.py) selects these four settings from the larger
grid in prior_config.py, which also supports a two-mode mixture prior; that file
stays unchanged because its bytes identify the cached runs.
No original experiment files, checkpoint identities or data are modified.
Completed training loads; interrupted runs resume from last completed epoch.
Evaluation loads matching cached results. Best-validation and last weights,
optimizer/RNG state and learning curves are saved, as in the original trainer.

Inference caches contain per-example coordinate all-token squared errors,
mean parameter predictions, targets, true parameters, and metadata.
Both all-token MSE and mean-output MSE are reported. These allow later
paired statistical comparisons without rerunning transformer inference.
No plotting or statistical-baseline run is launched automatically; see
notebooks/08_prior_statistical_comparisons.ipynb.

The training and evaluation implementations are isolated copies of the
original protocol with only data imports/cache identities changed.
Do not change core numerical settings mid-sweep: source fingerprints
intentionally distinguish different implementations.

