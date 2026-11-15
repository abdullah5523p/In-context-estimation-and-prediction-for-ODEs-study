# RMS calibration provenance

`rms_observation_interval.json` and `.csv` are outputs of the executed notebook `notebooks/01_rms_calibration.ipynb`. They estimate each state coordinate's uncentered RMS under the specified independent uniform parameter prior and physical observation time uniform on [0,3.2], without noise. The separate calibration seed is 20261019, with 8,388,608 draws per family.

The resulting `rms` values are now the active state divisors in `src/ode_paper/families.py`, for both observation noise and state normalization. Full-precision estimates are retained in the code; Monte Carlo standard errors and convergence records accompany them here.

The calculation was executed before these scales were adopted. Consequently, `existing_state_scale`, `training_scales_changed: false`, the source fingerprint, and the notebook's printed statements describe the calibration execution at that time. They are historical provenance, not the current training configuration. The old reference constants are not used by training. The calibration records have not been retroactively rewritten.

Re-executing the notebook recomputes and overwrites these calibration outputs, but never edits training constants automatically. No training, validation or test bank is used in calibration.
