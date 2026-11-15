# Data-generating protocol

All equations below use physical time t. The network receives s=t/4. Observation s is uniform on [0,0.8]; query s is independently uniform on [0,1]. All observations remain unsorted. Independent Gaussian noise has standard deviation 0.02 after state normalization. The same fixed scales are used for inputs, prediction targets, and inversion to physical units.

| Family | ODE | Uniform independent training parameters | State divisor |
|---|---|---|---|
| Forced linear | x' = -x+a+bt | (x0,a,b) each in [-0.8,0.8] | 0.6532414253078556 |
| Logistic | x' = r x (1-x/K) | x0 in [0.1,0.3], r in [0.6,1.8], K in [0.8,1.6] | 0.7160787998680879 |
| Birth–death | x' = (b-d)x-cx², known x0=0.2 | b in [0.8,1.8], d in [0.1,0.6], c in [0.2,0.8] | 1.0066636624210321 |
| Stuart–Landau | x'=ax-omega y-(x²+y²)x; y'=omega x+ay-(x²+y²)y | x0,y0 in [0.4,0.6], a in [0.8,1.2], omega in [2,3] | x: 0.6564067052579569; y: 0.6798484425369343 |

Parameter coordinate encoding: z_j = 1.6 [theta_j-(low_j+high_j)/2]/(high_j-low_j). Thus each target coordinate is uniform on [-0.8,0.8]. Forced-linear encoding is the identity. No per-trajectory normalization or calibration from validation/test tasks is used. State divisors estimate sqrt(E[X_k(t)^2]) under the parameter prior and t uniform on [0,3.2], using 8,388,608 independent calibration draws per family (seed 20261019). These fixed constants do not change with M or n and are not hard state bounds. They also normalize clean prediction targets throughout [0,4]. The displayed precision preserves the exact sampled estimates, not a claim of that many statistically accurate digits; uncertainty is recorded in the calibration JSON.

For Stuart–Landau let r0²=x0²+y0². Its exact solution is the rotation by omega*t of (x0,y0), multiplied by

    [exp(-2*a*t) + (r0²/a)*(1-exp(-2*a*t))]^(-1/2).

The implementation uses expm1 to reduce cancellation. All four families are evaluated from closed formulas; no ODE discretization error is introduced. Tests compare finite-difference time derivatives against the stated vector fields, including the factor of four from normalized time.

## Assumptions and intended contrasts

Forced linear is identifiable from its three independent trajectory basis functions. Logistic has x0<K uniformly, avoiding equilibrium trajectories; the rate and carrying capacity are identifiable from a nonconstant full trajectory. Stuart–Landau starts away from zero, with positive omega, so radius evolution and rotation identify a and omega. Its clean radius never exceeds sqrt(1.2). It completes at least one revolution within the physical observation horizon because omega>=2. These three families have smooth finite-horizon solutions; on the selected compact truth sets their full-trajectory sensitivity maps are injective, yielding a uniform positive Gram lower bound under full-support time sampling. This is not a claim of identifiability from every small finite set of observation times.

Birth–death deliberately violates parameter identifiability: only (b-d,c) affects the trajectory. Moreover, the preserved support includes the equilibrium surface b-d=0.2c. That surface has prior probability zero, but near-equilibrium tasks can be weakly informative about the combinations. Parameter MSE has an irreducible ambiguity component. Auxiliary combination errors use the mean token output decoded into physical (b,d,c), and are labeled separately from all-token risk.

## Metric and selection conventions

Training/validation loss is mean squared error over examples, tokens, and output dimensions, in normalized target units. Best epoch is selected using validation tasks at the run's training n. Prediction selection uses the full uniform query distribution; approximately 80% of queries are interpolation and 20% extrapolation. No test metric is used for model selection.

Inference reports normalized all-token risk, normalized mean-output risk, coordinatewise all-token risk, and corresponding aggregate physical-unit risks. Prediction also reports interpolation and extrapolation subsets and their actual counts. The per-example archive contains normalized per-coordinate all-token error and the mean prediction, not every token's full prediction tensor. This is sufficient for paired risk estimates without very large output caches.

Training n and inference n always count observed trajectory samples. Broadcast tau adds a coordinate, not an extra token. Inputs have dimension m+1 (parameter task) or m+2 (prediction); outputs have dimension p or m. Parameter targets have p=3 for the scalar families and p=4 for Stuart–Landau. State dimension is m=1 or m=2 respectively.
