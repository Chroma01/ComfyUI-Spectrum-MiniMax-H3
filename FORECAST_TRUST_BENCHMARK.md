# Forecast trust-region benchmark record

This document records the forecast-quality investigation following PR #39. Hidden-feature ratios and advantages are feature-space error measurements, **not perceptual-quality percentages**.

## Main finding: forecast geometry is part of the calibration problem

> **Parameters or directions calibrated for one forecast geometry cannot be assumed to transfer to a different endpoint geometry.**

PR #45 produced two independent examples.

### Causal trust kappa -> future-bracket replay

Causal trust is calibrated between the latest exact causal anchor and a causal forecast. The rejected replay transfer instead interpolated between a stale causal hold and a future-bracketed replay proposal. The replay-native leave-one-out oracle selected `kappa=1.0` throughout the scored audio/video replay targets. Full replay retention was optimal; movement toward the stale causal hold was rejected.

RACER Proposition 2 establishes that the MSE-optimal interpolation coefficient depends on the endpoint error statistics/correlation. That supports endpoint-geometry dependence in principle. RACER does **not** analyze Spectrum H3's future-bracket replay construction and does **not** directly predict the observed `kappa=1` replay oracle. The replay result here is empirical.

### Causal PR #39 latest-delta scalar -> future replay bracket direction

PR #39 remains supported in the geometry in which its scalar was learned:

```text
d_causal = latest_exact - previous_exact
```

Offline replay had transplanted the same scalar onto:

```text
d_replay = right_future_bracket_anchor - left_future_bracket_anchor
```

Across independent 25-step trace geometries, the persisted causal scalar was negative while replay-native useful projections were positive. The causal/replay direction cosine was approximately zero or negative, and applying the causal scalar in replay worsened hidden-feature reconstruction. This rejects the **transfer**, not the causal PR #39 mechanism.

## Replay generic correction: closed, default disabled

The predeclared decision criterion is satisfied:

- hidden-feature rejection of causal-scalar -> future-bracket replay transfer replicated across independent trace geometries;
- two same-pair D/B perceptual comparisons judged correction-disabled B **non-worse / indistinguishable**.

Therefore:

```text
model_aware_replay_generic_correction = false   # supported default
model_aware_replay_generic_correction = true    # explicit legacy/reproduction ablation
```

`false` suppresses only the transplanted replay scalar. It does **not** remove or weaken the causal PR #39 latest-delta correction. Validation attenuation, local/spectral blending, scheduling, exact anchors, transformer NFE, and ER-SDE tail policy are unchanged.

`true` remains supported for regression tests, scientific reproduction, and legacy ablation.

No perceptual improvement is claimed. The perceptual result is limited to the tested comparisons being non-worse/indistinguishable after disabling the replay transfer.

This replay-generic-correction question is closed unless concrete regression evidence gives a reason to reopen it.

## Trace identity and provenance discipline

Historical runs must not be casually renamed as "seed 2" or "seed 3" when seed/config provenance was not actually logged.

Two known distinct trace geometries are:

```text
trace_37905_57x70x38_t927
  target_video_rows = 37905
  video_shape       = (1, 24, 57, 70, 38)
  text_length       = 927

trace_37620_57x60x44_t922
  target_video_rows = 37620
  video_shape       = (1, 24, 57, 60, 44)
  text_length       = 922
```

They are not the same trace rescored by a later shadow. When a true seed is unavailable, the runtime calibration export records `seed=null`; the offline evaluator accepts an explicit external `--seed` and `--label` annotation instead of inventing provenance.

## Absolute video spectral evidence

Audio replay remains local-dominated with zero direct spectral blend. The local/spectral calibration work below is therefore **video-only**.

The earlier absolute-spectral trace `trace_37905_57x70x38_t927` reported:

```text
local ratio                         0.728943
current attenuated blend ratio      0.734564
full spectral ratio                 1.039023
oracle ratio                        0.719751
oracle advantage vs local          +1.2498%
oracle advantage vs current        +2.0028%

oracle absolute spectral weight mean  0.127587
current projected weight mean         0.257248
```

A fixed absolute-weight sweep selected local among the fixed controls. Useful video spectral contribution remained small and target-dependent.

## Pure multiplicative alpha: useful baseline, rejected complete solution

The latest alpha-capable trace is `trace_37620_57x60x44_t922`.

```text
local ratio       0.730744
current ratio     0.734151
oracle ratio      0.721140
```

Predeclared alpha sweep:

```text
alpha 0.00    ratio 0.730744    headroom capture   0.000%
alpha 0.25    ratio 0.726611    headroom capture  43.039%
alpha 0.50    ratio 0.725793    headroom capture  51.553%
alpha 0.75    ratio 0.728315    headroom capture  25.294%
alpha 1.00    ratio 0.734151    headroom capture -35.476%
```

Best predeclared alpha:

```text
alpha = 0.50
```

Weight prediction:

```text
                         alpha=1/current     alpha=.5
MAE                         0.131288          0.090482
RMSE                        0.149899          0.100304
bias                       +0.131288         -0.000126
```

The approximate factor-of-two over-scaling hypothesis was real. However, the predeclared complete-solution requirement was at least 60% oracle-vs-local headroom capture. The best predeclared alpha reaches only `51.553%` on its development trace.

Conclusion:

> **Pure global multiplicative alpha is retained as a useful calibration baseline and rejected as a complete replay solution.**

Do not densify the alpha sweep, add an applied alpha setting, or spend another generation trying to rescue the alpha-only hypothesis. `alpha=0.5` remains the residual-analysis baseline.

## Residual structure after alpha=.5

On `trace_37620_57x60x44_t922`:

```text
residual vs causal disagreement      +0.922956
residual vs validation penalty       -0.795547
residual vs spectral gap             +0.661601
residual vs coordinate               +0.816378
residual vs current weight           +0.835505
```

The same trace's floor diagnostic showed:

```text
oracle-near-zero targets               2 / 11 = 18.18%
current weight there                    0.227179
alpha=.5 weight there                   0.113590
share of alpha=.5 MAE                  22.825%
share of alpha=.5 squared error        23.685%
```

Global scaling fixes much of the bias; inability to express a target-specific zero is a real residual; the floor explains only part of the remaining error.

There are only `n=11` sequential trajectory targets. Residual correlations are exploratory diagnostics, not formal IID statistical inference:

```text
|r| < 0.5          weak
0.5 <= |r| < 0.6  indeterminate / noise-floor region
|r| >= 0.6         candidate residual structure on that trace
```

A reusable residual relationship requires the **same predictor**, **same sign**, and `|r|>=0.6` on independent traces. No zero gate, disagreement controller, coordinate controller, validation-penalty controller, or learned controller is authorized from this one trace.

## Exact per-target replay calibration dataset

PR #45 now exports a compact machine-readable VIDEO replay calibration block in existing `debug=true` research logging. Detailed export is active only for `model_aware_mode=full` with offline replay; normal non-debug production execution does not collect the extra research rows.

Log marker:

```text
SPECTRUM_REPLAY_CALIBRATION_JSON={...}
```

The block contains scalar provenance/metadata and `target_rows[]`. No hidden-feature tensors or raw feature arrays are serialized. Payload size is bounded.

### Authoritative quadratic representation

Sign convention:

```text
e = local - withheld_target
d = spectral - local
```

Serialized moments use exactly the replay-shadow sampled VIDEO scoring subset:

```text
A = local_error_sq_mean
  = mean(e^2)

B = local_error_dot_spectral_delta_mean
  = mean(e * d)

C = spectral_delta_sq_mean
  = mean(d^2)
```

Any absolute local->spectral mixture weight `w`, clipped to `[0,1]`, has:

```text
candidate_mse(w) = A + 2*w*B + w^2*C
```

The analytic unconstrained optimum is `-B/C`. Runtime parity preserves the existing denominator guard by using:

```text
w_oracle = clamp(-B / max(C, ratio_epsilon^2), 0, 1)
```

### Exact ratio normalization

The runtime replay metric is a per-target RMS error normalized by the stale-hold error RMS, with the existing epsilon floor:

```text
actual_rms = sqrt(mean(actual^2))
ratio_epsilon = max(actual_rms * 1e-6, float32_epsilon)

hold_error_sq_mean = mean((actual - hold)^2)
ratio_denominator_rms = max(sqrt(hold_error_sq_mean), ratio_epsilon)

ratio(w) = sqrt(candidate_mse(w)) / ratio_denominator_rms
```

Aggregate reporting is the **mean of per-target ratios**, not a globally pooled RMS.

The serialized moments must reconstruct runtime local/current/full-spectral/fixed-control/oracle values within the declared tight parity tolerance (`2e-5`). A block that fails parity is marked/rejected as incompatible rather than fitted.

### Calibration row schema

Identity/geometry:

```text
schema_version
run_id
trace_fingerprint
target_step_id
target_anchor_index
coordinate
left_anchor_step_id
left_anchor_index
right_anchor_step_id
right_anchor_index
bracket_coordinate_spacing
bracket_fraction
target_video_rows
scoring_sample_count
```

Quadratic/normalization:

```text
local_error_sq_mean
spectral_delta_sq_mean
local_error_dot_spectral_delta_mean
hold_error_sq_mean
ratio_epsilon
ratio_denominator_rms
```

Deployable pre-target predictors:

```text
current_weight
causal_disagreement
validation_penalty
spectral_gap
coordinate
```

Post-target labels/parity:

```text
oracle_weight
required_adjustment
local_ratio
current_ratio
full_spectral_ratio
oracle_ratio
fixed_0p00_ratio
fixed_0p25_ratio
fixed_0p50_ratio
fixed_0p75_ratio
fixed_1p00_ratio
current_axis_ratio
moment_oracle_weight
moment_oracle_ratio
max_parity_abs_error
row_compatible
```

Predictor construction is completed before the withheld target is read. The withheld target is then used only to compute moments, errors, oracle labels, and evaluation values. No production replay path consumes oracle fields.

### Provenance and fingerprints

Block provenance includes:

```text
package_name
package_version
source_schema_revision
source_revision
source_revision_source
seed
label
config_hash
schedule_fingerprint
topology_fingerprint
trace_fingerprint
```

`config_hash` is SHA-256 of canonical JSON for the complete `SpectrumH3Config` snapshot.

`topology_fingerprint` covers the available native H3 topology/configuration identity, including feature/video/audio shapes, text length, hidden width, target row counts, patch size, sigma shifts, AdaLN mode, segments, refs, and keyframes.

`schedule_fingerprint` covers the complete archive sequence of `(step_id, coordinate, actual)`.

`trace_fingerprint` combines schema/source revision information, package/source revision when available, config hash, sampler/step count, schedule fingerprint, topology fingerprint, and the pre-target per-row target signature. Oracle/post-target values are deliberately excluded.

No hot-path Git subprocess or `.git` dependency exists. A source revision is recorded only when safely provided through `SPECTRUM_H3_SOURCE_REVISION`; otherwise it is null. Runtime seed is null unless cleanly available; the analysis tool accepts external annotations.

## Offline multi-run evaluator

Tool:

```text
tools/analyze_replay_calibration.py
```

It is standard-library/CPU-only and starts neither ComfyUI nor a model/GPU runtime.

Examples:

```bash
python tools/analyze_replay_calibration.py run1.log
python tools/analyze_replay_calibration.py run1.log run2.log
python tools/analyze_replay_calibration.py run1.log --label development --seed 123
python tools/analyze_replay_calibration.py run1.log run2.log \
  --label run1.log=trace_A --seed run1.log=123 \
  --label run2.log=trace_B --seed run2.log=456
python tools/analyze_replay_calibration.py run1.log run2.log --json
```

It accepts direct calibration JSON or full ComfyUI logs containing the calibration marker. Each calibration block is a run; it never treats the 11 rows as independent train/test samples.

### Level 0: fixed/known baselines

```text
local:    w=0
current:  w=current_weight
alpha:    w=alpha*current_weight
          alpha in {0,.25,.50,.75,1}
```

No dense alpha optimization is performed.

### Level 1: affine current-weight recalibration

```text
w_hat = clamp(a + b*current_weight, 0, 1)
```

This is the mandatory baseline before attributing residual structure to a new predictor.

### Level 2: current weight + one residual predictor

Separately:

```text
w_hat = clamp(a + b*current_weight + c*predictor, 0, 1)
```

for:

```text
causal_disagreement
validation_penalty
spectral_gap
coordinate
```

Coordinate is the explicit confound/control. There is no combined multi-predictor model, interaction, polynomial, threshold search, tree, neural network, or AutoML path.

### Fitting objective

Level 1/2 use the exact training-run quadratic moments and minimize the unweighted mean of per-target **squared normalized hidden-feature error before clipping**:

```text
q_i = 1 / ratio_denominator_rms_i^2
L = mean_i q_i * (A_i + 2*B_i*w_i + C_i*w_i^2)
w_i = x_i^T beta
```

This gives an auditable closed-form linear system:

```text
sum(q*C*x*x^T) beta = -sum(q*B*x)
```

A single fixed numerical ridge of `1e-8` stabilizes the solve. There is no hyperparameter sweep and no predictor normalization; final weights are clipped to `[0,1]` and evaluated from the exact moments.

### Whole-run validation and evidence labels

The trajectory/run is the unit of generalization:

```text
1 run  -> in-sample exploratory only
          NON-CONFIRMATORY / DEVELOPMENT ONLY

2 runs -> leave-one-run-out
          WEAK / PRELIMINARY

>=3    -> leave-one-run-out over complete trajectories
          MULTI-RUN / GENERALIZATION TEST
```

No row-randomized split exists. Held-out runs cannot influence coefficients, normalization, regularization, or model selection.

For each fold/model the evaluator reports mean hidden-feature ratio, advantage vs local/current, oracle-vs-local headroom capture, oracle-weight MAE/RMSE/bias, and coefficients. Level 2 reports deltas and incremental headroom versus Level 1. Non-coordinate Level 2 candidates are also compared directly with the coordinate-control Level 2 family.

A Level 2 predictor does not survive because it beats alpha=.5 or correlates with an oracle residual. It must improve held-out results over Level 1 affine current-weight calibration. Disagreement/penalty/spectral-gap claims additionally require comparison with coordinate control.

## Historical exact-calibration compatibility

```text
trace_37905_57x70x38_t927: NO
```

It predates the exact alpha/moment export. The available record contains aggregate endpoint/oracle diagnostics, not the serialized per-target quadratic moments and exact denominator state required for arbitrary offline `w` scoring.

```text
trace_37620_57x60x44_t922: NO
```

It contains the alpha aggregate evidence that closed the pure-alpha hypothesis, but it also predates the quadratic-moment schema. Missing per-target moments are not synthesized from aggregate/oracle/endpoint summaries.

Both traces remain valid historical evidence and development context; neither is admitted into fitted exact offline-controller evaluation.

## C1: next expensive user-facing gate

The next user generation is **C1 causal single-pass trust**, not another replay-controller experiment.

Both runs:

```text
sampler = sample_er_sde
steps = 25
model_aware_mode = full
model_aware_risk_threshold = 0.65
offline_smoothing_replay = false
```

Baseline:

```text
model_aware_trust_shrinkage = false
```

Candidate:

```text
model_aware_trust_shrinkage = true
```

Keep seed, prompt, references, resolution, duration/frame count, scheduler, CFG, model/checkpoint, precision, and all remaining workflow settings identical.

Expected baseline telemetry:

```text
model_aware_trust_enabled=0
model_aware_trust_applied=0
model_aware_trust_path=disabled
model_aware_trust_applications=0
model_aware_trust_extra_transformer_nfe=0
```

Expected candidate telemetry:

```text
model_aware_trust_enabled=1
model_aware_trust_applied=1
model_aware_trust_path=causal_single_pass
model_aware_trust_applications>0
model_aware_trust_extra_transformer_nfe=0
model_aware_trust_failures=0
```

Both must retain the same 14-actual / 11-forecast schedule, 14 actual transformer calls, zero fallbacks, exact ER-SDE tail steps 23/24, and no offline replay. Existing counted C1 regressions cover the isolation mechanically; the trust formula is unchanged.

Prior causal shadow evidence is much larger than the residual replay spectral headroom:

```text
AUDIO generic-corrected ~1.54 -> trust candidate ~0.99
      oracle headroom ~35%
VIDEO generic-corrected ~1.24 -> trust candidate ~1.00
      oracle headroom ~19%
```

C1 asks whether that large hidden-feature gain produces an obvious user-facing visual/audio gain.

- **C1 positive:** causal single-pass trust becomes the primary confirmation/promotion candidate; replay datasets can be collected in parallel.
- **C1 neutral:** strongly deprioritize engineering work aimed at another ~1-2% replay hidden-feature gain while retaining the measurement infrastructure.
- **C1 negative:** stop trust promotion and investigate feature/perceptual mismatch.

Regardless of C1, do not transfer causal trust into future-bracket replay.

## Current research state

### Supported

- causal PR #39 generic latest-delta correction in causal geometry.

### Supported for testing / active

- causal single-pass trust; next gate is C1.

### Rejected / closed

- causal trust kappa -> replay transfer;
- hold-anchor replay shrinkage;
- causal scalar -> replay correction transfer;
- pure global multiplicative alpha as a complete replay solution;
- K=2 correction;
- FinalLayer-transformed correction;
- previous-error direction families.

### Active research

- VIDEO replay local/spectral residual calibration using exact per-target quadratic moments and whole-run offline evaluation.

Audio remains local-dominated in replay and is not part of the video mixture-controller research.

## Safety / implementation boundaries

- no applied affine replay calibration;
- no applied residual replay controller;
- no zero/floor gate;
- no change to causal trust formula/default;
- no causal-kappa replay transfer;
- no extra transformer NFE;
- no model-weight scan for calibration export;
- no retained calibration feature tensors;
- no broad runtime refactor;
- replay calibration export is debug/research-only and scalar;
- ordinary calibration-diagnostic failures are isolated; CUDA OOM propagates;
- PR #45 remains draft and must not be merged solely because CI is green.
