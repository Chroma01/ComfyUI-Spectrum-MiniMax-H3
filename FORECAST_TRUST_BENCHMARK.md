# Forecast trust-region benchmark record

This document records the continuous forecast-quality investigation following PR #39. Hidden-feature ratios and advantages below are feature-space error measurements; they are **not** perceptual-quality percentages.

Retired/rejected directions remain out of scope: K=2 trajectory correction, FinalLayer-transformed correction, previous-error directions, hold-anchor replay shrinkage, hard refresh/re-pay, extra transformer evaluations, and dense threshold/controller fitting on one trace.

## Headline finding: causal calibration does not generally transfer to future-bracket replay

**Parameters or directions calibrated in causal forecast geometry do not generally transfer to future-bracketed offline replay geometry.**

Two independent failures now support this conclusion.

### 1. Causal trust kappa -> replay

Causal trust is calibrated on:

```text
latest exact causal anchor
    <->
causal forecast
```

The attempted replay transfer instead used:

```text
stale causal hold
    <->
future-bracketed replay proposal
```

The replay-native LOO oracle selected `kappa=1.0` for every scored audio and video target on the stale-hold -> current-replay segment. Full replay retention was optimal; movement toward the stale causal hold was rejected.

This is consistent with the endpoint dependence described by RACER Proposition 2: the MSE-optimal interpolation coefficient depends on endpoint error second moments and their error correlation. Changing the endpoint pair therefore changes the trust problem.

This does **not** mean Proposition 2 predicts our observed replay `kappa=1`. RACER does not analyze this future-bracket replay geometry. The replay oracle is the empirical evidence.

### 2. PR #39 causal latest-delta scalar -> replay bracket direction

PR #39 established a useful bounded scalar correction in causal geometry:

```text
d_causal = latest_exact - previous_exact
r = actual - forecast_uncorrected
g = projection of r onto d_causal
corrected = forecast + g * d_causal
```

That scalar remains supported when applied to `d_causal`.

Offline replay had transplanted the same scalar onto:

```text
d_replay = right_future_bracket_anchor - left_future_bracket_anchor
```

Two independent 25-step traces reject that transfer.

#### Seed 1

```text
AUDIO
replay correction effect            -8.7879%
persisted causal gain               -0.129556
replay-native projection            +0.086194
causal/replay direction cosine      +0.004650

VIDEO
local correction effect             -5.8856%
blend correction effect             -5.5538%
persisted causal gain               -0.108478
replay projection local             +0.108435
replay projection blend             +0.092066
causal/replay direction cosine      -0.098340
```

#### Seed 2

```text
AUDIO
local                                0.688827
corrected local                      0.746748
correction effect                   -8.6128%
correction oracle kappa              0
persisted causal gain               -0.130713
replay-native projection            +0.080818

VIDEO
local                                0.728943
corrected local                      0.771162
uncorrected blend                    0.734564
corrected blend                      0.774853

local correction effect             -5.8441%
blend correction effect             -5.4923%

local correction oracle mean/min/max
                                     0 / 0 / 0
blend correction oracle mean        0.002648
blend correction oracle max         0.029126

persisted causal gain               -0.110935
replay-native projection local      +0.109721
replay-native projection blend      +0.094006
causal/replay direction cosine      -0.116476
```

The precise conclusion is:

> PR #39's scalar correction remains supported in the causal direction in which it was learned. Its scalar does not transfer to the geometrically different future-bracket replay direction.

Neither Spectrum nor RACER directly studies this offline future-bracket replay geometry, so this remains a standalone reusable negative result.

## Causal trust remains supported in single-pass geometry

The causal observer compares the global Chebyshev proposal with the local linear/secant proposal on existing bounded cache evidence:

```text
disagreement = RMS(p_spectral - p_linear) / max(RMS(p_spectral), epsilon)

kappa =
    exp(-0.3 * max(horizon - 1, 0))
    * sigmoid(4.0 * (0.15 - disagreement))
```

Representative 25-step causal shadow evidence:

```text
                              audio       video
generic-corrected ratio       1.542742    1.240801
causal oracle ratio           0.987258    0.998710
causal oracle advantage      +34.9660%   +19.3114%
theta=0.15 ratio              0.990551    1.005021
theta=0.15 advantage         +34.7545%   +18.7989%
```

Public setting:

```text
model_aware_trust_shrinkage: bool = false
```

Single-pass semantics:

```text
offline_smoothing_replay=false
model_aware_trust_path=causal_single_pass
```

Offline replay semantics remain deliberately separate:

```text
offline_smoothing_replay=true
model_aware_trust_path=offline_replay_shadow_only
model_aware_trust_applied=0
model_aware_trust_applications=0
model_aware_trust_replay_application=disabled_rejected_causal_transfer
```

A positive single-pass causal result would not authorize transferring causal trust into replay.

## Replay A/B/C/D decomposition

The replay LOO diagnostic separates:

```text
A = local
B = validation-attenuated spectral/local blend without replay generic correction
C = local + transplanted replay generic correction
D = blend + transplanted replay generic correction
```

Seed 1:

```text
AUDIO
A local                  0.681730
B blend_uncorrected      0.681730
C local_corrected        0.739193
D blend_corrected        0.739193
B advantage vs D         +7.8915%

VIDEO
A local                  0.722708
B blend_uncorrected      0.728863
C local_corrected        0.764653
D blend_corrected        0.769209
B advantage vs D         +5.2181%
```

Seed 2 independently reproduced rejection of the replay correction as recorded above.

Do not add correction-removal and spectral-oracle percentages together. They use different endpoints/baselines. Any combined gain must be measured directly on a joint candidate.

## D -> B applied gate and predeclared default criterion

Public setting:

```text
model_aware_replay_generic_correction: bool = true
```

Semantics:

```text
true  = existing/default D replay
false = experimental B replay; suppress only transplanted generic replay correction
```

The mechanical D/B gate passed: schedule, actual/forecast counts, transformer calls, fallbacks, causal PR #39 metrics, and ER-SDE exact tail were unchanged while only offline replay correction application changed.

Feature-space status:

```text
seed 1: replay correction rejected
seed 2: replay correction rejected
```

Perceptual status:

```text
seed 1 same-seed D/B: neutral / indistinguishable
seed 2: D exists; matching B not yet run
```

### Predeclared default-flip criterion

Do not renegotiate this after the result.

Flip the offline replay default away from transplanted generic correction only after **both**:

1. feature-space rejection reproduced on at least two independent 25-step seeds;
2. correction-disabled B is perceptually non-worse on at least two same-seed D/B comparisons.

Condition 1 is satisfied.

Condition 2 currently has one neutral/non-worse comparison and needs one additional comparison. One matching seed-2 run with:

```text
model_aware_replay_generic_correction=false
```

and otherwise identical seed/settings is sufficient to close the second perceptual D/B comparison.

If seed-2 B is neutral or better, the evidence-based default decision is to stop applying the replay scalar transfer by default, while documenting that hidden-feature accuracy improved without an obvious perceptual gain in the tested cases.

If seed-2 B is perceptually worse, keep the default and investigate the feature/perception mismatch.

No default changes in the current implementation iteration.

## Stream separation

Audio and video replay remain separate.

Audio replay has:

```text
spectral blend = 0
```

Therefore spectral-mixture and spectral-alpha diagnostics are **video-only by construction**. They must not touch audio tensors, audio weights, or audio replay behavior.

## Absolute video spectral result: seed 2

The direct target-withheld local -> full-spectral shadow removes the old validation-attenuated endpoint confound.

```text
local ratio                         0.728943
current attenuated blend ratio      0.734564
full spectral ratio                 1.039023
oracle ratio                        0.719751

oracle advantage vs local          +1.2498%
oracle advantage vs current        +2.0028%

absolute oracle spectral weight
  mean                              0.127587
  min                               0.000000
  max                               0.453452

current projected spectral weight
  mean                              0.257248
  min                               0.197381
  max                               0.402140
```

Full spectral is clearly bad. Useful spectral contribution is small and target-dependent.

### Fixed absolute sweep: rejected on seed 2

```text
absolute weight 0.00    ratio 0.728943
absolute weight 0.25    ratio 0.741953
absolute weight 0.50    ratio 0.806428
absolute weight 0.75    ratio 0.909442
absolute weight 1.00    ratio 1.039023
```

Pure local is the best fixed candidate on this trace.

The direct per-target oracle still improves `0.728943 -> 0.719751`, leaving only:

```text
+1.2498% vs local
+2.0028% vs current attenuated blend
```

on this spectral axis. The alpha experiment is therefore model-selection/diagnostic work, not a production-quality candidate.

### Seed-2 predictor relationships and n=11 caution

```text
current projected weight -> oracle weight                  +0.916659

causal disagreement -> oracle weight                       +0.853961
causal disagreement -> required adjustment                 +0.868151

validation penalty -> oracle weight                        -0.854319
validation penalty -> required adjustment                  -0.687826

spectral gap -> oracle weight                              +0.552068
spectral gap -> required adjustment                        +0.559945

coordinate -> oracle weight                                +0.646567
coordinate -> required adjustment                          +0.701142
```

There are only 11 sequential trajectory targets. Pairwise Pearson correlations are therefore descriptive diagnostics, not formal IID significance tests.

The current-weight/oracle, causal-disagreement/oracle, and validation-penalty/oracle relationships are strong enough to remain headline candidate relationships on this discovery trace. The spectral-gap values around `0.55` are **suggestive/marginal**, not established. Coordinate remains an explicit confound.

Seed 2 is the **discovery/development trace** for multiplicative alpha scaling. It is not independent confirmation of `alpha ~= 0.5`.

## Shadow-only multiplicative spectral recalibration

No production replay mechanism changes.

The predeclared sweep is fixed at:

```text
alpha = 0.00
        0.25
        0.50
        0.75
        1.00
```

No `1.25` control is included: the discovery trace already indicates over-scaling, and a >1 one-sided control would not materially improve this diagnosis.

The candidate rescales the existing validation-attenuated replay blend displacement:

```text
candidate_alpha =
    local + alpha * (current_uncorrected_blend - local)
```

Therefore:

```text
alpha=0 -> exactly local
alpha=1 -> exactly current uncorrected validation-attenuated blend
```

For weight-mechanism telemetry, the corresponding projected absolute weight is:

```text
w_alpha = clamp(alpha * w_current_projection, 0, 1)
```

This preserves the production branch-specific validation attenuation in the shadow candidate while comparing its projected absolute placement to the direct local -> full-spectral oracle weight.

Top-level telemetry:

```text
model_aware_trust_replay_spectral_alpha=
    video_current_blend_multiplicative_shadow
model_aware_trust_replay_spectral_alpha_applied=0
model_aware_trust_replay_spectral_alpha_sweep=0p00_0p25_0p50_0p75_1p00
model_aware_trust_replay_spectral_alpha_selection=
    lowest_mean_hidden_feature_ratio_then_lower_alpha
```

No alpha user setting exists.

### Alpha selection rule

One alpha is selected for all model-selection diagnostics:

1. compute mean hidden-feature ratio for each predeclared alpha;
2. select the alpha with the **lowest mean hidden-feature ratio**;
3. if exact means tie, select the lower alpha.

Weight MAE/RMSE and residual correlations do not select different alphas.

### Headroom-capture definition

Using aggregate mean ratios:

```text
captured_fraction =
    (mean_local_ratio - mean_alpha_ratio)
    /
    (mean_local_ratio - mean_oracle_ratio)
```

If:

```text
mean_local_ratio - mean_oracle_ratio <= 1e-12
```

the result is marked non-evaluable and reported as zero rather than dividing by a tiny denominator.

The fraction is not artificially clamped; negative capture or capture above one remains visible.

### Floor diagnostic

Near-zero oracle spectral weight is predeclared as:

```text
oracle_weight <= 1e-3
```

Telemetry reports:

```text
near-zero count/fraction
near-zero current projected weight mean
near-zero alpha-scaled weight mean
near-zero absolute weight error mean
near-zero contribution to total MAE
near-zero contribution to total squared error
```

This distinguishes a globally over-scaled placement from a remaining floor/gating problem. No floor gate is implemented.

### Alpha telemetry

For every alpha:

```text
model_aware_trust_replay_spectral_alpha_video_{alpha}_ratio_mean
model_aware_trust_replay_spectral_alpha_video_{alpha}_advantage_vs_local_mean
model_aware_trust_replay_spectral_alpha_video_{alpha}_advantage_vs_current_mean
model_aware_trust_replay_spectral_alpha_video_{alpha}_advantage_vs_oracle_mean
model_aware_trust_replay_spectral_alpha_video_{alpha}_headroom_capture_evaluable
model_aware_trust_replay_spectral_alpha_video_{alpha}_headroom_capture_fraction
model_aware_trust_replay_spectral_alpha_video_{alpha}_weight_mean
model_aware_trust_replay_spectral_alpha_video_{alpha}_weight_min
model_aware_trust_replay_spectral_alpha_video_{alpha}_weight_max
model_aware_trust_replay_spectral_alpha_video_{alpha}_weight_mae
model_aware_trust_replay_spectral_alpha_video_{alpha}_weight_rmse
model_aware_trust_replay_spectral_alpha_video_{alpha}_weight_bias
```

where `{alpha}` is `0p00`, `0p25`, `0p50`, `0p75`, or `1p00`.

The selected alpha additionally reports residual correlations for:

```text
causal disagreement
validation penalty
replay spectral gap
coordinate
current projected weight
```

Raw correlations remain visible regardless of outcome.

### Predeclared alpha model-selection gates

A simple multiplicative recalibration is strongly supported only if the same predeclared rule satisfies all of these on a development trace and a future independent confirmation trace:

1. selected alpha captures at least 60% of oracle-vs-local hidden-feature headroom on both evaluable traces;
2. best predeclared alpha agrees within `+/-0.25` across the two traces;
3. oracle-weight MAE/RMSE materially improves relative to `alpha=1/current`;
4. residual structure does not replicate against a pure global-alpha explanation.

Residual structure uses a replication-based interpretation because each trace has only `n=11` sequential targets:

```text
DISCOVERY TRACE
|r| < 0.5          weak residual structure
0.5 <= |r| < 0.6  indeterminate / noise-floor region
|r| >= 0.6         candidate residual structure only
```

A residual predictor counts as evidence **against** pure global alpha only if the **same predictor** shows:

```text
|r| >= 0.6
```

with the **same sign** on both discovery and independent confirmation traces.

A single `r=0.52`, `r=0.57`, or even one isolated `|r|>=0.6` does not reject global recalibration or establish a reusable controller signal. Cross-seed same-sign replication is required.

These thresholds are research/model-selection gates, not perceptual-quality thresholds.

Even if all alpha model-selection gates pass, do not apply alpha to production immediately. The absolute spectral headroom on the discovery trace is only about 1-2%, so any applied gate must be reassessed after the causal C1 perceptual result.

## Retrospective alpha scoring status

### Original seed 1

```text
retrospectively scorable: NO
```

It predates the absolute spectral-mixture shadow. The available benchmark/log material contains aggregate replay/decomposition telemetry, not the per-target current blend placements and withheld-target candidate scores required to evaluate the alpha family.

### Seed 2 / current discovery trace

```text
retrospectively scorable from available artifacts: NO
```

The absolute spectral shadow existed and produced the aggregate values above, but its per-target LOO tensors/records were transient runtime/archive state. The available log emits aggregate means/correlations/fixed-absolute-weight sweep only. Those aggregates cannot reconstruct the new per-target multiplicative candidates or their weight errors without fabricating information.

Therefore no retrospective alpha ratios, best alpha, MAE/RMSE, floor contribution, or residual-alpha correlations are claimed for either prior seed.

The existing seed-2 D output remains useful. When the matching seed-2 B run is eventually performed to close the D/B default criterion, the new alpha shadow can score that same deterministic seed in parallel without changing the B production mechanism; this can supply the development alpha telemetry without requiring a separate extra seed-2 generation.

A later different fixed seed remains the independent alpha confirmation trace if the program still warrants it.

## Immediate perceptual priority: causal C1

The next expensive user-facing research gate is **C1**, before seed-3 replay-alpha confirmation or any applied replay spectral controller.

Reason: causal trust has much larger hidden-feature headroom than the remaining replay spectral axis and has never received a direct same-seed perceptual A/B.

C1 tests single-pass causal behavior only:

```text
offline_smoothing_replay=false
sampler=sample_er_sde
steps=25
model_aware_mode=full
model_aware_risk_threshold=0.65
```

Baseline:

```text
model_aware_trust_shrinkage=false
```

Candidate:

```text
model_aware_trust_shrinkage=true
```

Keep the same seed, prompt, references, resolution, duration/frame count, scheduler, CFG, model/precision, and every other workflow setting.

The existing single-pass implementation and telemetry are sufficient; no new C1 algorithm or telemetry path is required.

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

Also verify both runs retain the same:

```text
actual_steps=14
forecast_steps=11
actual_transformer_calls=14
fallbacks=0
ER-SDE final steps 23/24 exact
```

and that causal PR #39 generic-correction evidence remains otherwise unchanged. Offline replay must not run.

C1 interpretation is perceptual, not a numeric percentage:

```text
positive
neutral / indistinguishable
negative
```

- Positive: validate causal single-pass trust on another case before attempting replay control; do not transfer it into replay.
- Neutral: strongly deprioritize increasingly elaborate trust/mixing work; the alpha shadow may remain scientific documentation.
- Negative: stop applied causal trust promotion and investigate the feature/perception mismatch.

A neutral C1 would weaken the expected perceptual value of this trust/feature-error-control program; it would not prove all Spectrum quality work is worthless.

## Priority of future real runs

Do not require all at once.

1. **C1 causal single-pass same-seed A/B**: replay off, trust false vs true.
2. **Close D/B default criterion when convenient**: use existing seed-2 D output and run one matching seed-2 B with replay generic correction disabled. The alpha shadow can collect development alpha telemetry during that run.
3. **Only if still warranted**: run a different fixed seed as independent alpha confirmation with replay on and the alpha diagnostics still shadow-only.

No dynamic spectral controller, positive replay correction, floor gate, new alpha user setting, scheduling/risk change, extra transformer evaluation, hold-anchor trust, K=2, FinalLayer correction, previous-error family, or ER-SDE tail change is introduced here.

`model_aware_trust_shrinkage` remains **false by default**. `model_aware_replay_generic_correction` remains **true by default**. PR #45 remains **draft**.
