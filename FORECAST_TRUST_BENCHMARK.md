# Forecast trust-region benchmark record

This document records the forecasting-quality investigation that follows PR #39.

## Starting point: the PR #39 breach

PR #39 established one useful generic correction mechanism on real MiniMax-H3:

```text
d = latest_exact - previous_exact
r = actual - forecast_uncorrected
g = projection of r onto d
corrected = forecast + g * d
```

The scalar gain is confidence-scaled and bounded. On the final same-seed PR #39 gate, this reduced the measured hidden-feature forecast ratio by approximately **6.2% for audio** and **5.5% for video** relative to the uncorrected forecast. These are hidden-feature forecast-error improvements, not literal perceptual-quality percentages.

PR #39 also retired K=2 causal trajectory correction, FinalLayer-transformed correction directions, previous-error directions, and related model-specific correction geometry after real H3 tests failed to materially beat the scalar latest-delta baseline. PR #45 keeps that successful local breach and asks how far an already-corrected forecast should be trusted away from the latest exact feature.

## Shadow observer

The shadow probe compares the global Chebyshev forecast with the local linear/secant forecast on the existing bounded per-stream evidence samples:

```text
p_spectral = global Chebyshev forecast
p_linear   = local linear/secant forecast

disagreement =
    RMS(p_spectral - p_linear)
    / max(RMS(p_spectral), epsilon)
```

The investigated trust segment is:

```text
p_trust = latest_exact + kappa * (p_corrected - latest_exact)

kappa =
    exp(-0.3 * max(horizon - 1, 0))
    * sigmoid(4.0 * (theta - disagreement))
```

The shadow sweep evaluates `theta = 0.15`, `0.25`, and `0.40`. The probe remains explicitly diagnostic:

```text
trust_probe=shadow_only
trust_probe_applied=0
trust_probe_extra_transformer_nfe=0
```

The applied experiment has separate `model_aware_trust_*` telemetry. `trust_probe_applied` is not overloaded.

## Real trace 1: normal accelerated 25-step ER-SDE run

```text
sampler                         sample_er_sde
steps                           25
actual_steps                    14
forecast_steps                  11
actual_transformer_calls        14
model_aware_extra_nfes          0
trust_probe_failures            0
trust_probe_samples             12 audio / 12 video
trust_probe_horizon_mean        1.833333 audio / 1.833333 video
trust_probe_horizon_max         2.0 audio / 2.0 video
```

Current PR #39 correction and remaining same-direction scalar headroom:

```text
                              audio       video
current corrected ratio      1.542742    1.240801
bounded-delta ratio          1.484296    1.194263
bounded-delta advantage      3.6977%     3.7016%
```

Trust-segment oracle:

```text
                              audio       video
oracle trust ratio           0.987258    0.998710
oracle kappa                 0.089179    0.035404
oracle relative advantage    34.9660%    19.3114%
```

Disagreement observer:

```text
                              audio       video
disagreement mean            0.721785    0.670032
error correlation            0.859295    0.559713
required-shrink correlation  0.907817    0.751129
```

Fixed `theta=0.15`:

```text
                              audio       video
ratio                         0.990551    1.005021
relative advantage            34.7545%    18.7989%
```

The first trace showed much more trust-segment headroom than remaining scalar-gain headroom. Its probe targets were mostly horizon 2, while the live ER-SDE forecasts were horizon 1, so a direct horizon-1 calibration was required.

## Real trace 2: completed direct horizon-1 calibration

The second trace used the same 25-step configuration with:

```text
model_aware_risk_threshold = 0.0
```

This intentionally converted every prospective model-aware forecast into an exact transformer evaluation:

```text
actual_steps                    25
forecast_steps                  0
actual_transformer_calls        25
model_aware_extra_nfes          22
first-pass sampler wall time    352.156 s
```

The long runtime was expected. This was not a cold-start or profile-cache failure:

```text
model_aware_profile_cache_hit=True
model_aware_profile_lookup_s=0.000137
```

The calibration deliberately paid for exact targets at all 25 logical steps. Do not repeat this dense run unless a concrete validity problem is found.

Direct horizon-1 coverage:

```text
trust_probe_audio_samples       23
trust_probe_video_samples       23
trust_probe_audio_horizon_mean  1.000000
trust_probe_video_horizon_mean  1.000000
trust_probe_audio_horizon_max   1.000000
trust_probe_video_horizon_max   1.000000
```

Current correction and remaining scalar-gain headroom:

```text
                              audio       video
current corrected ratio      1.456770    1.204774
bounded-delta gain mean     -0.206311   -0.187437
bounded-delta ratio          1.405965    1.166521
bounded-delta advantage      3.3762%     3.1744%
```

Trust-segment oracle:

```text
                              audio       video
oracle trust ratio           0.987952    0.999173
oracle kappa mean            0.075656    0.015570
oracle relative advantage    31.3663%    17.0017%
```

Fixed transfer candidates at the live horizon:

```text
                              audio       video
theta=0.15 ratio             1.006423    1.021728
theta=0.15 advantage         30.1357%    15.1502%
theta=0.25 advantage         29.0662%    14.2591%
theta=0.40 advantage         26.2317%    12.3941%
```

Disagreement signal:

```text
                              audio       video
disagreement mean            0.541178    0.517512
disagreement max             0.903309    0.927804
error correlation           +0.768393   -0.118432
required-shrink correlation +0.766415   +0.347676
```

`theta=0.15` is the strongest fixed tested candidate on both real traces and remains strongly positive at direct horizon 1. Audio disagreement is a strong observer in this calibration. Video error correlation is weak/negative, while video required-shrink correlation remains positive and `theta=0.15` is beneficial in aggregate. The video observer is not treated as universally calibrated, and separate audio/video thresholds are not fitted from only 23 samples.

## Mandatory applied-data-flow audit

The default workflow uses `offline_smoothing_replay=true`. The audit found two distinct forecasting geometries.

### Offline first pass is deliberately local-only

During `offline_first_pass`, `_causal_prediction_blends()` returns zero audio and video spectral blend. The normal model-aware weighted-segment path is bypassed while `_offline_phase` is active. The first pass therefore uses the local causal forecast for skipped features:

```text
causal_video_blend_weight=0.0
causal_audio_blend_weight=0.0
model_aware_causal_correction_s=0.0
```

A naive patch to the ordinary model-aware weighted-segment function would be a no-op for the default offline first pass.

### PR #39 correction is applied in offline replay construction

The first pass still computes and archives each model-aware decision. `OfflineSmoother` later rebuilds forecast weights from all retained exact anchors, combines global spectral weights with future-bracketed local interpolation, and applies the persisted PR #39 scalar correction. The replay correction is applied to replay weights, not to the local-only first-pass skipped feature.

For the generic scalar correction, the current replay fallback applies the gain across the bracketing replay anchors. That geometry differs from the causal latest-two-exact-anchor correction used by the live single-pass forecaster and by the shadow calibration.

### First-pass forecasts still matter to final replay

The local-only first-pass skipped features drive the sampler state. Subsequent exact transformer evaluations are exact features on that first-pass trajectory and become the retained offline anchors. Replay restarts from the same original inputs and random stream, then substitutes archived exact anchors or future-bracketed smoothed features. A first-pass change can therefore alter later anchors and the final replay output, while also changing the established local-only capture architecture.

### Existing shadow `p_corrected` is counterfactual in default offline mode

The shadow probe reconstructs the configured/adaptive causal spectral-linear proposal plus the PR #39 causal latest-delta correction. It is not the actual local-only first-pass skipped feature, the stored model-aware decision itself, or the future-bracketed corrected replay proposal. The shadow result is valid evidence for the causal trust mechanism; it is not a direct measurement of the current replay proposal.

## Applied integration decision

The earlier benchmark plan said to change causal first-pass weights only. The audit invalidated that plan as a clean A/B. Activating the counterfactual model-aware corrected causal proposal in the first pass would simultaneously change the local-only capture forecast, PR #39 correction placement, and the new trust coefficient. That would confound the trust experiment.

The smallest baseline-preserving applied experiment is therefore:

### Single-pass mode

When offline replay is disabled, apply the calibrated causal mechanism directly:

```text
w_corrected =
    existing model-aware history weights
    + existing PR #39 latest-delta correction

w_final =
    kappa * w_corrected
    + (1 - kappa) * one_hot(latest_causal_anchor)
```

### Default offline replay mode

During the local-only first pass:

1. compute audio/video disagreement from existing bounded causal evidence;
2. compute fixed `theta=0.15` `kappa` independently for each stream;
3. persist the scalar `kappa` and exact causal anchor ID for that forecast step;
4. leave the first-pass skipped feature unchanged.

During `OfflineSmoother` weight construction:

```text
w_replay_corrected =
    existing future-bracketed replay weights
    + existing PR #39 replay correction

w_final =
    kappa_causal * w_replay_corrected
    + (1 - kappa_causal) * one_hot(persisted_latest_causal_anchor)
```

The replay proposal may contain future information. The trust coefficient and shrink target do not: they are computed causally during the first pass and persisted deterministically.

This path is reported as:

```text
model_aware_trust_path=offline_replay_causal_kappa_transfer
```

It is intentionally labeled a **transfer** because the replay proposal geometry is not identical to the causal proposal that produced the 30.1% / 15.2% horizon-1 shadow gains.

## Replay-transfer shadow validation

Because the default replay integration uses a different proposal geometry, the implementation adds a separate bounded leave-one-out replay-transfer diagnostic. At eligible exact anchors it records causal disagreement, fixed `theta=0.15` kappa, latest causal anchor ID, model-aware blend, and PR #39 scalar gain. Once the full exact-anchor archive exists, it withholds that target and constructs a future-bracketed replay-style proposal from bounded sampled features.

It reports:

```text
model_aware_trust_replay_shadow=loo_unattenuated_future_bracket
model_aware_trust_replay_shadow_audio_*
model_aware_trust_replay_shadow_video_*
```

The diagnostic intentionally omits the live smoother's branch-specific validation attenuation. It is a replay-geometry sanity check, not proof that the applied replay transfer is perceptually better. The same-seed generated-output A/B remains authoritative.

## Applied controller

Public option:

```text
model_aware_trust_shrinkage
```

Properties:

- boolean;
- default `false`;
- effective only with `model_aware_mode="full"`;
- no default behavior change;
- no scheduling change or hard refresh/re-pay;
- no additional transformer NFE;
- no full hidden-feature duplicate prediction;
- no persistent large tensor state.

Fixed first experimental mapping:

```text
kappa =
    exp(-0.3 * max(horizon - 1, 0))
    * sigmoid(4.0 * (0.15 - disagreement))
```

Audio and video are computed and applied separately. Packed topology without a proven target audio/video boundary remains baseline-identical. If causal evidence is insufficient, bootstrap-only, malformed, or otherwise cannot prove the observer, trust is not applied (`kappa=1` behavior). Non-OOM trust failures fall back to baseline for that stream/step and increment the failure counter. CUDA OOM propagates.

## Applied telemetry

```text
model_aware_trust_enabled
model_aware_trust_applied
model_aware_trust_path
model_aware_trust_applications

model_aware_trust_audio_disagreement_mean
model_aware_trust_audio_disagreement_max
model_aware_trust_video_disagreement_mean
model_aware_trust_video_disagreement_max

model_aware_trust_audio_kappa_mean
model_aware_trust_audio_kappa_min
model_aware_trust_audio_kappa_max
model_aware_trust_video_kappa_mean
model_aware_trust_video_kappa_min
model_aware_trust_video_kappa_max

model_aware_trust_failures
model_aware_trust_compute_s
model_aware_trust_scalar_transfer_s
model_aware_trust_weight_apply_s
model_aware_trust_total_s
model_aware_trust_extra_transformer_nfe=0
```

Audio/video disagreement scalars are reduced together before the GPU-to-CPU transfer. History weights are already bounded CPU coefficients, so shrinkage does not copy full hidden history to the host. `model_aware_trust_scalar_transfer_s` is the synchronization boundary; pending bounded GPU observer work may be charged there because GPU execution is asynchronous.

## Mechanical invariants

With `model_aware_trust_shrinkage=false`, behavior must remain baseline-identical. With it enabled:

- model-aware scheduling remains unchanged;
- ER-SDE's final two logical steps remain actual;
- no trust-added actual evaluation is allowed;
- the normal 25-step schedule should remain 14 actual / 11 forecast;
- `model_aware_trust_extra_transformer_nfe=0`;
- offline first-pass local-only capture remains unchanged;
- replay chronology and seeded replay remain unchanged;
- exact replay anchors remain exact one-hot anchors;
- output shape, dtype, and device remain unchanged;
- audio/video trust decisions remain independent;
- packed topology never fabricates a modality split.

## Real applied gate

Unit tests and CI are necessary and are not the promotion gate.

### Gate C1: 25-step same-seed applied A/B

Use the same normal accelerated 25-step workflow that produced the calibration traces.

Baseline:

```text
model_aware_mode = full
model_aware_trust_shrinkage = false
normal model_aware_risk_threshold
sampler = sample_er_sde
offline_smoothing_replay = unchanged
```

Applied: identical except `model_aware_trust_shrinkage=true`.

Keep seed, prompt, checkpoint, precision, references, resolution, frame count, scheduler, CFG, and all other Spectrum settings identical.

Mechanical checks:

- 14 actual / 11 forecast remains intact unless the baseline itself differs;
- no extra transformer NFE;
- no trust failures;
- ER-SDE exact tail preserved;
- offline replay completes with no fallback;
- trust applications are nonzero;
- audio/video kappa and disagreement are sensible;
- replay-transfer shadow telemetry is present where eligible;
- controller overhead is acceptable.

Then compare generated video and audio quality. This is the first gate that can establish whether the hidden-feature error mechanism improves actual output.

### Gate C2: known-good 20-step regression

If C1 is positive, repeat the known-good 20-step regime used around PR #39. The controller must not improve the problematic 25-step case by degrading the already-good 20-step case.

### Gate C3: another seed / prompt / reference case

Before any default promotion, test another independent content case. Only one direct horizon-1 calibration trace exists. Do not fit separate audio/video thresholds or repeatedly tune `theta` to one seed.

## Promotion rules

For now:

```text
model_aware_trust_shrinkage = false
```

Do not promote it to the default from sampled feature metrics alone. Do not merge PR #45 solely because unit tests and CI pass. Keep it draft until the generated-output applied gate is evaluated unless explicitly instructed otherwise.

If the applied A/B fails, distinguish observer quality, mapping calibration, audio/video asymmetry, replay-transfer geometry, incorrect integration, and hidden-feature error not mapping to final perceptual quality. Do not revive retired K=2 / FinalLayer / previous-error families without genuinely new evidence, and do not tune `theta` until one calibration seed happens to look good.
