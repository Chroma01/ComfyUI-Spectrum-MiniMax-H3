# Forecast trust-region benchmark record

This document records the forecasting-quality investigation that follows PR #39.

## Starting point

PR #39 established one useful generic correction mechanism on real MiniMax-H3:

```text
d = latest_exact - previous_exact
r = actual - forecast_uncorrected
g = projection of r onto d
corrected = forecast + g * d
```

The scalar gain is confidence-scaled and bounded. The final same-seed PR #39 gate reduced the measured hidden-feature forecast ratio by approximately **6.2% for audio** and **5.5% for video** relative to the uncorrected forecast. These are hidden-feature forecast-error improvements. They are not perceptual-quality percentages.

PR #39 retired K=2 causal trajectory correction, FinalLayer-transformed correction directions, previous-error directions, and related model-specific correction geometry after real H3 tests failed to materially beat the scalar latest-delta baseline.

PR #45 investigates forecast trust distance while preserving the successful PR #39 generic correction.

## Causal trust observer

The causal shadow probe compares the global Chebyshev forecast with the local linear/secant forecast on existing bounded per-stream evidence samples:

```text
p_spectral = global Chebyshev forecast
p_linear   = local linear/secant forecast

disagreement =
    RMS(p_spectral - p_linear)
    / max(RMS(p_spectral), epsilon)
```

The calibrated causal trust segment is:

```text
p_trust = latest_exact + kappa * (p_corrected - latest_exact)

kappa =
    exp(-0.3 * max(horizon - 1, 0))
    * sigmoid(4.0 * (theta - disagreement))
```

The fixed applied causal candidate uses `theta=0.15`. The ordinary shadow sweep also reports `theta=0.25` and `theta=0.40`.

## Causal calibration evidence

### Normal accelerated 25-step ER-SDE trace

```text
sampler                         sample_er_sde
steps                           25
actual_steps                    14
forecast_steps                  11
actual_transformer_calls        14
model_aware_extra_nfes          0
trust_probe_samples             12 audio / 12 video
trust_probe_horizon_mean        1.833333 audio / 1.833333 video
```

Current PR #39 correction:

```text
                              audio       video
raw forecast ratio            1.650739    1.307939
current corrected ratio       1.542742    1.240801
```

Causal trust oracle and fixed `theta=0.15`:

```text
                              audio       video
oracle ratio                  0.987258    0.998710
oracle kappa mean             0.089179    0.035404
oracle advantage             +34.9660%   +19.3114%
theta=0.15 ratio              0.990551    1.005021
theta=0.15 advantage         +34.7545%   +18.7989%
```

### Direct horizon-1 calibration

A single 25-step `model_aware_risk_threshold=0.0` trace intentionally paid for exact targets at every logical step:

```text
actual_steps                    25
forecast_steps                  0
actual_transformer_calls        25
model_aware_extra_nfes          22
first-pass sampler wall time    352.156 s
trust_probe_samples             23 audio / 23 video
trust_probe_horizon             exactly 1
```

Results:

```text
                              audio       video
current corrected ratio       1.456770    1.204774
oracle ratio                  0.987952    0.999173
oracle kappa mean             0.075656    0.015570
oracle advantage             +31.3663%   +17.0017%
theta=0.15 ratio              1.006423    1.021728
theta=0.15 advantage         +30.1357%   +15.1502%
```

Disagreement correlations:

```text
                              audio       video
error correlation            +0.768393   -0.118432
required-shrink correlation  +0.766415   +0.347676
```

The causal trust-distance result remains strong. The existing observer is especially strong for audio. Video remains less clean and is not treated as universally calibrated from this sample count.

No additional dense `risk_threshold=0.0` calibration is currently justified.

## Offline replay data-flow audit

The default workflow uses:

```text
offline_smoothing_replay=true
```

The first pass is deliberately local-only:

```text
causal_video_blend_weight=0.0
causal_audio_blend_weight=0.0
model_aware_causal_correction_s=0.0
```

The first-pass skipped feature drives the sampler state and therefore determines later exact anchors. `OfflineSmoother` subsequently rebuilds forecast weights from the full retained exact-anchor archive. It combines future-bracketed local interpolation with a spectral proposal, applies validation attenuation, and applies the persisted PR #39 correction.

The causal trust calibration and the offline smoother therefore operate on different proposal geometries.

## Real applied 25-step run: mechanical PASS, replay-transfer rejection

The first applied real run used:

```text
sampler = sample_er_sde
steps = 25
model_aware_mode = full
model_aware_risk_threshold = 0.65
model_aware_trust_shrinkage = true
offline_smoothing_replay = true
```

The generated output looked fine in this run. This run is not an artifact-fix test. ER-SDE exact-tail handling is already a separate solved path.

### Mechanics

The ordinary accelerated schedule was preserved exactly:

```text
actual_steps                    14
forecast_steps                  11
actual_transformer_calls        14
forecast_calls                  11
```

No extra transformer work:

```text
model_aware_extra_nfes                  0
model_aware_trust_extra_transformer_nfe 0
```

No failures:

```text
fallbacks                                  0
model_aware_failures                       0
trust_probe_failures                       0
model_aware_trust_failures                 0
model_aware_trust_replay_shadow_failures   0
```

ER-SDE exact tail remained intact:

```text
step 23 = actual, reason=final actual tail
step 24 = actual, reason=final actual tail
```

The causal-kappa replay transfer was genuinely active in this historical run:

```text
model_aware_trust_enabled=1
model_aware_trust_applied=1
model_aware_trust_path=offline_replay_causal_kappa_transfer
model_aware_trust_applications=20
```

Controller cost:

```text
model_aware_trust_compute_s          0.071898
model_aware_trust_scalar_transfer_s  0.004118
model_aware_trust_weight_apply_s     0.000338
model_aware_trust_total_s            0.076354
```

The scheduling, NFE accounting, exact tail, failure behavior, and controller overhead all passed.

### Causal trust reproduced

The ordinary causal shadow result reproduced the earlier 25-step trace:

```text
                              audio       video
current corrected ratio       1.542742    1.240801
oracle ratio                  0.987258    0.998710
oracle kappa mean             0.089179    0.035404
oracle advantage             +34.9660%   +19.3114%
theta=0.15 ratio              0.990551    1.005021
theta=0.15 advantage         +34.7545%   +18.7989%
```

### Replay-transfer shadow rejected the applied mapping

The leave-one-out replay diagnostic reported:

```text
model_aware_trust_replay_shadow=loo_unattenuated_future_bracket
```

Audio:

```text
samples                         11
replay baseline ratio           0.739193
causal-kappa transfer ratio     0.969816
relative advantage             -32.2191%
```

Video:

```text
samples                         11
replay baseline ratio           0.795838
causal-kappa transfer ratio     0.967608
relative advantage             -22.1356%
```

The persisted causal coefficients were small:

```text
                              audio       video
disagreement mean             0.567507    0.512896
kappa mean                    0.170582    0.201150
kappa min                     0.073230    0.100513
kappa max                     0.362033    0.367816
```

The future-bracketed replay proposal already scored well below the normalized hold-anchor error of `1.0`. Applying the causal coefficient retained only about 17-20% of that replay proposal and moved the result near the hold anchor. The replay-specific shadow therefore rejects the direct causal-kappa transfer.

## Architectural conclusion

Two separate findings are now established.

### Causal geometry

```text
current corrected causal proposal:
    audio 1.542742
    video 1.240801

theta=0.15 causal trust:
    audio 0.990551
    video 1.005021
```

Trust distance is a validated causal failure axis on the measured traces.

### Offline replay geometry

```text
existing replay proposal:
    audio 0.739193
    video 0.795838

causal-kappa transfer:
    audio 0.969816
    video 0.967608
```

The causal coefficient does not transfer directly to the future-bracketed replay proposal.

## Revised implementation semantics

### Single-pass mode

With `offline_smoothing_replay=false`, the calibrated causal mechanism remains applied:

```text
model_aware_trust_path=causal_single_pass
```

The existing model-aware corrected history weights are shrunk toward the latest causal exact anchor using the causal `kappa`.

### Default offline replay mode

With `offline_smoothing_replay=true` and trust requested:

```text
model_aware_trust_path=offline_replay_shadow_only
model_aware_trust_applied=0
model_aware_trust_applications=0
model_aware_trust_replay_application=disabled_rejected_causal_transfer
```

The causal trust calculation remains available for telemetry and for comparison with replay-native results. The causal coefficient no longer modifies offline replay weights.

The existing PR #39 generic replay correction remains active.

Replay-native investigation is shadow-only:

```text
model_aware_trust_replay_shadow=loo_unattenuated_replay_native_calibration
```

No replay controller is applied at this stage.

## Replay-native shadow questions

The next ordinary accelerated run answers five questions:

1. Does the future-bracketed replay proposal have shrinkage oracle headroom?
2. If it does, how large is the required shrinkage?
3. Does persisted causal disagreement predict replay error or required replay shrinkage?
4. Does a replay-native spectral-vs-local disagreement predict replay error or required shrinkage?
5. Does the latest causal anchor remain a useful endpoint relative to the replay-local interpolation endpoint?

## Replay-native oracle

For each eligible withheld exact target:

```text
h = withheld exact target
p = replay proposal constructed without h
a = persisted latest causal exact anchor available at original forecast time
u = p - a

kappa_replay_oracle =
    clamp(<h - a, u> / <u, u>, 0, 1)

p_oracle = a + kappa_replay_oracle * u
```

The implementation reports for audio/video:

```text
model_aware_trust_replay_shadow_{stream}_oracle_ratio_mean
model_aware_trust_replay_shadow_{stream}_oracle_advantage_mean
model_aware_trust_replay_shadow_{stream}_oracle_kappa_mean
model_aware_trust_replay_shadow_{stream}_oracle_kappa_min
model_aware_trust_replay_shadow_{stream}_oracle_kappa_max
```

`kappa=1` is the current replay baseline. Small oracle advantage with kappa near `1` means replay shrinkage is largely exhausted on this segment. Material oracle advantage establishes headroom for replay-specific calibration.

## Fixed replay-kappa sweep

The shadow-only sweep evaluates:

```text
kappa = 0.50
kappa = 0.70
kappa = 0.85
kappa = 0.95
kappa = 1.00
```

For every coefficient:

```text
model_aware_trust_replay_shadow_{stream}_kappa_0p50_ratio_mean
model_aware_trust_replay_shadow_{stream}_kappa_0p50_advantage_mean
...
model_aware_trust_replay_shadow_{stream}_kappa_1p00_ratio_mean
model_aware_trust_replay_shadow_{stream}_kappa_1p00_advantage_mean
```

The `1.00` candidate must reproduce the replay baseline exactly up to floating-point tolerance.

These coefficients are telemetry only. They are not user controls.

## Does the causal observer transfer?

The persisted causal disagreement is correlated separately with replay baseline error and required replay shrinkage:

```text
model_aware_trust_replay_shadow_{stream}_causal_disagreement_error_corr
model_aware_trust_replay_shadow_{stream}_causal_disagreement_shrink_corr
```

The shrink target is:

```text
1 - kappa_replay_oracle
```

These metrics test observer transfer directly without assuming the causal `kappa` mapping transfers.

## Replay-native observer

`OfflineSmoother` already has a future-bracketed local interpolation branch and a spectral branch. The replay-native shadow uses their bounded sampled proposals directly:

```text
r_replay =
    RMS(p_spectral - p_local)
    / max(RMS(p_replay_baseline), eps)
```

The observer is enabled only when the configured replay stream has a nonzero spectral branch. A stream with zero spectral blend does not fabricate a spectral-vs-local observer.

Telemetry:

```text
model_aware_trust_replay_shadow_{stream}_observer
model_aware_trust_replay_shadow_{stream}_observer_samples
model_aware_trust_replay_shadow_{stream}_replay_disagreement_mean
model_aware_trust_replay_shadow_{stream}_replay_disagreement_max
model_aware_trust_replay_shadow_{stream}_replay_disagreement_error_corr
model_aware_trust_replay_shadow_{stream}_replay_disagreement_shrink_corr
```

For the observed real run, audio had configured/effective spectral blend zero, so a replay spectral-vs-local observer is intentionally absent for audio. Video has a meaningful spectral replay branch and receives the replay-native observer.

## Endpoint audit

The shadow records the uncorrected replay-local interpolation ratio:

```text
model_aware_trust_replay_shadow_{stream}_local_ratio_mean
```

It also evaluates the best interpolation coefficient on the segment from replay-local interpolation to the corrected replay proposal:

```text
model_aware_trust_replay_shadow_{stream}_local_oracle_ratio_mean
model_aware_trust_replay_shadow_{stream}_local_oracle_advantage_mean
model_aware_trust_replay_shadow_{stream}_local_oracle_kappa_mean
model_aware_trust_replay_shadow_{stream}_local_oracle_kappa_min
model_aware_trust_replay_shadow_{stream}_local_oracle_kappa_max
```

This gives a cheap replay-native endpoint audit using candidates the smoother already computes. The withheld target is used only for shadow scoring and oracle metrics. It never enters live controller state or the retained LOO forecaster history.

The previous/next replay bracket is used to construct the local future-bracketed proposal. The latest causal anchor remains a separately persisted causal endpoint. The corrected future-bracketed proposal remains the replay baseline. No new live endpoint is selected from this telemetry.

## Direct-transfer counterfactual remains visible as shadow telemetry

For continuity with the rejected run, the diagnostic still reports the counterfactual causal-kappa transfer without applying it:

```text
model_aware_trust_replay_shadow_{stream}_baseline_ratio_mean
model_aware_trust_replay_shadow_{stream}_causal_transfer_ratio_mean
model_aware_trust_replay_shadow_{stream}_causal_transfer_advantage_mean
```

This is diagnostic evidence only.

## Safety invariants

The revised path preserves these invariants:

- `model_aware_trust_shrinkage=false` keeps existing workflow behavior;
- single-pass causal trust remains unchanged;
- default offline first-pass local-only capture remains unchanged;
- offline replay no longer applies causal `kappa`;
- replay-native logic is shadow-only;
- ordinary model-aware scheduling is unchanged;
- no hard refresh/re-pay is added;
- no transformer evaluation is added;
- ER-SDE exact two-logical-step tail remains intact;
- exact replay anchors remain exact;
- target audio/video streams remain independent;
- packed topology does not fabricate a modality split;
- LOO targets are excluded from the retained shadow forecaster history;
- future exact targets are used only for offline scoring/oracle telemetry;
- non-OOM diagnostic failures do not abort replay;
- CUDA OOM propagates;
- replay shadow state is archive-scoped and resets with a new capture.

## Performance accounting

The real applied run measured causal trust-controller cost at approximately:

```text
0.076354 s
```

The replay-native LOO calibration runs during offline smoother construction and is reported separately:

```text
model_aware_trust_replay_shadow_compute_s
```

It is not added to `model_aware_trust_total_s`, which continues to represent the causal controller computation/synchronization/weight-application path.

The existing generic model-aware evidence cost from the real run was:

```text
model_aware_overhead_s=3.648930
model_aware_evidence_s=3.647071
model_aware_evidence_device_transfer_s=3.024835
```

That transfer cost remains a separate optimization topic.

## Next real gate

Use one ordinary accelerated run:

```text
steps = 25
sampler = native sample_er_sde
model_aware_mode = full
model_aware_risk_threshold = 0.65
model_aware_trust_shrinkage = true
offline_smoothing_replay = true
```

Keep the same workflow, seed, prompt, checkpoint, precision, references, resolution, frame count, scheduler, CFG, and Spectrum settings.

Do not run a dense force-actual calibration.

Expected mechanics:

```text
14 actual / 11 forecast
zero extra transformer NFE
zero trust failures
ER-SDE exact tail
model_aware_trust_path=offline_replay_shadow_only
model_aware_trust_applied=0
model_aware_trust_applications=0
```

Inspect the replay oracle, fixed-kappa sweep, causal-disagreement correlations, replay-native disagreement correlations, and endpoint audit fields listed above.

## Stop point

After the replay-native shadow data is collected, choose the next step from the measured result. No new replay controller is applied in this revision.

`model_aware_trust_shrinkage` remains default-off. PR #45 remains draft. A green CI run establishes implementation correctness for the covered cases; it does not establish perceptual-quality gain.
