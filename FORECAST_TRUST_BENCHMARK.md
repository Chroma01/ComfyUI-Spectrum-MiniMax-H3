# Forecast trust-region benchmark record

This document starts the next forecasting-quality investigation after PR #39.

## Starting point

PR #39 established one useful generic mechanism on real MiniMax-H3: a bounded scalar correction in the latest causal trajectory direction.

```text
d = h[-1] - h[-2]
r = h_actual - h_pred_uncorrected
g_raw = <r,d> / <d,d>
h_corrected = h_pred + g*d
```

On the final same-seed 20-step gate, that mechanism reduced the measured hidden-feature forecast ratio by approximately **6.2% for audio** and **5.5% for video** relative to the uncorrected forecast. This is a forecast-error result. It is not, by itself, a measured 5-6% perceptual video-quality result.

The same PR also established several useful negative results. Increasing the temporal correction rank, transforming the latest delta through FinalLayer geometry, and persisting the previous forecast residual all lost to the scalar latest-delta baseline. The next experiment therefore keeps the successful correction direction intact and attacks a different part of the same error: **how far the forecast should be trusted away from the latest exact anchor**.

## Hypothesis: cache-only predictor disagreement can control forecast trust

A recent closed-loop Spectrum extension, RACER (Li et al., arXiv:2608.01740), reports that disagreement between the global Chebyshev forecast and a local Taylor forecast is informative about forecast error. Its continuous controller shrinks an uncertain forecast toward the latest computed feature, and its separate refresh controller can relocate exact evaluations under a fixed NFE budget. The reported video experiments include Wan2.1-14B and HunyuanVideo.

This repository already has the ingredients needed for a lower-risk MiniMax-H3 test:

- the global Chebyshev predictor;
- the local linear/secant predictor used by Spectrum's existing blend;
- device-local sampled evidence for audio and video in `full` mode;
- the validated generic latest-delta correction from PR #39.

The first H3 experiment uses **unblended spectral-vs-linear disagreement** as the observer. It deliberately does not copy RACER's calibrated thresholds as MiniMax-H3 constants because those thresholds were fitted for other models and schedules.

## Shadow-only experiment

The first implementation changes no generated feature and no schedule decision. At each eligible exact anchor in `model_aware_mode=full`, it reconstructs the two cache-only sampled proposals:

```text
p_spectral = Chebyshev(history, target)
p_linear   = local_linear(history, target)

risk = RMS(p_spectral - p_linear) / max(RMS(p_spectral), eps)
```

The live PR-39 proposal remains:

```text
p = current blended Spectrum forecast + generic_latest_delta_correction
```

The probe then measures three questions against the exact anchor that is already being computed.

### 1. Is there still headroom in the successful latest-delta direction itself?

The existing anchor evidence already measures the instantaneous exact residual projection onto `d = latest - previous`. The probe passes that instantaneous coefficient through the same rational `0.25` magnitude bound used by the shipping generic correction and shadow-scores it:

```text
g_anchor = <actual - p_uncorrected, d> / <d,d>
g_bound = g_anchor / (1 + abs(g_anchor) / 0.25)
p_delta_bound = p_uncorrected + g_bound*d
```

`delta_bound_advantage_mean` compares this non-causal, same-anchor bounded candidate with the currently applied causal correction. A material gap means the PR-39 direction still has useful headroom and the next gain may come from estimating its scalar coefficient better rather than changing direction.

### 2. Does the trust segment contain additional error-reduction headroom?

Let `a` be the latest exact cached feature. The best diagnostic interpolation on the segment from `a` to the current corrected proposal `p` is:

```text
u = p - a
kappa_oracle = clamp(<h_actual-a,u> / <u,u>, 0, 1)
p_oracle = a + kappa_oracle*u
```

`oracle_advantage_mean` measures how much lower the sampled forecast ratio could become if a perfect trust coefficient were known. This does not leak into generation; it only answers whether trust shrinkage is worth pursuing after the existing scalar correction.

### 3. Does a cache-only disagreement signal point in the useful direction?

The probe records Pearson correlation between disagreement and:

- the current generic-corrected forecast ratio;
- the amount of oracle shrink required, `1 - kappa_oracle`.

It also shadow-scores three fixed transfer candidates using the recent closed-loop form:

```text
kappa = exp(-0.3 * max(horizon - 1, 0)) * sigmoid(4 * (theta - risk))
```

with `theta` values `0.15`, `0.25`, and `0.40`. These values are a diagnostic sweep only. None is applied to generated output.

## Mechanical invariants

This stage must preserve all behavior outside telemetry:

```text
trust_probe=shadow_only
trust_probe_applied=0
trust_probe_extra_transformer_nfe=0
```

Required invariants:

- identical actual/forecast logical step IDs to the same configuration on `main`;
- identical transformer NFE count;
- identical model-aware risk/confidence/ridge/degree/blend decisions;
- identical PR-39 generic correction gains;
- identical ER-SDE two-logical-step actual tail policy;
- identical offline replay decisions and generated output;
- no full hidden-feature duplicate forecast solely for disagreement;
- no FinalLayer operator materialization;
- no persistent large tensor state added by the probe.

The observer operates only on the existing bounded per-stream evidence samples. The three candidate scores are computed on those samples and cannot affect the live forecast.

## Real MiniMax-H3 gates

### Gate A: ordinary same-seed parity run

Use the same saved base-H3 workflow used for the PR-39 final gate:

```text
sampler: native sample_er_sde
steps: 20
model_aware_mode: full
same seed/checkpoint/precision/scheduler/prompt/references/resolution/frame count/CFG
same Spectrum settings and storage mode
debug: enabled
```

First verify the mechanical invariants above. The summary must report nonzero trust-probe samples and `trust_probe_failures=0` while the actual/forecast schedule and NFE remain identical to `main`.

### Exploratory result: supplied 25-step ordinary run

The first real trace was accidentally run at **25 steps**, so it is not the exact 20-step parity gate above. It is still valid mechanism evidence for the 25-step regime and contains more shadow samples.

First-pass schedule and probe integrity:

```text
sampler                         sample_er_sde
steps                           25
actual_steps                    14
forecast_steps                  11
actual_transformer_calls        14
model_aware_extra_nfes          0
trust_probe_failures            0
trust_probe_extra_transformer_nfe 0
trust_probe_samples             12 audio / 12 video
trust_probe_horizon_mean        1.833333 audio / 1.833333 video
trust_probe_horizon_max         2.0 audio / 2.0 video
```

The generic PR-39 correction still helps, and the same-direction oracle shows a further modest gain:

```text
                              audio       video
current corrected ratio      1.542742    1.240801
bounded-delta ratio          1.484296    1.194263
bounded-delta advantage      3.6977%     3.7016%
```

The trust segment exposes substantially larger headroom:

```text
                              audio       video
oracle trust ratio           0.987258    0.998710
oracle kappa                 0.089179    0.035404
oracle relative advantage    34.9660%    19.3114%
```

The cache-only disagreement is also informative in this trace:

```text
                              audio       video
disagreement mean            0.721785    0.670032
error correlation            0.859295    0.559713
required-shrink correlation  0.907817    0.751129
```

All three fixed trust candidates improve the sampled ratio at these evaluated anchors. `theta=0.15` is the strongest of the fixed sweep on both streams:

```text
                              audio       video
theta=0.15 ratio             0.990551    1.005021
theta=0.15 advantage         34.7545%    18.7989%
theta=0.25 advantage         34.4788%    18.4254%
theta=0.40 advantage         33.4501%    17.5175%
```

Two conclusions are justified from this trace:

1. The trust branch dominates the remaining same-direction scalar-gain headroom in this 25-step regime. The measured oracle advantage is roughly 9.5x the bounded-delta advantage for audio and 5.2x for video.
2. The current corrected sampled forecast ratio is above the hold baseline (`ratio=1`) on average at these evaluated exact anchors, while oracle shrinkage moves both streams back to approximately the hold-error level. This is strong evidence that forecast distance from the latest exact anchor is a real failure axis in this regime.

One limitation is decisive before applying a controller: these ordinary-run probe samples are mostly **horizon 2** (`horizon_mean=1.8333`), because the probe scores the next exact anchor from the preceding actual history. The live ER-SDE forecast decisions in this run are horizon 1. Therefore the excellent `theta=0.15` result is not yet a direct same-horizon validation of the coefficient that would be applied at the live forecast steps.

The 25-step run is therefore accepted as strong Branch-B evidence, not as the final applied-controller calibration and not as the exact 20-step parity gate.

### Gate B: direct horizon-1 calibration trace

The next run should keep the supplied **25-step configuration** fixed and turn would-be forecasts into exact evaluations so the probe sees ground truth at the same short horizon where an applied controller would act:

```text
model_aware_risk_threshold = 0.0
```

The existing force-actual rule then converts prospective model-aware forecasts into exact steps once the model-aware path is active. This run is intentionally slower and is for calibration evidence only; it is not a shipping configuration.

Required checks:

- `trust_probe_failures=0`;
- no probe-added transformer NFE beyond the exact evaluations intentionally caused by the threshold;
- `trust_probe_horizon_mean` should move close to `1.0` for the direct forecast locations;
- the fixed theta sweep must be evaluated again at that horizon;
- do not promote `theta=0.15` merely because it was best at the horizon-2-heavy ordinary trace.

If a fixed candidate remains materially positive for both streams at horizon 1, the next patch may add an **opt-in applied trust controller** for same-seed output A/B. The first applied experiment should change the causal first-pass forecast weights only and leave offline smoother weight construction unchanged until separate evidence justifies modifying that path. This isolates whether better causal anchors improve the final replay trajectory without conflating the result with a second change to offline interpolation.

After an applied 25-step A/B, repeat the good 20-step regime as a regression check before considering any default change.

## Promotion gate for the next applied mechanism

Do not alter the default `full` path from this probe alone. Promote an applied mechanism only after real H3 evidence resolves which branch has actual headroom at the horizon where the controller acts.

### Branch A: improve the existing residual gain

Prefer this branch when `delta_bound_advantage_mean` is materially positive and larger than the trust-segment advantage. That means the successful direction from PR #39 remains useful and the limiting factor is the causal estimate of `g`, not the geometry of the direction.

Candidate follow-up work should stay scalar first: forecast the projection coefficient itself, improve its time/horizon calibration, or replace the fixed EWMA with a causal estimator validated against the anchor trace. Keep the `0.25` safety bound unless real evidence justifies changing it.

### Branch B: add disagreement-controlled trust shrinkage

Prefer this branch when all of the following hold:

1. **Headroom exists:** oracle segment shrinkage materially improves the generic-corrected sampled forecast ratio. A mean relative improvement around or above 2% is enough to justify a real applied experiment.
2. **The observer is informative:** disagreement has a stable positive relationship with current forecast error and/or required shrink. Small-sample correlation from one prompt is supporting evidence, not a universal calibration claim.
3. **The mapping is validated at the live horizon:** at least one fixed theta candidate improves the sampled ratio at the same horizon where it will be applied, without a meaningful regression in the other stream.
4. **A causal mapping survives another seed/prompt:** the selected fixed mapping should remain useful outside the calibration trace.
5. **Applied A/B quality is real:** once an opt-in causal mapping is implemented, compare same-seed generated outputs and forecast telemetry against the current generic-correction baseline. Do not call a feature-space percentage a perceptual-quality percentage.

When the trust branch materially dominates Branch A, as it does in the supplied 25-step horizon-2-heavy trace, it should be tested first after direct-horizon calibration rather than mechanically preferring the smaller scalar-gain change.

If this branch passes, the smallest applied causal patch keeps the PR-39 correction and adds one scalar trust coefficient after it:

```text
w_corrected = existing Spectrum weights + generic latest-delta correction
w_final = kappa * w_corrected + (1 - kappa) * one_hot(latest_causal_anchor)
```

The first applied experiment should operate on the causal first-pass weights only. Offline replay intentionally uses a different, future-bracketed smoother; copying the causal trust coefficient into that smoother would combine two mechanisms and is not justified by the current probe. If better first-pass trust improves the retained exact anchors, that improvement will already propagate into the replay archive.

Hard refresh/re-pay scheduling is intentionally deferred. It changes exact-step placement and interacts with native sampler history, offline replay, ER-SDE tail protection, rollback, and schedule accounting. Continuous trust shrinkage is the lower-regression-risk mechanism to validate first.

## Interpretation map

The shadow telemetry is designed to make the next decision explicit:

- **Large bounded-delta advantage:** improve causal gain estimation around the already-successful PR-39 correction.
- **Large trust oracle advantage + useful disagreement correlation:** calibrate and then test causal disagreement-controlled shrinkage.
- **Large trust oracle advantage + weak disagreement signal:** retain the trust-segment idea and search for a better cache-only observer.
- **Both branches show material headroom:** follow the materially dominant branch after matching the calibration horizon; use implementation size only as a tie-breaker.
- **Small headroom in both:** the PR-39 breach is close to saturated under this local geometry; move to a different forecast representation or coordinate rather than reviving rejected K=2/FinalLayer-adjoint families.
