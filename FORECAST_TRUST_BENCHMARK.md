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

The probe then measures two questions against the exact anchor that is already being computed.

### 1. Does the trust segment contain additional error-reduction headroom?

Let `a` be the latest exact cached feature. The best diagnostic interpolation on the segment from `a` to `p` is:

```text
u = p - a
kappa_oracle = clamp(<h_actual-a,u> / <u,u>, 0, 1)
p_oracle = a + kappa_oracle*u
```

`oracle_advantage_mean` measures how much lower the sampled forecast ratio could become if a perfect trust coefficient were known. This does not leak into generation; it only answers whether trust shrinkage is worth pursuing after the existing scalar correction.

### 2. Does a cache-only disagreement signal point in the useful direction?

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

### Gate B: calibration-quality trace

A stronger calibration run should turn would-be forecasts into exact evaluations so the probe sees short-horizon ground truth at the actual locations. With the same `full` configuration, set:

```text
model_aware_risk_threshold = 0.0
```

The existing force-actual rule then converts forecast candidates into exact steps once the model-aware path is active. This run is intentionally slower and is for calibration evidence only; it is not a shipping configuration.

Verify that the probe's reported `horizon_mean` is near the short forecast horizon being calibrated and that sample count is materially larger than in the normal accelerated gate.

## Promotion gate for an applied trust controller

Do not alter `full` from this probe alone. Promote a trust controller only if real H3 evidence establishes all of the following:

1. **Headroom exists:** oracle segment shrinkage materially improves the generic-corrected sampled forecast ratio. A mean relative improvement around or above 2% is enough to justify a real applied experiment.
2. **The observer is informative:** disagreement has a stable positive relationship with current forecast error and/or required shrink. Small-sample correlation from one prompt is supporting evidence, not a universal calibration claim.
3. **A causal mapping survives another seed/prompt:** at least one fixed theta candidate improves the sampled ratio without a meaningful regression in the other stream.
4. **Applied A/B quality is real:** once a causal mapping is implemented, compare same-seed generated outputs and forecast telemetry against the current generic-correction baseline. Do not call a feature-space percentage a perceptual-quality percentage.

## Intended Phase 2 if the gate passes

The smallest applied patch keeps the successful PR-39 correction and adds one scalar trust coefficient after it:

```text
w_corrected = existing Spectrum weights + generic latest-delta correction
w_final = kappa * w_corrected + (1 - kappa) * one_hot(latest_anchor)
```

This applies the trust region directly in history-weight space, so it does not require a second full-feature forecast tensor. `kappa` must be computed from bounded sampled evidence before the forecast, persisted in the per-step offline decision, and replayed exactly. The generated-path change must remain zero-extra-transformer-NFE.

Hard refresh/re-pay scheduling is intentionally deferred. It changes exact-step placement and interacts with native sampler history, offline replay, ER-SDE tail protection, rollback, and schedule accounting. Continuous trust shrinkage is the lower-regression-risk mechanism to validate first.

## Interpretation map

The shadow telemetry is designed to make the next decision unambiguous:

- **Large oracle advantage + useful disagreement correlation:** implement causal disagreement-controlled shrinkage.
- **Large oracle advantage + weak disagreement signal:** retain the trust-segment idea and search for a better cache-only observer.
- **Small trust-segment headroom + remaining scalar-direction headroom:** improve gain prediction for the existing latest-delta correction rather than changing direction rank.
- **Small headroom in both:** the PR-39 breach is close to saturated under this local geometry; move to a different forecast representation or coordinate rather than reviving rejected K=2/FinalLayer-adjoint families.
