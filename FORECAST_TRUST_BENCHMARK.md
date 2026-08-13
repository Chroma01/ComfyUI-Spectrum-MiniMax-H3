# Forecast trust-region benchmark record

This document records the forecast-quality investigation following PR #39.

## Starting point: PR #39

PR #39 established one useful causal scalar correction on real MiniMax-H3:

```text
d_causal = latest_exact - previous_exact
r = actual - forecast_uncorrected
g = projection of r onto d_causal
corrected = forecast + g * d_causal
```

The gain is confidence-scaled and bounded. Its final same-seed gate reduced measured hidden-feature forecast ratio by about **6.2% audio** and **5.5% video**. Those are hidden-feature forecast-error improvements, not perceptual-quality percentages.

K=2 causal trajectory correction, FinalLayer-transformed directions, previous-error directions, and related retired model-specific correction families remain out of scope.

## Causal trust remains supported

The causal observer compares the global Chebyshev proposal with the local linear/secant proposal on existing bounded cache evidence:

```text
disagreement = RMS(p_spectral - p_linear) / max(RMS(p_spectral), epsilon)

kappa =
    exp(-0.3 * max(horizon - 1, 0))
    * sigmoid(4.0 * (0.15 - disagreement))
```

The calibrated causal segment is:

```text
latest causal exact anchor -> corrected causal proposal
```

Authoritative normal accelerated 25-step native `sample_er_sde` trace:

```text
actual_steps                    14
forecast_steps                  11
actual_transformer_calls        14
model_aware_extra_nfes           0

                              audio       video
raw causal ratio              1.650739    1.307939
PR #39 corrected ratio        1.542742    1.240801
causal oracle ratio           0.987258    0.998710
causal oracle advantage      +34.9660%   +19.3114%
theta=0.15 ratio              0.990551    1.005021
theta=0.15 advantage         +34.7545%   +18.7989%
```

The direct horizon-1 calibration also remained strongly positive at `+30.1357%` audio / `+15.1502%` video for `theta=0.15`.

**Conclusion:** causal trust distance remains a supported failure axis in causal/single-pass geometry.

## Single-pass and offline-replay trust semantics

Public trust option:

```text
model_aware_trust_shrinkage: bool = false
```

It remains default-off and requires `model_aware_mode="full"`.

Single-pass keeps causal trust applied:

```text
offline_smoothing_replay=false
model_aware_trust_path=causal_single_pass
```

Offline replay keeps the rejected causal-kappa transfer disabled:

```text
offline_smoothing_replay=true
model_aware_trust_path=offline_replay_shadow_only
model_aware_trust_applied=0
model_aware_trust_applications=0
model_aware_trust_replay_application=disabled_rejected_causal_transfer
```

## Closed replay axis: latest causal hold -> current replay

The replay-native LOO oracle tested:

```text
latest causal hold anchor
    ->
validation-attenuated corrected replay proposal
```

Authoritative result:

```text
                              audio       video
current replay baseline       0.739193    0.769209
oracle ratio                  0.739193    0.769209
oracle kappa mean             1.000000    1.000000
oracle kappa min              1.000000    1.000000
oracle kappa max              1.000000    1.000000
```

Every scored target selected `kappa=1`: full retention of the replay proposal and zero movement toward the latest causal anchor.

The hold-anchor replay controller direction is therefore closed. Its telemetry remains a negative control only.

## Replay A/B/C/D decomposition

The LOO diagnostic decomposes replay into target-withheld candidates:

```text
A = local

B = validation-attenuated spectral/local blend
    without generic replay correction

C = local
    + transplanted PR #39 generic replay correction

D = validation-attenuated spectral/local blend
    + transplanted PR #39 generic replay correction
```

The authoritative trace used:

```text
sampler = sample_er_sde
steps = 25
model_aware_mode = full
model_aware_risk_threshold = 0.65
model_aware_trust_shrinkage = true
offline_smoothing_replay = true
```

Mechanics passed:

```text
actual_steps                              14
forecast_steps                            11
actual_transformer_calls                  14
model_aware_extra_nfes                     0
model_aware_trust_extra_transformer_nfe    0
fallbacks                                  0
model_aware_failures                       0
trust_probe_failures                       0
model_aware_trust_failures                 0
model_aware_trust_replay_shadow_failures   0
```

ER-SDE steps 23 and 24 remained exact final-tail steps.

### Audio decomposition

```text
A local                  0.681730
B blend_uncorrected      0.681730
C local_corrected        0.739193
D blend_corrected        0.739193

B advantage vs D         +7.8915%
local correction effect  -8.7879%
blend correction effect  -8.7879%
```

Audio spectral blend is zero, so this is a clean isolated measurement of replay-applied generic correction.

`local -> local_corrected` oracle:

```text
oracle ratio          0.681449
oracle kappa mean     0.040767
oracle kappa min      0.000000
oracle kappa max      0.448442
```

Direction audit:

```text
persisted causal gain                              -0.129556
replay-native optimal projection on d_replay       +0.086194
projection minus causal gain                       +0.215750
residual / replay-delta cosine                     +0.184667
causal-delta / replay-delta cosine                 +0.004650
```

### Video decomposition

```text
A local                  0.722708
B blend_uncorrected      0.728863
C local_corrected        0.764653
D blend_corrected        0.769209

B advantage vs D         +5.2181%
local correction effect  -5.8856%
blend correction effect  -5.5538%
interaction ratio delta  -0.001599
```

Correction-specific oracles:

```text
local -> local_corrected
  oracle ratio          0.722707
  oracle kappa mean     0.001675
  oracle kappa max      0.018422

blend_uncorrected -> blend_corrected
  oracle ratio          0.728825
  oracle kappa mean     0.009633
  oracle kappa max      0.105961
```

Direction audit:

```text
persisted causal gain                              -0.108478
replay-native projection from local                +0.108435
replay-native projection from uncorrected blend    +0.092066
residual / replay-delta cosine from local          +0.186146
residual / replay-delta cosine from blend          +0.162314
causal-delta / replay-delta cosine                 -0.098340
```

### Replay-correction conclusion

The PR #39 scalar remains beneficial in the causal geometry in which it was learned:

```text
d_causal = latest_exact - previous_exact
```

Offline replay had been applying that scalar to a different direction:

```text
d_replay = right_future_bracket_anchor - left_future_bracket_anchor
```

The authoritative replay trace shows negative persisted causal gains, positive replay-native residual projections, and weak causal/replay direction alignment. The causal scalar transfer is therefore strongly rejected in replay hidden-feature geometry.

This does **not** reject PR #39's causal correction.

## Applied D -> B gate

Explicit setting:

```text
model_aware_replay_generic_correction: bool = true
```

Semantics:

```text
true  = existing/default D replay
false = experimental B replay; suppress only transplanted generic replay correction
```

The setting remains independent from `model_aware_trust_shrinkage`.

### Mechanical result: PASS

Enabled/default D run:

```text
model_aware_replay_generic_correction_enabled=1
model_aware_replay_generic_correction_path=current_causal_gain_transfer
model_aware_replay_generic_correction_applications=18
model_aware_replay_generic_correction_skips=0
```

Disabled/experimental B run:

```text
model_aware_replay_generic_correction_enabled=0
model_aware_replay_generic_correction_path=disabled_replay_geometry_experiment
model_aware_replay_generic_correction_applications=0
model_aware_replay_generic_correction_skips=18
```

The normal causal trajectory remained unchanged:

```text
25 logical steps
14 actual
11 forecast
14 actual transformer calls
0 extra model-aware NFE
0 extra trust NFE
0 replay-gate extra NFE
0 fallbacks
exact ER-SDE final tail
```

Causal PR #39 metrics remained numerically unchanged. The A/B isolation therefore succeeded: only offline replay generic-correction application changed.

### Feature-space result

LOO hidden-feature evidence substantially favors B over D:

```text
audio B advantage vs D   +7.8915%
video B advantage vs D   +5.2181%
```

The replay-geometry diagnosis remains supported.

### Perceptual result

The same-seed generated D and B outputs were judged **indistinguishable** by the user:

```text
no obvious visible improvement
no obvious audible improvement
no obvious visible regression
no obvious audible regression
```

This is a neutral end-to-end result on this sample.

It is scientifically incorrect to promote the feature-space gain to a perceptual-quality claim, and equally incorrect to treat the neutral perceptual result as disproving the replay-geometry diagnosis.

### Current replay-generic-correction status

For now:

```text
model_aware_replay_generic_correction=true
```

remains the backward-compatible default.

The `false` B path remains available as the measured experimental candidate. One perceptually neutral sample is insufficient to flip the default.

## Next research choice: video local/spectral replay mixture

The highest-information next step is **shadow-only instrumentation of video replay local/spectral geometry**, not another applied controller.

### Why this direction

Current video result:

```text
local                         0.722708
current uncorrected blend     0.728863
oracle local->current blend   0.712966
oracle advantage              +7.2693%
oracle kappa mean             0.376027
oracle kappa min              0
oracle kappa max              1
```

Existing observer correlations:

```text
causal disagreement
  vs local->current oracle kappa                +0.688708

replay spectral/local disagreement
  vs local->current oracle kappa                +0.598033

replay spectral/local disagreement
  vs local->current-blend oracle kappa          +0.586401
```

This is qualitatively different from generic replay correction: the correction-specific oracle wants approximately zero coefficient, while the spectral contribution is heterogeneous across the full measured segment.

### Important code-level limitation in the old oracle

The existing axis is:

```text
local -> blend_uncorrected
```

where `blend_uncorrected` is **already validation-attenuated** and may use branch-specific effective blend weights.

Therefore:

```text
local->blend oracle kappa = 1
```

means "use all of the current attenuated blend endpoint", **not** "use full spectral weight = 1".

The old `0..1` oracle range cannot by itself identify the absolute optimal spectral share. A production controller fitted directly to that kappa would mix two different quantities: validation attenuation and spectral preference.

That missing geometric distinction is the reason to instrument before applying anything.

## New video spectral-mixture shadow

No production replay mechanism is changed.

The new shadow evaluates the direct segment:

```text
local -> full spectral
```

on target-withheld LOO replay evidence.

It reports:

```text
model_aware_trust_replay_spectral_mixture=video_local_to_full_spectral_shadow
model_aware_trust_replay_spectral_mixture_applied=0
model_aware_trust_replay_spectral_mixture_baseline=uncorrected_validation_attenuated_blend
```

### Direct absolute oracle

For video:

```text
model_aware_trust_replay_spectral_video_local_ratio_mean
model_aware_trust_replay_spectral_video_current_blend_ratio_mean
model_aware_trust_replay_spectral_video_full_spectral_ratio_mean
model_aware_trust_replay_spectral_video_oracle_ratio_mean
model_aware_trust_replay_spectral_video_oracle_advantage_vs_local_mean
model_aware_trust_replay_spectral_video_oracle_advantage_vs_current_mean
model_aware_trust_replay_spectral_video_oracle_weight_mean
model_aware_trust_replay_spectral_video_oracle_weight_min
model_aware_trust_replay_spectral_video_oracle_weight_max
```

Here `oracle_weight` is an **absolute scalar weight on the full local->spectral direction**.

### Current replay-placement diagnostics

The shadow also reports:

```text
model_aware_trust_replay_spectral_video_current_weight_projection_mean
model_aware_trust_replay_spectral_video_current_weight_projection_min
model_aware_trust_replay_spectral_video_current_weight_projection_max
model_aware_trust_replay_spectral_video_effective_blend_mean
model_aware_trust_replay_spectral_video_validation_penalty_mean
```

`current_weight_projection` projects the actual validation-attenuated blend displacement onto the same full local->spectral direction as the oracle. This makes the current placement and target weight geometrically comparable.

`validation_penalty` is reconstructed from the exact replay attenuation relation:

```text
effective_blend = configured_blend / max(1, validation_score)
```

so the shadow measures the penalty that actually affects production replay rather than introducing another validator.

### Correction-free spectral gap

The new observer:

```text
model_aware_trust_replay_spectral_video_spectral_gap_mean
```

uses:

```text
RMS(full_spectral - local) / max(RMS(local), epsilon)
```

It deliberately excludes the rejected generic correction from its normalization.

### Predeclared fixed-weight sweep

To distinguish a genuinely dynamic problem from a simpler globally mis-set blend, the shadow evaluates a fixed, predeclared absolute spectral-weight sweep:

```text
0.00
0.25
0.50
0.75
1.00
```

For each weight it reports mean ratio plus advantage versus local and versus the current validation-attenuated blend.

This sweep is diagnostic only. No coefficient is applied to production replay.

### Predictor correlations

The shadow compares the direct absolute oracle weight and the required adjustment:

```text
required_adjustment = oracle_weight - current_weight_projection
```

against cheap pre-target observables:

```text
causal disagreement
correction-free spectral gap
validation penalty
current spectral-weight projection
step coordinate
```

No multivariate model is fitted and no threshold is tuned on the 11-target trace.

## Hypothesis and falsification

Primary hypothesis:

```text
Video replay contains real per-target local/spectral mixture headroom,
and at least one existing cheap replay observable carries repeatable
information about the absolute optimal spectral weight or about the
adjustment required from the current validation-attenuated placement.
```

The direction is falsified or deprioritized if an independent trace shows one or more of the following:

- direct `local -> full spectral` oracle headroom collapses;
- oracle weights collapse close to the current projected blend weights;
- the fixed absolute-weight sweep shows no stable improvement over local/current blend;
- the previously promising disagreement relationships disappear or reverse without another stable predictor taking their place;
- apparent correlations are dominated by coordinate alone, indicating schedule position rather than a reusable replay-quality signal.

If a single fixed spectral weight consistently wins, prefer that simpler explanation before considering a dynamic controller.

If the absolute oracle remains heterogeneous and a cheap observable relationship repeats across independent traces, only then design a narrowly applied controller gate.

## Why not the alternatives yet

### Another D/B perceptual validation

A second D/B pair is low-risk, but its information value is now lower than one shadow-calibration run. The first applied gate already proved isolation and produced a neutral perceptual result. Another paired D/B sample would mainly add one more subjective endpoint while providing no new mechanism for the remaining replay headroom.

The new shadow run still uses an independent latent trajectory and therefore also tests whether the feature-space spectral evidence survives another case, without requiring two expensive generations.

### Replay-native positive correction

The measured replay residual projections around `+0.09..+0.11` are interesting, but they are one-trace averages. Hardcoding `+0.1` would be unjustified, and a new correction-direction calibration would compete with the already-observed heterogeneous spectral headroom.

Positive replay correction remains a later candidate if the spectral mixture direction fails.

## Next real run

Use **one independent 25-step case** with no new applied replay controller.

Keep the same prompt, references, resolution, frame count, scheduler, CFG, checkpoint/precision, and remaining workflow settings as the authoritative 25-step trace. Change only the seed to a different fixed seed.

Use:

```text
sampler = sample_er_sde
steps = 25
model_aware_mode = full
model_aware_risk_threshold = 0.65
model_aware_trust_shrinkage = true
offline_smoothing_replay = true
model_aware_replay_generic_correction = true
```

The replay-generic-correction setting is returned to its backward-compatible default because this next gate is shadow-only spectral research. The spectral shadow internally evaluates uncorrected local/blend/spectral candidates, so the production D/B choice is not part of the hypothesis.

Expected mechanics remain:

```text
actual_steps=14
forecast_steps=11
actual_transformer_calls=14
fallbacks=0
model_aware_extra_nfes=0
model_aware_trust_extra_transformer_nfe=0
model_aware_replay_generic_correction_extra_transformer_nfe=0
model_aware_trust_replay_shadow_failures=0
ER-SDE steps 23/24 exact
11 offline smoothed replay steps
model_aware_trust_replay_spectral_mixture_applied=0
```

The decisive output is the new `model_aware_trust_replay_spectral_video_*` telemetry, especially:

```text
oracle advantage vs local/current
absolute oracle-weight mean/min/max
current projected spectral weight
fixed 0/0.25/0.5/0.75/1.0 sweep
causal-disagreement correlations
spectral-gap correlations
validation-penalty correlations
coordinate correlations
```

No dynamic replay spectral controller, replay-native positive correction, hold-anchor shrinkage, hard refresh, or extra transformer evaluation is introduced before that evidence exists.

`model_aware_trust_shrinkage` remains **false by default**. `model_aware_replay_generic_correction` remains **true by default**. PR #45 remains **draft**.
