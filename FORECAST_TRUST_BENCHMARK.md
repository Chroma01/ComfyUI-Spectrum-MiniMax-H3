# Forecast trust-region benchmark record

This document records the forecasting-quality investigation that follows PR #39.

## Starting point: PR #39

PR #39 established one useful causal scalar correction mechanism on real MiniMax-H3:

```text
d_causal = latest_exact - previous_exact
r = actual - forecast_uncorrected
g = projection of r onto d_causal
corrected = forecast + g * d_causal
```

The gain is confidence-scaled and bounded. The final same-seed PR #39 gate reduced measured hidden-feature forecast ratio by about **6.2% for audio** and **5.5% for video** relative to the uncorrected forecast. These are hidden-feature forecast-error improvements, not perceptual-quality percentages.

K=2 causal trajectory correction, FinalLayer-transformed directions, previous-error directions, and related model-specific correction families remain retired.

PR #45 investigates subsequent forecast-quality breaches while preserving the successful PR #39 **causal** mechanism.

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

Normal accelerated 25-step native `sample_er_sde` trace:

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

**Conclusion:** causal trust distance remains a supported failure axis. This result applies to causal/single-pass geometry.

## Single-pass and offline-replay trust semantics

Public trust option:

```text
model_aware_trust_shrinkage: bool = false
```

It remains default-off and requires `model_aware_mode="full"`.

### Single-pass

With:

```text
offline_smoothing_replay=false
```

causal trust remains applied:

```text
model_aware_trust_path=causal_single_pass
```

### Offline replay

With:

```text
offline_smoothing_replay=true
```

the first pass remains local-only and the rejected causal-kappa replay transfer remains disabled:

```text
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

Fixed-kappa sweep:

```text
AUDIO
kappa=0.50   ratio=0.853396   advantage=-16.0403%
kappa=0.70   ratio=0.802909   advantage= -8.9545%
kappa=0.85   ratio=0.769044   advantage= -4.1964%
kappa=0.95   ratio=0.748662   advantage= -1.3313%
kappa=1.00   ratio=0.739193   advantage=  0

VIDEO
kappa=0.50   ratio=0.858995   advantage=-11.8148%
kappa=0.70   ratio=0.815726   advantage= -6.1206%
kappa=0.85   ratio=0.789490   advantage= -2.6678%
kappa=0.95   ratio=0.775276   advantage= -0.7978%
kappa=1.00   ratio=0.769209   advantage=  0
```

**Conclusion:** offline replay should not be shrunk toward the latest causal anchor for the current replay geometry. No further observer tuning is justified for this exact segment because its measured oracle optimum is identically `kappa=1`.

The old hold-axis telemetry remains useful as a documented negative control, not as a replay controller candidate.

## Replay component decomposition

The next LOO diagnostic isolated the replay construction into four target-withheld candidates:

```text
A = local

B = validation-attenuated spectral/local blend
    WITHOUT generic replay correction

C = local
    + transplanted PR #39 generic replay correction

D = validation-attenuated spectral/local blend
    + transplanted PR #39 generic replay correction

D = current production replay baseline
```

The run used:

```text
sampler = sample_er_sde
steps = 25
model_aware_mode = full
model_aware_risk_threshold = 0.65
model_aware_trust_shrinkage = true
offline_smoothing_replay = true
```

Same workflow/seed/settings as the preceding traces.

### Mechanics: PASS

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

Offline replay remained diagnostic-only for trust:

```text
model_aware_trust_path=offline_replay_shadow_only
model_aware_trust_applied=0
model_aware_trust_applications=0
model_aware_trust_replay_application=disabled_rejected_causal_transfer
```

Replay decomposition was active:

```text
model_aware_trust_replay_decomposition=loo_component_geometry_shadow
model_aware_trust_replay_decomposition_baseline=blend_corrected_current_replay
model_aware_trust_replay_decomposition_compute_s=0.031472
```

ER-SDE steps 23 and 24 remained exact final-tail steps. No schedule or transformer-NFE change occurred.

## Authoritative A/B/C/D result

### Audio

```text
A local                  0.681730
B blend_uncorrected      0.681730
C local_corrected        0.739193
D blend_corrected        0.739193
```

Audio spectral blend is exactly zero, so:

```text
A == B
C == D
```

The transplanted generic replay correction therefore has a clean isolated effect:

```text
0.681730 -> 0.739193

audio_local_correction_advantage_mean = -0.087879
audio_blend_correction_advantage_mean = -0.087879
```

This is about **8.8% worse** relative to the uncorrected replay candidate on these LOO hidden-feature measurements.

The oracle on `local -> local_corrected` reports:

```text
oracle ratio          0.681449
oracle kappa mean     0.040767
oracle kappa min      0.000000
oracle kappa max      0.448442
```

The current replay correction therefore wants almost zero weight on average.

### Audio direction audit

```text
persisted causal gain                              -0.129556
replay-native optimal projection on d_replay       +0.086194
projection minus causal gain                       +0.215750
residual / replay-delta cosine                     +0.184667
causal-delta / replay-delta cosine                 +0.004650
```

The causal gain is negative while the replay-native residual wants a small positive projection along the replay bracket direction. The causal and replay directions are essentially orthogonal.

This is not just an over-strong scalar. The scalar sign and the geometry on which it was learned do not transfer.

### Video

```text
A local                  0.722708
B blend_uncorrected      0.728863
C local_corrected        0.764653
D blend_corrected        0.769209
```

Generic replay correction applied to local:

```text
0.722708 -> 0.764653
local_correction_advantage_mean = -0.058856
```

Generic replay correction applied after spectral blending:

```text
0.728863 -> 0.769209
blend_correction_advantage_mean = -0.055538
```

The ratio-space interaction is tiny:

```text
correction_blend_interaction_ratio_delta_mean = -0.001599
```

The replay correction is therefore the dominant negative component, not a pathological blend/correction interaction.

### Video correction oracles

`local -> local_corrected`:

```text
oracle ratio          0.722707
oracle kappa mean     0.001675
oracle kappa min      0.000000
oracle kappa max      0.018422
```

`blend_uncorrected -> blend_corrected`:

```text
oracle ratio          0.728825
oracle kappa mean     0.009633
oracle kappa min      0.000000
oracle kappa max      0.105961
```

Both correction-specific replay oracles are essentially zero.

### Video direction audit

```text
persisted causal gain                              -0.108478
replay-native projection from local                +0.108435
replay-native projection from uncorrected blend    +0.092066
residual / replay-delta cosine from local          +0.186146
residual / replay-delta cosine from blend          +0.162314
causal-delta / replay-delta cosine                 -0.098340
```

Again, the causal coefficient is negative while the replay residual wants a positive projection. The replay bracket direction is only weakly and negatively related to the causal latest-delta direction.

## Precise correction conclusion

The PR #39 scalar correction remains beneficial in the causal geometry in which it was measured and calibrated.

Its **offline replay implementation is not geometry-preserving**:

```text
d_causal = latest_exact - previous_exact
```

is the direction on which the scalar was learned, while replay applies the persisted scalar to:

```text
d_replay = right_future_bracket_anchor - left_future_bracket_anchor
```

In the authoritative replay trace, the persisted gains are negative while replay-native residual projections on `d_replay` are positive, and `d_causal` is weakly related to `d_replay`.

Therefore the next applied gate removes only this transplanted replay transfer. It does **not** remove or weaken PR #39's causal correction.

## Spectral replay result remains separate

Video's current uncorrected blend is slightly worse on average than pure local:

```text
A local              0.722708
B blend_uncorrected  0.728863
```

That does not reject the spectral branch. The oracle on:

```text
local -> blend_uncorrected
```

reports:

```text
oracle ratio          0.712966
oracle advantage      +7.2693%
oracle kappa mean     0.376027
oracle kappa min      0.000000
oracle kappa max      1.000000
```

Unlike the generic replay correction, the spectral contribution has heterogeneous useful signal: some targets want local, some want the full current blend, and intermediate targets exist.

No dynamic spectral controller is applied in the immediate D -> B gate.

## Observer signals retained for later work

Video correlations from the decomposition:

```text
causal disagreement
  vs local->current oracle kappa                         +0.688708

replay spectral-vs-local disagreement
  vs local->current oracle kappa                         +0.598033

replay spectral-vs-local disagreement
  vs local->blend oracle kappa                           +0.586401
```

These are interesting but come from only 11 LOO targets in one case. They are not production mappings.

The sign is also replay-specific: in this trace, larger disagreement correlates with wanting **more** of the enhanced/spectral proposal. The causal RACER-style interpretation `more disagreement -> less trust` must not be transplanted into replay geometry.

## Applied D -> B experiment

A new explicit configuration setting controls only the generic scalar correction in smoothed offline replay:

```text
model_aware_replay_generic_correction: bool = true
```

Default `true` preserves the existing D replay path exactly.

Experimental `false`, when used with `model_aware_mode="full"` and `offline_smoothing_replay=true`, changes only:

```text
D = blend_uncorrected + transplanted generic replay correction
```

to:

```text
B = blend_uncorrected
```

It does not change:

- first-pass scheduling;
- causal PR #39 correction;
- causal trust single-pass behavior;
- archive contents;
- validation attenuation;
- local interpolation;
- spectral interpolation;
- audio/video blend weights;
- exact replay anchors;
- ER-SDE tail policy;
- transformer NFE.

The implementation preserves the archived first-pass decisions verbatim. During replay-weight construction only, the disabled gate presents the existing replay builder with a temporary decision view in which the one-scalar generic correction is absent. The original archive is restored even if replay building raises. A/B/C/D diagnostics use their independently persisted shadow records, so D remains available as a counterfactual while production uses B.

### Applied-gate telemetry

```text
model_aware_replay_generic_correction_enabled=0/1
model_aware_replay_generic_correction_path=current_causal_gain_transfer
model_aware_replay_generic_correction_path=disabled_replay_geometry_experiment
model_aware_replay_generic_correction_applications=<count>
model_aware_replay_generic_correction_skips=<count>
model_aware_replay_generic_correction_extra_transformer_nfe=0
```

For the experimental B run, the expected path is:

```text
model_aware_replay_generic_correction_enabled=0
model_aware_replay_generic_correction_path=disabled_replay_geometry_experiment
model_aware_replay_generic_correction_applications=0
model_aware_replay_generic_correction_skips>0
model_aware_replay_generic_correction_extra_transformer_nfe=0
```

Existing causal model-aware telemetry should remain nonzero where it was nonzero before, proving that only replay application was suppressed.

## Invariants for the applied gate

- `model_aware_replay_generic_correction=true` is baseline-identical to the previous/current replay path;
- `model_aware_replay_generic_correction=false` is replay-only and does not alter causal weight construction;
- the new setting is independent of `model_aware_trust_shrinkage`;
- outside full offline replay, the setting has no algorithmic effect;
- first-pass scheduling remains unchanged;
- no extra transformer evaluation is added;
- exact replay anchor steps remain exact;
- validation attenuation and local/spectral blending remain unchanged;
- archived model-aware decisions remain unchanged;
- A/B/C/D shadow decomposition remains valid in either production mode;
- packed topology remains safe;
- ordinary diagnostic failures do not corrupt production replay state;
- CUDA OOM propagates;
- archive-scoped gate state is reset on the next capture;
- ER-SDE final two logical steps remain exact.

## Next real gate

The existing same-seed D output is already the baseline. The next expensive run is the B candidate only:

```text
sampler = sample_er_sde
steps = 25
model_aware_mode = full
model_aware_risk_threshold = 0.65
model_aware_trust_shrinkage = true
offline_smoothing_replay = true
model_aware_replay_generic_correction = false
```

Use the same seed, prompt, references, resolution, scheduler, CFG, checkpoint/precision, and remaining workflow settings.

Expected mechanics:

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
```

The scientific question is deliberately narrow:

```text
Does removing only the transplanted generic correction from offline replay
improve, preserve, or harm the actual same-seed generated output while
preserving all speed, NFE, scheduling, blend, and tail invariants?
```

Do not add a positive replay-native correction or dynamic spectral controller before this A/B result.

The A/B/C/D feature-ratio predictions are evidence for selecting the experiment, not perceptual-quality percentages.

`model_aware_trust_shrinkage` remains **false by default**. `model_aware_replay_generic_correction` remains **true by default**. PR #45 remains **draft**.
