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

PR #45 investigates the next forecast-quality breach while preserving the successful PR #39 causal mechanism.

## Causal trust result remains supported

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
raw forecast ratio            1.650739    1.307939
PR #39 corrected ratio        1.542742    1.240801
causal oracle ratio           0.987258    0.998710
causal oracle advantage      +34.9660%   +19.3114%
theta=0.15 ratio              0.990551    1.005021
theta=0.15 advantage         +34.7545%   +18.7989%
```

The direct horizon-1 calibration also remained strongly positive at `+30.1357%` audio / `+15.1502%` video for `theta=0.15`.

**Conclusion:** causal trust distance remains a supported failure axis. This result applies to the causal/single-pass geometry.

## Single-pass and offline-replay semantics

Public option:

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

The PR #39 generic replay correction remains present while its replay-specific geometry is investigated.

## Authoritative replay-native run

Configuration:

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
fallbacks                                  0
model_aware_failures                       0
trust_probe_failures                       0
model_aware_trust_failures                 0
model_aware_trust_replay_shadow_failures   0
model_aware_trust_extra_transformer_nfe    0
```

ER-SDE steps 23 and 24 remained exact final-tail steps.

```text
model_aware_trust_total_s                    0.143981
model_aware_trust_replay_shadow_compute_s   0.028194
```

The generated output looked fine. This run is replay-geometry evidence, not an ER-SDE artifact-repair test.

## Closed replay axis: latest causal hold -> current replay

The replay-native LOO oracle tested:

```text
latest causal hold anchor
    ->
validation-attenuated corrected replay proposal
```

Result:

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

The old hold-axis telemetry may remain as a documented negative control, but it is no longer the replay optimization target.

## New replay breach: local endpoint beats current replay

The same LOO run scored the raw future-bracketed local interpolation.

```text
                              audio       video
current replay baseline       0.739193    0.769209
raw local                     0.681730    0.722708
local->current oracle         0.681449    0.718418
oracle advantage             +7.9232%    +6.5366%
local->current kappa mean     0.040767    0.151893
local->current kappa min      0.000000    0.000000
local->current kappa max      0.448442    0.964545
```

For this axis:

```text
p(kappa) = local + kappa * (current_replay - local)
```

so `kappa=0` is pure local and `kappa=1` is the current replay construction.

### Audio diagnosis

Audio replay spectral contribution was exactly zero:

```text
effective_blend_mean = 0.000000
effective_blend_min  = 0.000000
effective_blend_max  = 0.000000
```

Therefore, for audio in this trace:

```text
current replay = local + persisted PR #39 generic replay correction
```

Measured:

```text
local      = 0.681730
corrected  = 0.739193
```

This is direct evidence that the PR #39 scalar mechanism, while useful in its validated causal geometry, is harmful when its persisted gain is transplanted onto the future-bracket replay delta in this audio trace.

Do not remove or weaken the causal PR #39 correction. The replay-specific application is the target of the next investigation.

### Video uncertainty

Video retains a nonzero validation-attenuated spectral branch:

```text
effective_blend_mean = 0.250490
effective_blend_min  = 0.185651
effective_blend_max  = 0.413509
```

Measured:

```text
local                  = 0.722708
current replay         = 0.769209
local->current oracle  = 0.718418
```

The regression currently contains at least two contributions:

1. validation-attenuated spectral/local blending;
2. persisted PR #39 generic replay correction.

Their individual effects are not yet identified. No production video change is justified before decomposition.

## Replay observer status

Video spectral-vs-local disagreement from the authoritative run:

```text
samples                     11
disagreement mean           0.209262
disagreement max            0.314062
disagreement/error corr    +0.282510
```

The old replay-disagreement/shrink and causal-disagreement/shrink correlations used the now-dead hold-axis target `1 - hold_to_replay_oracle_kappa`. Because that oracle kappa is identically `1`, the target has zero variance and is no longer useful.

The relevant target is now the amount of replay enhancement beyond local that should be retained, represented by `local_to_current` oracle kappa or equivalently `1 - kappa` as required local weight.

## Current implementation gate: shadow-only replay component decomposition

No new replay controller is applied in this revision.

The next LOO diagnostic constructs four candidates from the same target-withheld replay cache:

```text
A = local

B = validation-attenuated spectral/local blend
    without generic correction

C = local
    + current generic replay correction

D = validation-attenuated spectral/local blend
    + current generic replay correction

D = current replay baseline
```

For audio, `A == B` because its configured/effective spectral blend is zero. Therefore `A -> C` cleanly isolates the generic replay correction.

For video, all four candidates are measured so the blend, correction, and their interaction can be separated.

### Candidate telemetry

Per stream, the decomposition reports ratios and relative advantage versus the current replay baseline for:

```text
local
blend_uncorrected
local_corrected
blend_corrected
```

The final candidate is the current replay baseline and therefore has zero advantage by definition.

Correction-specific telemetry additionally reports the relative effect of adding the generic replay correction to:

```text
local
blend_uncorrected
```

and a ratio-space interaction term.

### New oracle axes

Shadow-only oracle coefficients are reported for:

```text
local -> blend_uncorrected
local -> local_corrected
blend_uncorrected -> blend_corrected
local -> current replay
```

Each axis reports:

```text
oracle ratio
oracle advantage versus current replay baseline
oracle kappa mean/min/max
```

These oracle coefficients are never applied.

### Retargeted observer correlations

The decomposition correlates causal disagreement and, when available, replay spectral-vs-local disagreement against:

```text
local_to_current oracle kappa
1 - local_to_current oracle kappa
```

Video replay disagreement is also correlated with the isolated:

```text
local -> blend_uncorrected oracle kappa
```

Audio does not fabricate a spectral observer when spectral blend is zero.

### Generic replay-correction direction audit

The persisted causal scalar gain was learned for:

```text
d_causal = latest_exact - previous_exact
```

Offline replay applies it to:

```text
d_replay = right_future_bracket_anchor - left_future_bracket_anchor
```

The decomposition therefore records bounded-sample diagnostics for:

```text
<actual - local, d_replay> / ||d_replay||^2
<actual - blend_uncorrected, d_replay> / ||d_replay||^2

cos(actual - local, d_replay)
cos(actual - blend_uncorrected, d_replay)
cos(d_causal, d_replay)

replay-optimal projection - persisted causal gain
```

This distinguishes a replay gain-magnitude problem from replay-direction misalignment without reviving K=2, FinalLayer, or previous-error correction families.

## Invariants

- default workflow remains unchanged when trust is false;
- single-pass causal trust remains unchanged;
- offline first-pass local-only capture remains unchanged;
- offline replay never applies causal kappa;
- replay decomposition is shadow-only;
- current production replay weights are unchanged by the decomposition;
- PR #39 causal correction remains intact;
- current PR #39 replay correction remains intact until measured decomposition justifies a change;
- ordinary 14/11 scheduling remains unchanged;
- no hard refresh/re-pay is introduced;
- zero trust-added transformer NFE;
- exact replay anchors remain exact;
- audio/video diagnostics remain independent;
- packed topology does not fabricate modality decomposition;
- the scored target anchor is excluded from candidate construction and nested validation histories;
- ordinary diagnostic failures do not abort replay;
- CUDA OOM propagates;
- ER-SDE final two logical steps remain exact;
- replay diagnostic state is archive-scoped.

## Next real gate

Run one ordinary accelerated trace:

```text
steps = 25
sampler = native sample_er_sde
model_aware_mode = full
model_aware_risk_threshold = 0.65
model_aware_trust_shrinkage = true
offline_smoothing_replay = true
```

Keep the same workflow/seed/settings. Do not run another dense force-actual calibration.

Expected mechanics remain:

```text
14 actual / 11 forecast
zero extra transformer NFE
zero trust/replay diagnostic failures
exact ER-SDE tail
model_aware_trust_path=offline_replay_shadow_only
model_aware_trust_applied=0
model_aware_trust_applications=0
```

The run must answer:

1. exactly how much the generic replay correction helps or hurts audio;
2. how much of the video regression comes from spectral blending;
3. how much comes from generic replay correction;
4. whether there is a useful blend/correction interaction;
5. whether replay correction has the wrong scalar magnitude or a misaligned future-bracket direction;
6. whether replay spectral-vs-local disagreement predicts how much departure from local should be retained;
7. whether the next viable mechanism is replay-correction removal/reduction, improved replay-correction geometry, dynamic local/spectral mixture, or a measured combination.

Stop after collecting this decomposition evidence. Do not apply a new replay controller before that result.

`model_aware_trust_shrinkage` remains **false by default** and PR #45 remains **draft**.
