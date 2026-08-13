# Spectrum MiniMax H3 v0.2.8

v0.2.8 finalizes the native ER-SDE replay work and closes the forecast-trust/replay-calibration research cycle with the current user-facing evidence.

## Native ER-SDE replay compatibility

Native ComfyUI `SamplerER_SDE` replay-safety detection now accepts the reviewed deterministic native `noise_scaler` closures used by upstream ER-SDE while continuing to fail closed for arbitrary/custom/stateful scalers. An explicitly supplied custom `noise_sampler` remains replay-unsafe. Unknown future closure contracts remain replay-unsafe until reviewed.

This restores the intended two-pass offline replay path for native/default ER-SDE without broadening custom stochastic replay support.

## Narrowed ER-SDE terminal replay safeguard

The v0.2.7 ER-SDE fix enforced a blanket minimum two-step actual tail. The reproduced failure class is narrower.

The 25-step failure ended as:

```text
22 actual
23 forecast
24 actual
```

Offline replay then reconstructed the penultimate feature across the final nonlinear bracket. v0.2.8 promotes the penultimate ER-SDE step only during offline capture when the normal runtime schedule would otherwise forecast it. The protected failing case becomes:

```text
22 actual
23 actual
24 actual
```

Normal 20-step and 32-step ER-SDE schedules already make their penultimate step actual, so they receive no additional transformer evaluation. Explicitly larger configured tails still take precedence. RES retains its independent three-step protected tail. Euler, RES/CFG++, and Turbo policies are unchanged.

## Generic causal correction retained

The generic latest-delta causal correction from PR #39 remains the supported correction used by `model_aware_mode=full`.

In a controlled same-seed native ER-SDE 20-step comparison, `schedule_confidence` and `full` both used exactly:

```text
11 actual steps
9 forecast steps
11 actual transformer calls
0 model-aware extra NFEs
```

`full` changed the measured hidden forecast ratios from:

```text
audio: 1.777636 -> 1.670690   (~6.02% hidden-error reduction)
video: 1.313055 -> 1.250087   (~4.80% hidden-error reduction)
```

The recurring false eye-motion artifact that remained subtly visible with `schedule_confidence` was absent with `full`. The eyes remained synchronized with the intended eye-closing motion. The hidden-error percentages describe this specific feature-space trace; they are not perceptual-quality percentages.

Earlier controlled native ER-SDE testing also isolated a pronunciation case where `full` retained the intended German pronunciation while `schedule_confidence` sounded like the base path.

These perceptual conclusions are established for the tested native ER-SDE configuration. They are not generalized to Euler, RES/RES CFG++, Turbo/LightX2V, or other samplers.

## Trust shrinkage closed for production promotion

`model_aware_trust_shrinkage` remains:

```text
false
```

Causal trust shrinkage produced substantial hidden-feature observer improvements, then failed to show a reliable user-facing perceptual benefit across the completed A/B gate. It remains available for research/reproduction and saved-workflow compatibility. It is not recommended for production promotion in this release.

No additional kappa tuning, denser search, or replacement trust formula is introduced.

## Replay generic correction remains disabled

`model_aware_replay_generic_correction` remains:

```text
false
```

Independent replay traces rejected transfer of the causal PR #39 scalar onto the different future-bracket replay direction. The causal correction remains unchanged in its supported causal geometry. The `true` replay setting remains a legacy/scientific-ablation path.

## Offline replay remains supported

Offline smoothing replay remains the compatibility-safe global default and retains its historical MiniMax H3 audio/stutter rationale.

In the final controlled native ER-SDE eye-motion comparison, `full` single-pass produced the best temporal facial behavior among the tested paths. The replay run still showed the abnormal eye motion, used the same 11 first-pass transformer evaluations, added a transformer-free replay pass of about 13.25 seconds in that trace, and retained about 4315.5 MiB of archive data.

This is an ER-SDE-scoped quality result. The release does not change the global replay default or infer the same ranking for other samplers.

## Replay calibration research infrastructure

PR #45 retains the research-only infrastructure needed to evaluate future replay calibration hypotheses without adding a runtime controller:

- exact VIDEO per-target quadratic moments;
- exact runtime ratio normalization/parity checks;
- source/config/schedule/topology/trace provenance and fingerprints;
- structural interior-target validation;
- CPU-only offline evaluator;
- whole-run cross-validation;
- fixed Level 0 controls;
- affine current-weight Level 1 baseline;
- one-predictor Level 2 residual tests with coordinate control.

No affine, disagreement, validation-penalty, coordinate, floor, interaction, tree, neural, AutoML, or other new applied replay controller is introduced.

The following research branches remain closed as complete runtime solutions: pure global alpha, causal-kappa replay transfer, hold-anchor replay shrinkage, causal scalar replay transfer, K=2, FinalLayer-transformed correction, and previous-error direction families.

## Compatibility and defaults

Shipping defaults remain compatibility-safe:

```text
model_aware_mode = off
model_aware_trust_shrinkage = false
model_aware_replay_generic_correction = false
offline_smoothing_replay = true
```

For current native ER-SDE quality testing, the preferred tested configuration is:

```text
model_aware_mode = full
model_aware_trust_shrinkage = false
offline_smoothing_replay = false
```

That recommendation is ER-SDE-specific. Existing saved input values continue to be honored.

## Validation

The release is validated against the repository's pinned four-revision ComfyUI compatibility matrix. The matrix builds the wheel, runs the forecaster smoke test, scoped Ruff, `compileall`, and the complete pytest suite against each reviewed native MiniMax H3 revision. Focused coverage includes the narrowed ER-SDE terminal policy, native ER-SDE replay guard, generic correction, trust-off/default semantics, replay-generic-correction-off semantics, exact replay calibration and evaluator, provenance/fingerprints, and saved configuration defaults.
